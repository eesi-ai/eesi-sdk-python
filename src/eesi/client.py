"""The entry point: credentials, characters, voices, memory and audio.

::

    from eesi import NurClient

    nur = NurClient()                      # EESI_API_KEY, EESI_API_URL from the environment
    kael = nur.npc(name="Kael", persona="An elven merchant who distrusts kings.", voice="Atlas", memory=True)
    await kael.start()                     # talk through this computer's microphone
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from eesi_sdk.errors import ApiError

from . import audio as _audio
from ._version import __version__
from .character import Character
from .errors import ConfigurationError, EESIError
from .memory import InMemoryStore, LocalMemoryStore, Memory, MemoryPolicy

log = logging.getLogger("eesi")

__all__ = [
    "ClientSecret",
    "NurClient",
    "PhoneCall",
    "PhoneNumber",
    "Voice",
    "VoiceDesign",
    "WebhookVerificationError",
    "verify_webhook",
]

DEFAULT_BASE_URL = "https://api.eesi.ai"
DEFAULT_MODEL = "nur-live-v1"
DEFAULT_MEMORY_DIR = ".eesi/memory"
#: A sentence a designed voice is frozen from; it becomes the clone's ref_text.
_DESIGN_SENTENCES = {
    "en": "Every voice carries a story, and how we say it matters.",
    "ar": "لكل صوت حكاية، وطريقة قولها لا تقل أهمية عن الكلمات.",
}


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    is_builtin: bool
    status: str = "ready"
    language: str | None = None
    gender: str | None = None
    category: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class VoiceDesign:
    """A voice described in words, created once in your organization's library and reused by name.

    ``description`` uses the voice-design vocabulary (``GET /v1/audio/voice-design``),
    e.g. ``"male, elderly, low pitch, british accent"``.
    """

    name: str
    description: str
    language: str = "en"


@dataclass(frozen=True)
class ClientSecret:
    """A credential an untrusted client (a browser, a game build) opens one session with.

    ``url`` is the WebSocket to open as-is; it carries ``value``. The session
    it opens is already the character, with its memory and context, and the
    client cannot change what ``lock`` names.
    """

    value: str
    expires_at: float
    url: str
    session: dict[str, Any]
    #: What the client may not change: instructions, tools, voice,
    #: turn_detection, context, conversation.
    lock: tuple[str, ...] = ("instructions", "tools")
    #: The server control channel, when minted with ``control=``: steer the
    #: session from your server with ``character.control(secret)``.
    control_id: str | None = None
    #: Who answers tool calls with a control channel: ``"server"`` or ``"client"``.
    control_tools: str | None = None
    #: Who the session is for, so ``character.control(secret)`` knows.
    player_id: str | None = None
    player_name: str | None = None
    #: What to send the player's client, as JSON: the TypeScript SDK's
    #: ``joinSession(join)`` takes it as it is (the URL, who greets, who runs
    #: tools, how replies are asked for). Never holds the key or the control id.
    join: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhoneCall:
    """A call Nur placed: where it stands, and once it is over, what was said."""

    id: str
    session_id: str
    #: ``dialing``, ``in_progress``, ``completed`` or ``not_answered``.
    status: str
    to: str
    created_at: float
    duration_seconds: float | None = None
    #: ``(role, text)`` lines once the call has ended: ``person`` or ``nur``.
    transcript: tuple[tuple[str, str], ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in ("completed", "not_answered")

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PhoneCall:
        return cls(
            id=str(data.get("id") or ""),
            session_id=str(data.get("session_id") or ""),
            status=str(data.get("status") or ""),
            to=str(data.get("to") or ""),
            created_at=float(data.get("created_at") or 0),
            duration_seconds=data.get("duration_seconds"),
            transcript=tuple(
                (str(turn.get("role") or ""), str(turn.get("text") or ""))
                for turn in (data.get("transcript") or [])
                if isinstance(turn, dict)
            ),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PhoneNumber:
    """A number Nur may call on the shared EESI line, or one whose code is on its way."""

    number: str
    #: ``confirmed``, or ``code_sent`` while the code Nur phoned in waits to be entered.
    status: str
    confirmed_at: float | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PhoneNumber:
        return cls(
            number=str(data.get("number") or ""),
            status=str(data.get("status") or ""),
            confirmed_at=data.get("confirmed_at"),
        )


class _MemoryFactory:
    def __init__(self, client: NurClient) -> None:
        self._client = client
        self._stores: dict[str, LocalMemoryStore] = {}

    def local(
        self,
        path: str | Path | None = None,
        *,
        policy: MemoryPolicy | None = None,
        namespace: str = "default",
    ) -> Memory:
        """Memory in a SQLite file on this machine (``path`` is a directory or a ``.db`` file)."""
        target = str(Path(path or self._client.memory_dir).expanduser().resolve())
        store = self._stores.get(target)
        if store is None:
            store = self._stores[target] = LocalMemoryStore(target)
        return Memory(store, policy=policy, namespace=namespace)

    def in_memory(
        self, *, policy: MemoryPolicy | None = None, namespace: str = "default"
    ) -> Memory:
        """Memory that lasts as long as this process: tests, demos, guests."""
        return Memory(InMemoryStore(), policy=policy, namespace=namespace)

    def default(self) -> Memory:
        return self.local(self._client.memory_dir)

    async def close(self) -> None:
        for store in self._stores.values():
            await store.close()
        self._stores.clear()


class _AudioFactory:
    """``nur.audio.local()``, ``nur.audio.wav_input(path)``, ``nur.audio.wav_output(path)``."""

    local = staticmethod(_audio.local)
    wav_input = _audio.WavInput
    wav_output = _audio.WavOutput


class NurClient:
    """Characters on Nur Live.

    ``api_key`` defaults to ``EESI_API_KEY`` and ``base_url`` to
    ``EESI_API_URL`` (``https://api.eesi.ai``). Keep the key on a machine you
    control; give browsers and game clients :meth:`client_secret` instead.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        timezone: str | None = None,
        memory_dir: str | Path | None = None,
        http_timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("EESI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "No API key: set EESI_API_KEY or pass NurClient(api_key=...). Create one on the API keys page of your EESI console."
            )
        base = (base_url or os.environ.get("EESI_API_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        parts = urlsplit(base)
        if parts.scheme not in ("http", "https", "ws", "wss") or not parts.netloc:
            raise ConfigurationError(
                f"base_url must look like https://api.eesi.ai, not {base!r}."
            )
        http_scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
        ws_scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
        self.base_url = urlunsplit((http_scheme, parts.netloc, parts.path, "", ""))
        self.realtime_url = urlunsplit(
            (ws_scheme, parts.netloc, f"{parts.path}/v1/realtime", "", "")
        )
        self.model = model
        self.timezone = timezone or _local_timezone()
        self.memory_dir = Path(
            memory_dir or os.environ.get("EESI_MEMORY_DIR") or DEFAULT_MEMORY_DIR
        )
        self.user_agent = f"eesi-python/{__version__}"
        self.memory = _MemoryFactory(self)
        self.audio = _AudioFactory()
        self._http_timeout = http_timeout
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._voices: tuple[float, list[Voice]] | None = None
        self._voice_lock: asyncio.Lock | None = None

    def __repr__(self) -> str:
        return f"NurClient(base_url={self.base_url!r}, model={self.model!r})"

    async def __aenter__(self) -> NurClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client and the memory databases."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        await self.memory.close()

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    # ── characters ───────────────────────────────────────────────────────

    def character(self, name: str, **options: Any) -> Character:
        """Any character: give it a persona and options, or a ``preset=``."""
        return Character(self, name=name, **options)

    def npc(self, name: str, **options: Any) -> Character:
        """A character who lives in your game's world (preset ``npc``)."""
        return Character(
            self, name=name, preset=options.pop("preset", "npc"), **options
        )

    def multiplayer_npc(self, name: str, **options: Any) -> Character:
        """An NPC several players talk to at once, keeping its story straight between them."""
        return Character(
            self, name=name, preset=options.pop("preset", "multiplayer_npc"), **options
        )

    def companion(self, name: str, **options: Any) -> Character:
        """A personal companion with long memory (preset ``companion``, memory on by default)."""
        options.setdefault("memory", True)
        return Character(
            self, name=name, preset=options.pop("preset", "companion"), **options
        )

    def storyteller(self, name: str, **options: Any) -> Character:
        return Character(
            self, name=name, preset=options.pop("preset", "storyteller"), **options
        )

    def support_agent(self, name: str, **options: Any) -> Character:
        return Character(
            self, name=name, preset=options.pop("preset", "support_agent"), **options
        )

    def assistant(self, name: str, **options: Any) -> Character:
        return Character(
            self, name=name, preset=options.pop("preset", "assistant"), **options
        )

    def load_character(
        self,
        source: str | Path | dict[str, Any],
        *,
        tools: Sequence[Any] = (),
        **overrides: Any,
    ) -> Character:
        """A character saved with :meth:`Character.save`. Pass its ``tools`` again: functions are not saved."""
        import json

        data = (
            dict(source)
            if isinstance(source, dict)
            else json.loads(Path(source).read_text(encoding="utf-8"))
        )
        names = set(data.pop("tools", []) or [])
        data.pop("format", None)
        data.pop("format_version", None)
        voice = data.get("voice")
        if isinstance(voice, dict) and "design" in voice:
            data["voice"] = VoiceDesign(**voice["design"])
        bound = list(tools)
        missing = names - {
            getattr(t, "name", getattr(t, "__name__", "")) for t in bound
        }
        if missing:
            log.warning(
                "loaded character expects tools you did not pass: %s",
                ", ".join(sorted(missing)),
            )
        data.update(overrides)
        return Character(self, tools=bound, **data)

    # ── REST ─────────────────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=f"{self.base_url}/v1",
                headers={
                    **self._auth_headers(),
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
                timeout=self._http_timeout,
                transport=self._transport,
            )
        return self._http

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client().request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise EESIError(f"{method} {path} timed out") from error
        except httpx.TransportError as error:
            raise EESIError(f"Could not reach {self.base_url}: {error}") from error
        if response.status_code >= 400:
            try:
                body: object = response.json()
            except ValueError:
                body = response.text
            raise ApiError.from_body(
                response.status_code,
                body,
                fallback=response.reason_phrase or "Request failed",
            )
        return response

    async def voices(self, *, refresh: bool = False) -> list[Voice]:
        """Built-in voices and your organization's own."""
        if self._voice_lock is None:
            self._voice_lock = asyncio.Lock()
        async with self._voice_lock:
            if (
                not refresh
                and self._voices is not None
                and time.monotonic() - self._voices[0] < 300
            ):
                return list(self._voices[1])
            data = (await self._request("GET", "/voices")).json()
            voices = [
                Voice(
                    id=item.get("voice_id") or item.get("id"),
                    name=item.get("name") or "",
                    is_builtin=bool(item.get("is_builtin")),
                    status=item.get("status") or "ready",
                    language=item.get("language"),
                    gender=item.get("gender"),
                    category=item.get("category"),
                    description=item.get("description"),
                )
                for item in data.get("data", [])
                if item.get("voice_id") or item.get("id")
            ]
            self._voices = (time.monotonic(), voices)
            return list(voices)

    async def resolve_voice(self, voice: str | VoiceDesign | None) -> str | None:
        """A voice id (``ev_…``) for a name, an id or a :class:`VoiceDesign`."""
        if voice is None:
            return None
        if isinstance(voice, VoiceDesign):
            return (
                await self.design_voice(
                    voice.name, voice.description, language=voice.language
                )
            ).id
        if voice.startswith("ev_"):
            return voice
        wanted = voice.strip().lower()
        voices = await self.voices()
        for item in voices:
            if item.name.lower() == wanted or item.id.lower() == wanted:
                return item.id
        names = ", ".join(sorted({v.name for v in voices if v.status == "ready"}))
        raise ConfigurationError(
            f"There is no voice named {voice!r}. Available: {names}. Use a voice id (ev_…), or describe one "
            'with voice=nur.voice_design("Kael", "male, elderly, low pitch, british accent").'
        )

    def voice_design(
        self, name: str, description: str, *, language: str = "en"
    ) -> VoiceDesign:
        """A described voice, created on first use and reused by name after that."""
        return VoiceDesign(name=name, description=description, language=language)

    async def design_voice(
        self, name: str, description: str, *, language: str = "en", wait: float = 60.0
    ) -> Voice:
        """Create a voice from a description (or return the voice already named *name*).

        The voice is synthesized once from the description, then saved to your
        library like a clone, so it sounds the same in every session. It is a
        synthetic voice: no person's recording is used.
        """
        for item in await self.voices():
            if item.name.lower() == name.lower() and not item.is_builtin:
                return item
        sentence = _DESIGN_SENTENCES.get(language, _DESIGN_SENTENCES["en"])
        speech = await self._request(
            "POST",
            "/audio/speech",
            json={
                "model": "nur-tts-v1",
                "input": sentence,
                "voice": "auto",
                "instructions": description,
                "language": language,
                "response_format": "wav",
            },
        )
        created = await self._request(
            "POST",
            "/voices",
            files={"file": (f"{name}.wav", speech.content, "audio/wav")},
            data={
                "name": name,
                # A designed voice is synthetic: no person's voice is cloned.
                "consent_attested": "true",
                "ref_text": sentence,
                "language": language,
                "category": "characters",
                "description": f"Designed: {description}"[:500],
            },
        )
        voice_id = created.json().get("voice_id")
        deadline = time.monotonic() + wait
        status = created.json().get("status") or "processing"
        while status != "ready" and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            status = (await self._request("GET", f"/voices/{voice_id}")).json().get(
                "status"
            ) or status
            if status == "failed":
                raise EESIError(f"Designing the voice {name!r} failed.")
        self._voices = None
        return Voice(
            id=voice_id,
            name=name,
            is_builtin=False,
            status=status,
            language=language,
            description=description,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "nur-llm-v1",
        temperature: float = 0.2,
        max_tokens: int = 900,
        json_output: bool = True,
    ) -> str:
        """One completion from EESI's text model (``/v1/chat/completions``). Memory extraction uses it."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_output:
            body["response_format"] = {"type": "json_object"}
        try:
            data = (await self._request("POST", "/chat/completions", json=body)).json()
        except ApiError as error:
            if json_output and error.status_code == 400:
                body.pop("response_format", None)
                data = (
                    await self._request("POST", "/chat/completions", json=body)
                ).json()
            else:
                raise
        choices = data.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else None
        content = (message or {}).get("content")
        return content if isinstance(content, str) else ""

    async def call(
        self,
        to: str,
        *,
        instructions: str,
        greeting: str | None = None,
        voice: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> PhoneCall:
        """Have Nur call a phone number (E.164) and hold the conversation *instructions* describe.

        Returns at once, while the phone rings; :meth:`wait_for_call` waits
        for it to end and returns its transcript.
        """
        body: dict[str, Any] = {"to": to, "instructions": instructions}
        if greeting:
            body["greeting"] = greeting
        if voice:
            body["voice"] = await self.resolve_voice(voice)
        if metadata:
            body["metadata"] = {str(k): str(v) for k, v in metadata.items()}
        response = await self._request("POST", "/phone/calls", json=body)
        return PhoneCall.from_json(response.json())

    async def phone_call(self, call_id: str) -> PhoneCall:
        """A call Nur placed, as it stands now."""
        response = await self._request("GET", f"/phone/calls/{call_id}")
        return PhoneCall.from_json(response.json())

    async def wait_for_call(
        self, call_id: str, *, timeout: float = 1800.0, poll: float = 5.0
    ) -> PhoneCall:
        """Wait until the call has ended (or was not answered), then return it."""
        deadline = time.monotonic() + timeout
        while True:
            current = await self.phone_call(call_id)
            if current.finished or time.monotonic() >= deadline:
                return current
            await asyncio.sleep(poll)

    async def phone_numbers(self) -> list[PhoneNumber]:
        """The numbers this organization confirmed for calls on the shared EESI line."""
        data = (await self._request("GET", "/phone/numbers")).json()
        return [PhoneNumber.from_json(item) for item in data.get("data") or []]

    async def verify_phone_number(self, number: str) -> PhoneNumber:
        """Have Nur phone *number* (E.164) a six-digit code; :meth:`confirm_phone_number` enters it.

        A number already confirmed comes back confirmed, without a call.
        """
        response = await self._request(
            "POST", "/phone/numbers", json={"number": number}
        )
        return PhoneNumber.from_json(response.json())

    async def confirm_phone_number(
        self, number: str, code: str, *, name: str | None = None
    ) -> PhoneNumber:
        """Enter the code Nur phoned in: from then on, Nur may call *number*.

        *name* is what Nur calls the person when they call its lines.
        """
        body = {"number": number, "code": code}
        if name:
            body["name"] = name
        response = await self._request("POST", "/phone/numbers/confirm", json=body)
        return PhoneNumber.from_json(response.json())

    async def publish_listener(
        self,
        *,
        phone: str,
        instructions: str,
        greeting: str | None = None,
        name: str | None = None,
        repo: str | None = None,
        texting: str | None = None,
    ) -> dict[str, Any]:
        """Brief the code line: what a call from *phone* about your code should know.

        Publish again before ``expires_in`` seconds pass; ``lines`` are the
        shared numbers that answer, and as which line (``code`` or ``live``).
        """
        body: dict[str, Any] = {"phone": phone, "instructions": instructions}
        for key, value in (
            ("greeting", greeting),
            ("name", name),
            ("repo", repo),
            ("texting", texting),
        ):
            if value:
                body[key] = value
        return dict((await self._request("PUT", "/phone/listener", json=body)).json())

    async def listener_events(self, *, wait: float = 20.0) -> list[dict[str, Any]]:
        """What reached the code line since you last asked (waits up to *wait* s).

        ``{"type": "call.ended", "call_id": ...}`` for a call (read it with
        :meth:`phone_call`), ``{"type": "text.received", "text": ..., "reply": ...}``
        for a text and Nur's reply.
        """
        response = await self._request(
            "GET",
            "/phone/listener/events",
            params={"wait": wait},
            timeout=wait + 15,
        )
        return [dict(event) for event in response.json().get("data") or []]

    async def send_text(self, to: str, text: str) -> dict[str, Any]:
        """Text *to* (a confirmed number on the shared line), from the code line."""
        response = await self._request(
            "POST", "/phone/messages", json={"to": to, "text": text}
        )
        return dict(response.json())

    async def forget_phone_number(self, number: str) -> None:
        """Stop calls to *number* on the shared line, until it is confirmed again."""
        await self._request("DELETE", f"/phone/numbers/{number}")

    async def ticket(self) -> str:
        """A one-shot WebSocket ticket (``rt_…``): the client configures the character itself."""
        return str((await self._request("POST", "/realtime/ticket")).json()["ticket"])

    async def client_secret(
        self,
        character: Character,
        *,
        player_id: str | None = None,
        player_name: str | None = None,
        context: dict[str, Any] | None = None,
        expires_in: int = 60,
        lock: Sequence[str] | str | None = None,
        control: bool | str = False,
        language: str | None = None,
        players: Any = None,
        max_duration: int | None = None,
        silence_hangup: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        metadata: dict[str, str] | None = None,
        train: bool = True,
        record: bool = True,
    ) -> ClientSecret:
        """A credential that opens a session already configured as *character* for one player.

        Your server keeps the API key; the client gets ``secret.url`` and
        nothing else. The session it opens has the character's instructions,
        voice and tools (which the client cannot replace), the player's
        memories and the context you pass, composed at mint time. The client
        streams audio, and answers tool calls by forwarding them to your
        server. Requires a gateway with ``/v1/realtime/client_secrets``.

        ``lock`` names what the client may not change: ``"instructions"``,
        ``"tools"``, ``"voice"``, ``"turn_detection"``, ``"context"`` (the
        state you pass here is final) and ``"conversation"`` (replies the
        client asks for are plain, and it may add only the player's own
        messages and tool results). ``"all"`` locks everything. By default
        the instructions and tools are locked.

        ``control=True`` also opens a server control channel: steer the
        session from your server with ``character.control(secret)`` (live
        state, tools, cues, memory) while the client only plays and records
        audio. Tool calls are then your server's to answer; pass
        ``control="client"`` to leave them to the client.

        Beyond the client, too: ``max_duration`` (seconds) ends the call and
        ``silence_hangup`` (seconds) hangs up when nobody speaks, types or is
        answered for that long, both with ``eesi.hangup``. ``webhook_url`` and
        ``webhook_secret`` post each reply and the session's end to your
        server, signed (check it with :func:`eesi.verify_webhook`), tagged with
        ``metadata``. ``train=False`` keeps the session out of any training,
        evaluation or improvement of a model; ``record=False`` keeps no
        recording or transcript after it.
        """
        from .session import Session

        probe = Session(
            character,
            player_id=player_id,
            player_name=player_name,
            context=context,
            language=language,
            players=players,
        )
        try:
            config = await character._session_config(probe)
            if context:
                await probe.context.update(context)
            await probe._load_memory()
            probe._refresh_party()
            blocks = probe._publisher.desired()
            probe._publisher.validate(blocks)
        finally:
            await probe._release()
        body: dict[str, Any] = {
            "session": {"type": "realtime", "model": character.model, **config},
            "expires_after": {"anchor": "created_at", "seconds": int(expires_in)},
            "eesi_context": blocks,
        }
        if lock is not None:
            body["eesi_lock"] = lock if isinstance(lock, str) else list(lock)
        if control:
            tools = "server" if control is True else str(control)
            if tools not in ("server", "client"):
                raise ConfigurationError('control is True, "server" or "client".')
            body["eesi_control"] = {"tools": tools}
        limits = {
            key: int(value)
            for key, value in (
                ("max_duration_seconds", max_duration),
                ("silence_hangup_seconds", silence_hangup),
            )
            if value is not None
        }
        if limits:
            body["eesi_limits"] = limits
        if webhook_url or webhook_secret:
            if not (webhook_url and webhook_secret):
                raise ConfigurationError(
                    "A webhook needs both webhook_url and webhook_secret."
                )
            body["eesi_webhook"] = {"url": webhook_url, "secret": webhook_secret}
        if metadata:
            body["eesi_metadata"] = {str(k): str(v) for k, v in metadata.items()}
        if not train or not record:
            body["eesi_data"] = {"train": bool(train), "record": bool(record)}
        try:
            data = (
                await self._request("POST", "/realtime/client_secrets", json=body)
            ).json()
        except ApiError as error:
            if error.status_code in (404, 405):
                raise ConfigurationError(
                    "This EESI deployment cannot bind a character to a client secret yet. Use nur.ticket() and send "
                    "await character.session_config() from the client as its first session.update."
                ) from error
            raise
        value = data.get("value") or data.get("client_secret", {}).get("value")
        query = urlencode(
            {"model": character.model, "source": "sdk-client", "token": value}
        )
        url = f"{self.realtime_url}?{query}"
        expires_at = float(data.get("expires_at") or time.time() + expires_in)
        locks = tuple(data.get("eesi_lock") or ("instructions", "tools"))
        channel = data.get("eesi_control") or {}
        controlled = bool(channel.get("id"))
        # A controlled session's greeting, replies and remarks are the server's.
        known = any(record.player_id is not None for record in probe._memory_records)
        greeting = (
            character._greeting_instructions(
                probe._greeting,
                known=known,
                stranger=probe.memory is not None and not known,
                name=player_name,
            )
            if probe._greeting and not controlled
            else None
        )
        proactive = character.proactive
        join = {
            "url": url,
            # The same one-use secret, for a WebRTC client: POST its SDP offer
            # to callsUrl with "Authorization: Bearer <token>".
            "token": value,
            "callsUrl": f"{self.base_url}/v1/realtime/calls",
            "expiresAt": expires_at,
            "character": character.name,
            "model": character.model,
            "turnTaking": (
                "natural"
                if character.turn_taking == "natural"
                else "director"
                if controlled
                else character.turn_taking
            ),
            "tools": "server"
            if controlled and channel.get("tools") == "server"
            else "client",
            "controlled": controlled,
            "greeting": greeting,
            "proactive": (
                {
                    "idleAfterMs": None
                    if proactive.idle_after is None
                    else proactive.idle_after * 1000,
                    "cooldownMs": proactive.cooldown * 1000,
                    "maxRemarks": proactive.max_remarks,
                    "guidance": proactive.guidance,
                }
                if proactive is not None
                and not controlled
                and "conversation" not in locks
                else None
            ),
            "outputSampleRate": character.output_sample_rate,
            "playerId": player_id,
            "playerName": player_name,
        }
        return ClientSecret(
            value=value,
            expires_at=expires_at,
            url=url,
            session=data.get("session") or body["session"],
            lock=locks,
            control_id=channel.get("id"),
            control_tools=channel.get("tools"),
            player_id=player_id,
            player_name=player_name,
            join=join,
        )


def _local_timezone() -> str | None:
    """This machine's IANA timezone, when it can be found."""
    tz = os.environ.get("TZ")
    if tz and "/" in tz:
        return tz
    try:
        link = os.readlink("/etc/localtime")
    except OSError:
        return None
    marker = "zoneinfo/"
    return link.split(marker, 1)[1] if marker in link else None


class WebhookVerificationError(EESIError):
    """A webhook whose signature or timestamp does not check out."""


def verify_webhook(
    body: bytes | str,
    headers: Any,
    secret: str,
    *,
    tolerance: float = 300.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Check one webhook post from a client secret's session and return its event.

    *body* is the raw request body, exactly as received; *headers* any mapping
    of the request's headers. Raises :class:`WebhookVerificationError` when the
    ``EESI-Signature`` does not match or the post is older than *tolerance*
    seconds (a replay).
    """
    import hashlib
    import hmac
    import json

    raw = body.encode() if isinstance(body, str) else bytes(body)
    lowered = {str(k).lower(): str(v) for k, v in dict(headers).items()}
    webhook_id = lowered.get("eesi-webhook-id", "")
    stamp = lowered.get("eesi-webhook-timestamp", "")
    signature = lowered.get("eesi-signature", "")
    try:
        timestamp = int(stamp)
    except ValueError:
        raise WebhookVerificationError("The webhook has no valid timestamp.") from None
    current = time.time() if now is None else now
    if abs(current - timestamp) > tolerance:
        raise WebhookVerificationError("The webhook is too old (or from the future).")
    expected = hmac.new(
        secret.encode(), f"{webhook_id}.{timestamp}.".encode() + raw, hashlib.sha256
    ).hexdigest()
    given = [
        part[3:] for part in signature.split(",") if part.strip().startswith("v1=")
    ]
    if not any(hmac.compare_digest(expected, candidate.strip()) for candidate in given):
        raise WebhookVerificationError("The webhook signature does not match.")
    event = json.loads(raw)
    if not isinstance(event, dict) or event.get("id") != webhook_id:
        raise WebhookVerificationError("The webhook body does not match its id.")
    return event
