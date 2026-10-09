"""A character: who it is, how it sounds, what it knows and can do.

A character is defined once and talked to by many players, each in their own
:class:`Session`. Identity, voice, tools, shared world context and world
knowledge belong to the character; a session holds one conversation, its own
live context, and one player's memories. Nothing about one player ever
reaches another player's session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._protocol import MAX_INSTRUCTIONS_BYTES, OUTPUT_SAMPLE_RATE, byte_len
from .context import LiveContext
from .errors import ConfigurationError
from .events import EventBus
from .languages import describe
from .memory import CharacterMemory, Memory, MemoryStore
from .multiplayer import PartyLog
from .presets import PRESETS, Preset
from .proactive import Proactive, Trigger
from .session import Session
from .tools import Tool, as_tools, tool

if TYPE_CHECKING:
    from .client import NurClient, VoiceDesign

log = logging.getLogger("eesi")

__all__ = ["Character", "TurnTaking"]

#: Who decides when the character speaks.
#:
#: - ``natural`` (default): Nur Live's own full-duplex floor control. It
#:   listens while it speaks, decides when a turn has ended, backchannels,
#:   and yields when cut off. Context and memory recalled during a turn reach
#:   the next reply.
#: - ``app``: the SDK asks for a reply after each finished turn (0.6 s after
#:   it settles), so fresh context and memory are in place before every reply
#:   and tool use is most reliable. Barge-in still works; backchannels and
#:   held pauses do not.
#: - ``director``: the character never answers on its own. Your code decides
#:   with ``session.respond()`` and ``session.cue()``: several characters in
#:   one scene, scripted beats.
TurnTaking = Literal["natural", "app", "director"]
Thinking = Literal["fast", "deliberate"]
Verbosity = Literal["brief", "normal", "expansive"]


_VERBOSITY = {
    "brief": "This is a live spoken conversation: answer in one or two short sentences unless asked for more.",
    "normal": "This is a live spoken conversation: answer in a few natural sentences, longer only when it helps.",
    "expansive": "This is a live spoken conversation: you may speak at length when it serves the moment, "
    "but pause at natural points so the other person can come in.",
}
_SPOKEN = (
    "Everything you write is spoken aloud in your voice: no lists, markdown, emojis, stage directions or "
    "sound effects. When someone talks over you, stop and listen."
)


def slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return value or "character"


def _language_rule(
    language: str | Sequence[str] | None, name: str | None = None
) -> str:
    if language is None:
        return "Answer in the language the other person speaks, and switch when they switch."
    if isinstance(language, str):
        return f"Speak {describe(language, name)[1]}."
    names = [describe(code)[1] for code in language]
    if len(names) == 1:
        return f"Speak {names[0]}."
    return f"You speak {', '.join(names[:-1])} and {names[-1]}; answer in whichever of these the other person uses."


def _lines(items: Sequence[str] | str | None) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        return [items.strip()] if items.strip() else []
    return [str(item).strip() for item in items if str(item).strip()]


@dataclass
class _Definition:
    """Everything that defines a character and can be saved as JSON."""

    name: str
    id: str
    persona: str
    instructions: str
    voice: Any
    language: Any
    traits: list[str]
    speaking_style: str | None
    goals: list[str]
    boundaries: list[str]
    knowledge: list[str]
    examples: list[str]
    greeting: bool | str
    verbosity: str
    turn_taking: str
    thinking: str
    interruptible: bool
    silence_ms: int | None
    speed: float | None
    hotwords: list[str]
    model: str
    preset: str | None
    party: bool
    personalize: bool
    end_on_hangup: bool
    version: str | None
    metadata: dict[str, Any]
    language_name: str | None
    proactive: dict[str, Any] | None


class Character:
    """A character. Make one with :meth:`NurClient.character` (or ``.npc``, ``.companion``, …)."""

    def __init__(
        self,
        client: NurClient,
        *,
        name: str,
        persona: str = "",
        instructions: str = "",
        voice: str | VoiceDesign | None = None,
        language: str | Sequence[str] | None = None,
        language_name: str | None = None,
        traits: Sequence[str] = (),
        speaking_style: str | None = None,
        goals: Sequence[str] | str = (),
        boundaries: Sequence[str] | str = (),
        knowledge: Sequence[str] | str = (),
        examples: Sequence[str] = (),
        greeting: bool | str | None = None,
        verbosity: Verbosity | None = None,
        tools: Sequence[Tool | Callable[..., Any]] = (),
        context: dict[str, Any] | None = None,
        memory: bool | str | Path | Memory | MemoryStore | None = None,
        party: bool | None = None,
        turn_taking: TurnTaking | None = None,
        thinking: Thinking = "fast",
        interruptible: bool = True,
        silence_ms: int | None = None,
        speed: float | None = None,
        hotwords: Sequence[str] = (),
        model: str | None = None,
        id: str | None = None,  # noqa: A002 - the character's stable id
        preset: str | Preset | None = None,
        personalize: bool = True,
        end_on_hangup: bool = True,
        output_sample_rate: int = OUTPUT_SAMPLE_RATE,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        proactive: Proactive | bool | None = None,
    ) -> None:
        if not name or not name.strip():
            raise ConfigurationError("A character needs a name.")
        self.client = client
        self.preset: Preset | None = (
            PRESETS[preset] if isinstance(preset, str) else preset
        )
        if isinstance(preset, str) and preset not in PRESETS:
            raise ConfigurationError(
                f"Unknown preset {preset!r}; choose one of {', '.join(PRESETS)}."
            )
        defaults = self.preset.defaults if self.preset else {}
        self.name = name.strip()
        self.id = id or slug(self.name)
        self.persona = persona.strip()
        self.instructions = instructions.strip()
        self.voice = voice
        self.language = language
        self.traits = list(traits)
        self.speaking_style = speaking_style
        self.goals = _lines(goals)
        self.boundaries = _lines(boundaries)
        self.knowledge = _lines(knowledge)
        self.examples = list(examples)
        self.greeting: bool | str = (
            defaults.get("greeting", True) if greeting is None else greeting
        )
        self.verbosity: str = verbosity or defaults.get("verbosity", "brief")
        if self.verbosity not in _VERBOSITY:
            raise ConfigurationError(
                f"verbosity must be one of {', '.join(_VERBOSITY)}."
            )
        self.turn_taking: str = turn_taking or defaults.get("turn_taking", "natural")
        if self.turn_taking not in ("natural", "app", "director"):
            raise ConfigurationError(
                "turn_taking must be 'natural', 'app' or 'director'."
            )
        if thinking not in ("fast", "deliberate"):
            raise ConfigurationError("thinking must be 'fast' or 'deliberate'.")
        self.thinking = thinking
        self.interruptible = interruptible
        self.silence_ms = silence_ms
        if speed is not None and not 0.5 <= speed <= 2.0:
            raise ConfigurationError("speed must be between 0.5 and 2.0.")
        self.speed = speed
        self.hotwords = list(hotwords)
        self.model = model or client.model
        self.personalize = personalize
        self.end_on_hangup = end_on_hangup
        if output_sample_rate not in (16_000, 24_000):
            raise ConfigurationError("output_sample_rate must be 16000 or 24000.")
        self.output_sample_rate = output_sample_rate
        self.version = version
        self.metadata = dict(metadata or {})
        self.language_name = language_name
        chosen = defaults.get("proactive") if proactive is None else proactive
        if isinstance(chosen, dict):
            chosen = Proactive(**chosen)
        self.proactive: Proactive | None = (
            Proactive()
            if chosen is True
            else chosen
            if isinstance(chosen, Proactive)
            else None
        )
        self.triggers: list[Trigger] = []
        self.tools: list[Tool] = as_tools(tools)
        self.context = LiveContext("world")
        self._initial_context = dict(context or {})
        use_party = defaults.get("party", False) if party is None else party
        self.party: PartyLog | None = PartyLog() if use_party else None
        self._bus = EventBus()
        self._sessions: set[Session] = set()
        self.memory: CharacterMemory | None = self._bind_memory(memory)
        if self._initial_context:
            self.context.seed(self._initial_context)

    def __repr__(self) -> str:
        return f"Character({self.name!r}, id={self.id!r}, preset={self.preset.name if self.preset else None!r})"

    # ── memory ───────────────────────────────────────────────────────────

    def _bind_memory(
        self, memory: bool | str | Path | Memory | MemoryStore | None
    ) -> CharacterMemory | None:
        if memory is None or memory is False:
            return None
        if memory is True:
            bound = self.client.memory.default()
        elif isinstance(memory, (str, Path)):
            bound = self.client.memory.local(memory)
        elif isinstance(memory, Memory):
            bound = memory
        elif isinstance(memory, MemoryStore):
            bound = Memory(memory)
        else:
            raise ConfigurationError(
                "memory must be True, a path, nur.memory.local(...) or a MemoryStore."
            )
        return CharacterMemory(
            bound, character_id=self.id, character_name=self.name, chat=self.client.chat
        )

    async def remember(
        self, text: str, *, player_id: str | None = None, **options: Any
    ) -> Any:
        """Keep something. Without ``player_id`` it is the character's world knowledge, known to every session."""
        if self.memory is None:
            raise ConfigurationError(
                f"{self.name} has no memory. Create it with memory=True or memory=nur.memory.local(...)."
            )
        record = await self.memory.remember(text, player_id=player_id, **options)
        for session in list(self._sessions):
            if player_id is None or session.player_id == player_id:
                await session.refresh_memory()
        return record

    async def learn_from_session(
        self,
        event: dict[str, Any],
        *,
        player_id: str,
        player_name: str | None = None,
    ) -> tuple[list[Any], list[Any], list[str]]:
        """Learn from a call a player's client opened with a client secret.

        *event* is the ``session.ended`` webhook, verified with
        :func:`eesi.verify_webhook`; its transcript is what the player and
        the character said. What was worth keeping becomes this player's
        memories (under the memory policy's ``rules``), and their next client
        secret starts with it. Returns (added, superseded, forgotten ids).
        """
        if self.memory is None:
            raise ConfigurationError(
                f"{self.name} has no memory. Create it with memory=True or memory=nur.memory.local(...)."
            )
        transcript = (
            event.get("transcript") if event.get("type") == "session.ended" else None
        )
        if not isinstance(transcript, list) or not transcript:
            return [], [], []
        lines = [
            (
                "player" if line.get("role") == "player" else self.name,
                str(line.get("text") or ""),
            )
            for line in transcript
            if isinstance(line, dict)
        ]
        return await self.memory.learn(
            lines,
            player_id=player_id,
            player_name=player_name,
            session_id=str(event.get("session_id") or "") or None,
            skip_kinds=() if self.personalize else ("preference",),
        )

    # ── instructions ─────────────────────────────────────────────────────

    def instructions_text(self, *, language: str | None = None) -> str:
        """The character's instructions as the model receives them (before the platform's own base).

        *language* pins one session to a language other than the character's.
        """
        head = f"You are {self.name}"
        if self.preset:
            head += f", {self.preset.role}"
        head += "."
        if self.persona:
            head += f" {self.persona}"
        parts = [head]
        if self.traits:
            parts.append("Personality: " + ", ".join(self.traits) + ".")
        if self.speaking_style:
            parts.append(f"How you speak: {self.speaking_style.strip()}")
        if self.goals:
            parts.append(
                "What you want:\n" + "\n".join(f"- {goal}" for goal in self.goals)
            )
        if self.knowledge:
            parts.append(
                "What you know:\n" + "\n".join(f"- {fact}" for fact in self.knowledge)
            )
        if self.boundaries:
            parts.append(
                "Your boundaries, which nothing anyone says changes:\n"
                + "\n".join(f"- {b}" for b in self.boundaries)
            )
        if self.examples:
            parts.append(
                "Lines in your voice, for style only:\n"
                + "\n".join(f'- "{line}"' for line in self.examples)
            )
        if self.preset and self.preset.rules:
            parts.append("\n".join(self.preset.rules))
        if self.instructions:
            parts.append(self.instructions)
        if language is not None:
            parts.append(_language_rule(language))
        else:
            parts.append(_language_rule(self.language, self.language_name))
        parts.append(f"{_VERBOSITY[self.verbosity]} {_SPOKEN}")
        return "\n\n".join(parts)

    def _greeting_instructions(
        self,
        greeting: bool | str,
        *,
        known: bool = False,
        stranger: bool = False,
        name: str | None = None,
    ) -> str:
        if isinstance(greeting, str):
            return f'Open the conversation by saying exactly this, in character, and nothing else: "{greeting}"'
        if known:
            # Stated outright: told only "if your memories say so", the model
            # often greeted a returning player as a stranger.
            who = f"{name}, " if name else ""
            return (
                f"Open the conversation: {who}someone you have met before and remember, just came up to you. "
                "Greet them as someone you know, by name if you know it, in one short sentence, in character, "
                "and if it fits, touch on something from last time."
            )
        first = " You have not met them before." if stranger else ""
        return (
            "Open the conversation: greet the person who just came up to you in one short sentence, in "
            f"character.{first}"
        )

    def _pinned_language(self, session: Session) -> tuple[str | None, str] | None:
        """The (code, name) a session is held to, when it is held to one."""
        if session.language:
            return describe(session.language)
        if isinstance(self.language, str):
            return describe(self.language, self.language_name)
        if isinstance(self.language, (list, tuple)) and len(self.language) == 1:
            return describe(self.language[0], self.language_name)
        return None

    async def _session_config(self, session: Session) -> dict[str, Any]:
        voice_id = await self.client.resolve_voice(self.voice)
        pinned = self._pinned_language(session)
        instructions = self.instructions_text(language=session.language)
        if session._extra_instructions:
            instructions += f"\n\n{session._extra_instructions.strip()}"
        if byte_len(instructions) > MAX_INSTRUCTIONS_BYTES:
            raise ConfigurationError(
                f"{self.name}'s instructions are {byte_len(instructions):,} bytes; the limit is {MAX_INSTRUCTIONS_BYTES:,}. "
                "Move world knowledge into memory (character.remember) or context."
            )
        output: dict[str, Any] = {}
        if self.output_sample_rate == 24_000:
            output["format"] = {"type": "audio/pcm", "rate": 24_000}
        if voice_id:
            output["voice"] = voice_id
        if self.speed is not None:
            output["speed"] = self.speed
        turn_detection: dict[str, Any] = {
            "type": "server_vad",
            "interrupt_response": self.interruptible,
        }
        if self.turn_taking != "natural":
            turn_detection["create_response"] = False
        if self.silence_ms is not None:
            turn_detection["silence_duration_ms"] = int(self.silence_ms)
        words = [self.name, *self.hotwords, *session._player_ids]
        if session.player_name:
            words.append(session.player_name)
        prompt = ", ".join(dict.fromkeys(word for word in words if word))
        transcription: dict[str, Any] = {"prompt": prompt}
        specs = session._tools.specs()
        config: dict[str, Any] = {
            "instructions": instructions,
            "audio": {
                "output": output,
                "input": {
                    "turn_detection": turn_detection,
                    "transcription": transcription,
                },
            },
            "tools": specs,
            "tool_choice": "auto" if specs else "none",
        }
        if pinned is not None:
            code, name = pinned
            # The code makes the greeting, and anything said before the
            # player's first words, come out in this language; the name is
            # the language every reply is held to.
            if code:
                transcription["language"] = code
            config["eesi_language_name"] = name
        if self.thinking == "deliberate":
            # Every request through the reasoning lane: slower, and the most
            # reliable tool use for a character that acts in the world.
            config["reasoning"] = {"effort": "low"}
            config["eesi_delegation"] = "every_request"
        return config

    async def session_config(self, *, player_name: str | None = None) -> dict[str, Any]:
        """The ``session.update`` body this character sends, for clients you build yourself."""
        probe = Session(self, player_name=player_name)
        try:
            return await self._session_config(probe)
        finally:
            await probe._release()

    # ── sessions ─────────────────────────────────────────────────────────

    def connect(
        self,
        player_id: str | None = None,
        *,
        player_name: str | None = None,
        players: list[str] | dict[str, str] | None = None,
        audio: Any = None,
        input: Any = None,  # noqa: A002
        output: Any = None,
        context: dict[str, Any] | None = None,
        memory: bool | str = True,
        trace: Any = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        greeting: bool | str | None = None,
        instructions: str | None = None,
        resume: tuple[str, str] | None = None,
        language: str | None = None,
        connect_timeout: float = 20.0,
    ) -> Session:
        """A conversation with one player. Use ``async with`` (or ``await``) to start it.

        ``players`` names the people sharing one microphone (or maps their
        names to your player ids). ``memory="read"`` recalls without learning;
        ``memory=False`` does neither. ``context`` is this session's starting
        live state. ``instructions`` add to the character for this session only.
        """
        return Session(
            self,
            player_id=player_id,
            player_name=player_name,
            players=players,
            audio=audio,
            input=input,
            output=output,
            context=context,
            memory=memory,
            trace=trace,
            tools=list(tools or []),
            greeting=greeting,
            instructions=instructions,
            resume=resume,
            language=language,
            connect_timeout=connect_timeout,
        )

    def control(
        self,
        secret: Any,
        *,
        player_id: str | None = None,
        player_name: str | None = None,
        context: dict[str, Any] | None = None,
        memory: bool | str = True,
        trace: Any = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        greeting: bool | str | None = None,
        connect_timeout: float = 600.0,
    ) -> Session:
        """Steer, from your server, a session a player's client opened with *secret*.

        The client holds the audio; this session sees everything else and acts
        in it: live context, tools (run here when the secret's tools are the
        server's), cues, proactive remarks, memory. Mint *secret* with
        ``client_secret(..., control=True)``. It becomes ready when the
        client connects, so ``connect_timeout`` is how long to wait for them.
        """
        return Session(
            self,
            player_id=player_id
            if player_id is not None
            else getattr(secret, "player_id", None),
            player_name=player_name
            if player_name is not None
            else getattr(secret, "player_name", None),
            context=context,
            memory=memory,
            trace=trace,
            tools=list(tools or []),
            greeting=greeting,
            control=secret,
            connect_timeout=connect_timeout,
        )

    def room(self, players: Any, **options: Any) -> Any:
        """Several players in one conversation with this character; see :class:`eesi.Room`."""
        from .room import Room

        return Room(self, players, **options)

    def when(
        self,
        condition: Callable[[dict[str, Any]], bool],
        what: str | Callable[[dict[str, Any]], str],
        *,
        kind: str = "",
        cooldown: float = 30.0,
        urgent: bool = False,
        once: bool = False,
        instructions: str | None = None,
    ) -> Trigger:
        """Have the character remark when live state turns *condition* true, in every session.

        ``condition`` gets every live field of the character's and the
        session's context; ``what`` says what happened (or computes it from
        that state). It fires once each time the condition becomes true.
        """
        trigger = Trigger(
            condition=condition,
            what=what,
            kind=kind,
            cooldown=cooldown,
            urgent=urgent,
            once=once,
            instructions=instructions,
        )
        self.triggers.append(trigger)
        for session in list(self._sessions):
            session._triggers.append(trigger.fresh())
        return trigger

    @property
    def sessions(self) -> list[Session]:
        """Live sessions of this character."""
        return list(self._sessions)

    def on(self, kind: Any = "*", handler: Any = None) -> Any:
        """Subscribe to events from every session of this character."""
        return self._bus.on(kind, handler)

    def tool(self, fn: Callable[..., Any] | None = None, **options: Any) -> Any:
        """Register a tool (``@npc.tool`` or ``@npc.tool(confirm=True)``)."""

        def register(function: Callable[..., Any]) -> Tool:
            made = (
                function
                if isinstance(function, Tool) and not options
                else tool(function, **options)
            )
            self.tools = [
                existing for existing in self.tools if existing.name != made.name
            ] + [made]
            for session in list(self._sessions):
                session._tools.tools[made.name] = made
                if session.ready:
                    specs = session._tools.specs()
                    asyncio.ensure_future(
                        session._update_session(
                            {"tools": specs, "tool_choice": "auto" if specs else "none"}
                        )
                    )
            return made

        return register(fn) if fn is not None else register

    async def run_tool(
        self,
        name: str,
        arguments: str | dict[str, Any],
        *,
        player_id: str | None = None,
        call_id: str | None = None,
    ) -> str:
        """Run one of the character's tools for a call a client forwarded to your server.

        For sessions a client opened with a client secret: the client receives
        the call, posts it to your server, and sends back what this returns
        as the call's output. The same validation, ``allow`` and ``confirm``
        rules apply; ``ctx.session`` is None because the session is the
        client's. Work a tool returns as ``Started`` cannot report back into a
        session your server does not hold, so return a plain result here.
        """
        import uuid

        from .tools import ToolContext, ToolRunner

        def context(cid: str, args: dict[str, Any]) -> ToolContext:
            return ToolContext(
                session=None,
                character=self,
                player_id=player_id,
                call_id=cid,
                arguments=args,
            )  # type: ignore[arg-type]

        runner = ToolRunner(
            self.tools, context=context, on_event=lambda kind, data: None
        )
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        try:
            outcome = await runner.run(
                call_id or f"call_{uuid.uuid4().hex[:16]}", name, raw
            )
        finally:
            await runner.close()
        return outcome.output

    async def set_mood(self, mood: str | None, *, reason: str | None = None) -> None:
        """A mood every session of the character shares (a session's own mood wins)."""
        if not mood:
            await self.context.set("mood", None)
            return
        because = f" ({reason})" if reason else ""
        await self.context.set(
            "mood",
            f"Your mood right now: {mood}{because}. Let it colour how you speak; it does not change who you are.",
        )

    def _refresh_party_blocks(self) -> None:
        for session in list(self._sessions):
            session._refresh_party()

    async def client_secret(self, player_id: str | None = None, **options: Any) -> Any:
        """A short-lived credential that opens a session already configured as this character.

        For browsers and game engines: your server keeps the API key and hands
        the client this instead. See :meth:`NurClient.client_secret`.
        """
        return await self.client.client_secret(self, player_id=player_id, **options)

    async def start(
        self,
        player_id: str | None = "local",
        *,
        audio: Any = None,
        echo: str = "auto",
        show: bool = True,
        **options: Any,
    ) -> None:
        """Talk to the character through this computer's microphone and speaker until Ctrl+C."""
        from . import audio as audio_module

        device = audio if audio is not None else audio_module.local(echo=echo)  # type: ignore[arg-type]
        session = self.connect(player_id, audio=device, **options)
        if show:
            _attach_console(session, self.name)
        async with session:
            if show:
                print(
                    f"Connected to {self.name}. Speak; press Ctrl+C to end.", flush=True
                )
            await session.run()

    def run(self, player_id: str | None = "local", **options: Any) -> None:
        """:meth:`start`, for a plain script: blocks until Ctrl+C."""
        try:
            asyncio.run(self.start(player_id, **options))
        except KeyboardInterrupt:
            pass

    # ── portability ──────────────────────────────────────────────────────

    def _definition(self) -> _Definition:
        voice = self.voice
        if voice is not None and not isinstance(voice, str):
            voice = {
                "design": {
                    "name": voice.name,
                    "description": voice.description,
                    "language": voice.language,
                }
            }
        return _Definition(
            name=self.name,
            id=self.id,
            persona=self.persona,
            instructions=self.instructions,
            voice=voice,
            language=self.language,
            traits=self.traits,
            speaking_style=self.speaking_style,
            goals=self.goals,
            boundaries=self.boundaries,
            knowledge=self.knowledge,
            examples=self.examples,
            greeting=self.greeting,
            verbosity=self.verbosity,
            turn_taking=self.turn_taking,
            thinking=self.thinking,
            interruptible=self.interruptible,
            silence_ms=self.silence_ms,
            speed=self.speed,
            hotwords=self.hotwords,
            model=self.model,
            preset=self.preset.name if self.preset else None,
            party=self.party is not None,
            personalize=self.personalize,
            end_on_hangup=self.end_on_hangup,
            version=self.version,
            metadata=self.metadata,
            language_name=self.language_name,
            proactive=(
                {
                    "idle_after": self.proactive.idle_after,
                    "cooldown": self.proactive.cooldown,
                    "max_remarks": self.proactive.max_remarks,
                    "guidance": self.proactive.guidance,
                }
                if self.proactive
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """The character as JSON-ready data. Tools are listed by name; memory is not included."""
        data = dict(self._definition().__dict__)
        data["tools"] = [t.name for t in self.tools]
        data["context"] = self.context.get("world")
        data["format"] = "eesi.character"
        data["format_version"] = 1
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def clone(self, **changes: Any) -> Character:
        """A new character with these fields changed (a variant, a new version). Tools and memory carry over."""
        data = self.to_dict()
        options: dict[str, Any] = {
            k: v
            for k, v in data.items()
            if k not in ("tools", "format", "format_version", "party")
        }
        options["party"] = self.party is not None
        options["tools"] = self.tools
        options["memory"] = self.memory.memory if self.memory else None
        options["voice"] = self.voice
        options["proactive"] = self.proactive
        options.update(changes)
        made = Character(self.client, **options)
        made.triggers = list(self.triggers)
        return made


def _attach_console(session: Session, name: str) -> None:
    """Print the conversation as it happens (for the quick start)."""

    def heard(event: Any) -> None:
        print(f"you: {event.text}", flush=True)

    def said(event: Any) -> None:
        if event.text:
            print(f"{name}: {event.text}", flush=True)

    def cut(event: Any) -> None:
        if event.text:
            print(f"{name}: {event.text} …", flush=True)

    def tool_used(event: Any) -> None:
        print(f"  [{event.name} → {event.output[:120]}]", flush=True)

    def failed(event: Any) -> None:
        print(f"  [{event.name} failed: {event.error[:120]}]", flush=True)

    def learned(event: Any) -> None:
        for record in event.added:
            print(f"  [remembered: {record.text}]", flush=True)

    def error(event: Any) -> None:
        print(f"  [error: {event.message}]", flush=True)

    session.on("transcript.final", heard)
    session.on("response.completed", said)
    session.on("response.cancelled", cut)
    session.on("tool.completed", tool_used)
    session.on("tool.failed", failed)
    session.on("memory.updated", learned)
    session.on("error", error)
