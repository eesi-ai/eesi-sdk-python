"""One conversation between a character and a player, in real time.

::

    async with npc.connect(player_id="p42", audio=nur.audio.local()) as session:
        await session.context.update({"location": "Northern Gate"})
        await session.run()  # until the conversation ends

The session holds an ordinary Nur Live realtime connection
(``wss://…/v1/realtime``) under your API key, so authentication, metering,
recording consent and organization scope stay at the gateway. Nur Live owns
the conversation itself: voice activity, turn-taking, interruptions,
backchannels, the history and cancellation. The session owns what the
application knows: the character, live context, memory, tools, and the audio
on this side.

Lifecycle: ``connect()`` dials, sends the character, sends context and memory,
and returns once the character is ready (``session.started``). A dropped
connection is resumed with the conversation intact when the gateway allows it
(``reconnecting`` / ``reconnected``); otherwise the session ends with
``connection_lost``. ``close()`` hangs up, waits for memory to be learned, and
ends every event stream. It is safe to call twice.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import random
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from . import _protocol as wire
from ._initiative import Initiative, Moment, moment_instructions
from .audio import AudioInput, AudioOutput, Resampler
from .context import ContextPublisher, LiveContext
from .errors import (
    AuthenticationError,
    ConfigurationError,
    ContextError,
    RealtimeConnectionError,
    SessionClosedError,
)
from .events import (
    AudioInputEnded,
    AudioInputStarted,
    AudioOutputChunk,
    ContextUpdated,
    ErrorEvent,
    Event,
    EventBus,
    Feedback,
    Hangup,
    MemoryRetrieved,
    MemoryUpdated,
    Reconnected,
    Reconnecting,
    ResponseCancelled,
    ResponseCompleted,
    ResponseStarted,
    ResponseTextDelta,
    SessionEnded,
    SessionStarted,
    SpeakerTurn,
    SpeechInterrupted,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
    TranscriptFinal,
    TranscriptPartial,
)
from .languages import describe
from .memory import MemoryRecord, PlayerMemory
from .multiplayer import AttributedTurn, Player, SpeakerRoster, as_players
from .proactive import IDLE_SKIP, Trigger
from .tools import Tool, ToolContext, ToolRunner, as_tools
from .trace import SessionMetrics, Trace

if TYPE_CHECKING:
    from .character import Character

log = logging.getLogger("eesi")

__all__ = ["Preferences", "Session"]

#: How long a finished turn waits before the app-driven reply is asked for:
#: the transcript of "Hi Kael." can land while the player is saying the rest.
TURN_SETTLE_S = 0.6
#: Redials spent on one drop before the session is declared lost.
MAX_RECONNECT_ATTEMPTS = 6
FATAL_CLOSE_CODES = {1002, 1003, 1008}
STATUS_INTERVAL_S = 0.5
KEEPALIVE_INTERVAL_S = 60.0
#: How long the gateway may take to accept the character after session.created.
CONFIGURE_TIMEOUT_S = 10.0
#: Memory search for a finished turn may hold an app-driven reply this long.
RECALL_WAIT_S = 0.35
LEARN_TIMEOUT_S = 45.0
#: How long ask() waits for its line to be acknowledged before going by timing alone.
ITEM_ACK_FALLBACK_S = 2.0
#: A memory must match what was just said at least this well to be recalled mid-session.
RECALL_MIN_SCORE = 0.35
RECALL_MAX = 3
#: How often a session looks for a quiet stretch worth a remark.
IDLE_POLL_S = 1.0
#: The metadata that marks the SDK's own out-of-band decisions.
INTERNAL_TAG = "eesi_sdk"


def _reconnect_delay(attempt: int) -> float:
    window = min(8.0, 1.0 * 2 ** max(0, attempt))
    return window / 2 + random.random() * window / 2


@dataclass
class _Response:
    id: str
    created_at: float
    deltas: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    segment_ids: set[str] = field(default_factory=set)
    had_audio: bool = False
    first_audio_at: float | None = None
    calls: int = 0
    outstanding: set[str] = field(default_factory=set)
    speak_after: bool = False
    done: bool = False
    cancelled: bool = False
    continued: bool = False
    #: The player turn this reply answers, when it answers one.
    turn_ended_at: float | None = None

    def text(self) -> str:
        if self.segments:
            return " ".join(s.strip() for s in self.segments if s.strip())
        return "".join(self.deltas).strip()


class Preferences:
    """How a player likes to be spoken to, applied at once.

    ``await session.preferences.set(verbosity="brief", language="Arabic")``
    keeps them for every later session with this character (in its memory);
    ``persist=False`` adapts only this session.
    """

    PHRASES: dict[str, Callable[[str], str]] = {
        "language": lambda v: f"Speak {v} with them unless they switch.",
        "verbosity": lambda v: {
            "brief": "Keep your answers brief.",
            "short": "Keep your answers brief.",
            "detailed": "They like fuller, more detailed answers.",
            "long": "They like fuller, more detailed answers.",
        }.get(v.lower(), f"Verbosity: {v}."),
        "pace": lambda v: {
            "slow": "They like an unhurried pace.",
            "fast": "They like a brisk pace.",
        }.get(v.lower(), f"Pace: {v}."),
        "formality": lambda v: {
            "formal": "Speak to them formally.",
            "casual": "Speak to them casually.",
        }.get(v.lower(), f"Formality: {v}."),
        "name": lambda v: f"Call them {v}.",
    }
    SPEED = {"slow": 0.9, "normal": 1.0, "fast": 1.1}

    def __init__(self, session: Session) -> None:
        self._session = session
        self.session_only: dict[str, str] = {}

    def phrase(self, key: str, value: str) -> str:
        render = self.PHRASES.get(key)
        return render(str(value)) if render else f"{key.replace('_', ' ')}: {value}."

    async def set(self, *, persist: bool = True, **values: str | None) -> None:
        session = self._session
        memory = session.memory
        for key, value in values.items():
            if value is None:
                self.session_only.pop(key, None)
                if persist and memory is not None:
                    for record in await memory.list(kinds=["preference"]):
                        if record.key == key:
                            await memory.forget(record.id)
                continue
            text = self.phrase(key, str(value))
            if persist and memory is not None and session._memory_writable:
                await memory.remember(
                    text, kind="preference", key=key, source="player", importance=0.8
                )
                self.session_only.pop(key, None)
            else:
                self.session_only[key] = text
            if key == "pace":
                speed = self.SPEED.get(str(value).lower())
                if speed is not None:
                    await session._update_session(
                        {"audio": {"output": {"speed": speed}}}
                    )
        await session.refresh_memory()

    async def get(self) -> dict[str, str]:
        stored = (
            await self._session.memory.preferences()
            if self._session.memory is not None
            else {}
        )
        return {**stored, **self.session_only}


class Session:
    """A live conversation. Make one with :meth:`Character.connect`."""

    def __init__(
        self,
        character: Character,
        *,
        player_id: str | None = None,
        player_name: str | None = None,
        players: Any = None,
        audio: Any = None,
        input: AudioInput | None = None,  # noqa: A002 - mirrors the audio pair
        output: AudioOutput | None = None,
        context: dict[str, Any] | None = None,
        memory: bool | str = True,
        trace: Trace | str | None = None,
        tools: list[Tool | Callable[..., Any]] | None = None,
        greeting: bool | str | None = None,
        instructions: str | None = None,
        resume: tuple[str, str] | None = None,
        language: str | None = None,
        control: Any = None,
        connect_timeout: float = 20.0,
    ) -> None:
        self.character = character
        self.client = character.client
        self.player_id = player_id
        self.player_name = player_name
        self.events = EventBus(parent=character._bus)
        self.context = LiveContext("scene")
        self._initial_context = context or {}
        self.metrics = SessionMetrics()
        self.session_id: str | None = None
        self.recording_session_id: str | None = None
        self.recorded = False
        self.disclosure: str | None = None
        self.capabilities: dict[str, Any] = {}
        #: (speaker, text) lines as the SDK saw them: "player", a player's name, "event" or the character's name.
        self.transcript: list[tuple[str, str]] = []
        self.preferences = Preferences(self)
        self._extra_instructions = instructions
        self._greeting = character.greeting if greeting is None else greeting
        self._resume = (
            {"resume": resume[0], "resume_token": resume[1]} if resume else None
        )
        self._connect_timeout = connect_timeout
        #: The language this session is pinned to, when it is: a code or a name.
        self.language = language
        control_id = (
            control
            if isinstance(control, str) or control is None
            else getattr(control, "control_id", None)
        )
        if control is not None and not control_id:
            raise ConfigurationError(
                "This client secret has no control channel: mint it with control=True."
            )
        self._control_id: str | None = control_id
        self._control_tools: str = getattr(control, "control_tools", None) or "server"

        if audio is not None:
            if isinstance(audio, tuple):
                input, output = audio
            else:
                input = audio if hasattr(audio, "frames") else input
                output = audio if hasattr(audio, "play") else output
        self.input = input
        self.output = output
        self._out_resampler = (
            Resampler(character.output_sample_rate, output.sample_rate)
            if output is not None and output.sample_rate != character.output_sample_rate
            else None
        )
        self._in_resampler: Resampler | None = None
        self._send_resamplers: dict[int, Resampler] = {}

        #: The people sharing this conversation, when several do.
        self.players: list[Player] = as_players(players)
        self._player_ids = {player.name: player.key for player in self.players}
        self._languages = {
            player.name: describe(player.language)[1]
            for player in self.players
            if player.language
        }
        self._roster = SpeakerRoster(list(self._player_ids)) if self.players else None
        #: A Room replaces voice-label guessing with who was actually loudest.
        self._attributor: Callable[[str], list[AttributedTurn]] | None = None
        self._last_player: str | None = player_id

        mode = (
            "off" if memory is False else "read" if memory == "read" else "read_write"
        )
        self._memory_mode = mode if character.memory is not None else "off"
        self.memory: PlayerMemory | None = (
            character.memory.for_player(player_id)
            if character.memory is not None
            and player_id is not None
            and self._memory_mode != "off"
            else None
        )
        self._memory_records: list[MemoryRecord] = []
        self._memory_task: asyncio.Task[Any] | None = None

        if isinstance(trace, str):
            trace = Trace(trace)
        self.trace = trace

        self._tools = ToolRunner(
            [*character.tools, *as_tools(tools or [])],
            context=self._tool_context,
            on_event=self._tool_event,
        )
        self._publisher = ContextPublisher(
            self._send_context,
            [character.context, self.context],
            on_sent=self._context_sent,
        )
        self._initiative: Initiative | None = None

        self._ws: Any = None
        self._send_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._ended = asyncio.Event()
        self._closing = False
        self._established = False
        self._ready_count = 0
        self._configured = False
        self._configure_timer: asyncio.TimerHandle | None = None
        self._conn_task: asyncio.Task[Any] | None = None
        self._shutdown_task: asyncio.Task[Any] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._resumed = False
        self._limit: dict[str, str] | None = None
        self._started_at = time.monotonic()
        self._muted = False
        self._end_reason: tuple[str, int | None, str | None] | None = None

        self._responses: dict[str, _Response] = {}
        self._current: str | None = None
        self._cancelled: deque[str] = deque(maxlen=64)
        self._active: set[str] = set()
        self._pending_continue: set[str] = set()
        self._item_waiters: dict[str, asyncio.Future[None]] = {}
        self._user_speaking = False
        self._speech_stopped_at: float | None = None
        self._reply_timer: asyncio.TimerHandle | None = None
        self._hangup_requested = False
        self._interrupted_ids: set[str] = set()
        self._play_rid = ""
        self._play_received = 0
        self._play_discarded = 0
        self._play_flushed = False
        self._partial = ""
        #: The last player turn's place in the audio sent, in ms: (start, end).
        self.speech_window: tuple[int | None, int | None] = (None, None)

        # Speaking first.
        self._proactive = character.proactive
        self._triggers: list[Trigger] = [
            trigger.fresh() for trigger in character.triggers
        ]
        self._trigger_unsubscribe: list[Callable[[], None]] = []
        self._last_activity = time.monotonic()
        self._idle_remarks = 0
        self._last_idle_check = float("-inf")
        self._idle_pending = False
        #: The SDK's own out-of-band decisions: never shown as replies.
        self._internal: dict[str, list[str]] = {}

    def __repr__(self) -> str:
        return f"Session({self.character.name!r}, player={self.player_id!r}, id={self.session_id!r})"

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def __aenter__(self) -> Session:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and not self._closing

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    async def start(self) -> None:
        """Dial, configure the character, send context and memory. Returns when ready."""
        if self._conn_task is not None:
            return
        self._started_at = time.monotonic()
        if self.trace is not None:
            self.trace.open()
            self.events.on("*", self.trace.event)
        self.character._sessions.add(self)
        try:
            # A controlled session's configuration was fixed when its secret was minted.
            self._config = (
                {} if self._control_id else await self.character._session_config(self)
            )
            if self._initial_context:
                await self.context.update(self._initial_context)
            await self._load_memory()
            self._refresh_party()
        except BaseException:
            self.character._sessions.discard(self)
            raise
        self._initiative = Initiative(self._speak_moment)
        self._conn_task = asyncio.create_task(
            self._connection_loop(), name="eesi-session"
        )
        ready = asyncio.create_task(self._ready.wait())
        try:
            done, _ = await asyncio.wait(
                {ready, self._conn_task},
                timeout=self._connect_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            ready.cancel()
        if self._ready.is_set():
            if not self._control_id:
                self._spawn(self._pump_audio(), "microphone")
                self._spawn(self._report_status(), "status")
            if self._proactive is not None and self._proactive.idle_after:
                self._spawn(self._idle_watch(), "proactive")
            self._trigger_unsubscribe = [
                self.character.context.subscribe(self._on_state_change),
                self.context.subscribe(self._on_state_change),
            ]
            return
        if self._conn_task in done:
            error = self._conn_task.exception()
            await self.close(learn=False)
            if error is not None:
                raise error
            raise RealtimeConnectionError("The session ended before it was ready.")
        await self.close(learn=False)
        raise RealtimeConnectionError(
            f"The character was not ready within {self._connect_timeout:.0f} s."
        )

    async def run(self) -> None:
        """Wait until the conversation ends (you close it, the character hangs up, or the connection is lost)."""
        if self._conn_task is None:
            await self.start()
        try:
            await self._ended.wait()
        except asyncio.CancelledError:
            await asyncio.shield(self.close())
            raise

    async def wait(
        self,
        kind: str,
        timeout: float | None = None,
        predicate: Callable[[Event], bool] | None = None,
    ) -> Event:
        """The next event of *kind* (``"response.completed"``, ``"tool.*"``)."""
        future: asyncio.Future[Event] = asyncio.get_running_loop().create_future()

        def check(event: Event) -> None:
            if not future.done() and (predicate is None or predicate(event)):
                future.set_result(event)

        self.events.on(kind, check)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self.events.off(check)

    def on(self, kind: Any = "*", handler: Any = None) -> Any:
        """Subscribe to this session's events; see :class:`eesi.events.EventBus`."""
        return self.events.on(kind, handler)

    def events_iter(self, *, include_audio: bool = False) -> AsyncIterator[Event]:
        return self.events.stream(include_audio=include_audio)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self.events.stream()

    async def close(self, *, learn: bool = True) -> None:
        """Hang up, learn from the conversation, end every event stream."""
        if self._shutdown_task is None:
            self._end_reason = self._end_reason or ("closed", None, None)
            self._shutdown_task = asyncio.create_task(self._teardown(learn=learn))
        await asyncio.shield(self._shutdown_task)

    async def _teardown(self, *, learn: bool) -> None:
        self._closing = True
        self._ready.clear()
        if self._reply_timer is not None:
            self._reply_timer.cancel()
        if self._configure_timer is not None:
            self._configure_timer.cancel()
        if self._initiative is not None:
            self._initiative.dispose()
        for unsubscribe in self._trigger_unsubscribe:
            unsubscribe()
        self._trigger_unsubscribe = []
        ws = self._ws
        if ws is not None:
            try:
                await ws.close(1000, "closed")
            except Exception:
                pass
        conn = self._conn_task
        if conn is not None and conn is not asyncio.current_task() and not conn.done():
            try:
                await asyncio.wait_for(asyncio.shield(conn), 5.0)
            except (asyncio.TimeoutError, Exception):
                conn.cancel()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                task.cancel()
        await asyncio.gather(
            *(t for t in self._tasks if t is not asyncio.current_task()),
            return_exceptions=True,
        )
        if self.output is not None:
            self.output.flush()
        await self._tools.close()
        await self._publisher.close()
        self.context.close()
        reason, code, message = self._end_reason or ("closed", None, None)
        if self._established:
            self._emit(SessionEnded(reason=reason, code=code, message=message))
        if self.trace is not None:
            self.trace.write("metrics", **self.metrics.summary())
        if learn and self._established:
            await self._learn()
        self.character._sessions.discard(self)
        self.character._refresh_party_blocks()
        await self.events.drain()
        self.events.close()
        if self.trace is not None:
            self.trace.close()
        closed: set[int] = set()
        for device in (self.output, self.input):
            close = getattr(device, "close", None)
            if device is None or id(device) in closed or not callable(close):
                continue
            closed.add(id(device))
            try:
                close()
            except Exception:
                log.exception("closing an audio device failed")
        self._ended.set()

    async def _release(self) -> None:
        """Drop what a session that never connected holds: its context subscriptions."""
        await self._publisher.close()
        self.context.close()

    def _finish(
        self, reason: str, code: int | None = None, message: str | None = None
    ) -> None:
        """The conversation ended from the far side."""
        if self._shutdown_task is None:
            self._end_reason = (reason, code, message)
            self._shutdown_task = asyncio.create_task(self._teardown(learn=True))

    # ── what the application can do ──────────────────────────────────────

    def _require_live(self) -> None:
        if self._closing or self._ended.is_set():
            raise SessionClosedError("This session has ended.")

    async def send_text(
        self, text: str, *, respond: bool = True, item_id: str | None = None
    ) -> None:
        """Type into the conversation as the player, and (by default) ask for a reply."""
        self._require_live()
        self.transcript.append(("player", text))
        self._last_player = self.player_id
        party = self.character.party
        if party is not None:
            party.heard(
                self._name_of(self.player_id) or "A player", text, origin=self._origin
            )
            self.character._refresh_party_blocks()
        await self._send(wire.user_text(text, item_id=item_id))
        if respond and self.character.turn_taking != "director":
            await self._request_reply(wire.response_create())

    async def ask(self, text: str, *, timeout: float = 60.0) -> ResponseCompleted:
        """Type *text* as the player and return the character's spoken answer to it.

        The answer is the first reply that starts after the line joined the
        conversation (``conversation.item.added`` names it), so a greeting
        still being generated is never mistaken for it, and a reply that only
        called tools is followed to the one that speaks.
        """
        loop = asyncio.get_running_loop()
        item_id = f"item_sdk_{uuid.uuid4().hex[:20]}"
        added: asyncio.Future[None] = loop.create_future()
        answer: asyncio.Future[ResponseCompleted] = loop.create_future()
        ours: set[str] = set()
        self._item_waiters[item_id] = added

        def on_started(event: Event) -> None:
            if added.done():
                ours.add(event.response_id)  # type: ignore[attr-defined]

        def on_completed(event: Event) -> None:
            if event.response_id in ours and event.text and not answer.done():  # type: ignore[attr-defined]
                answer.set_result(event)  # type: ignore[arg-type]

        def assume_added() -> None:
            # A gateway that never acknowledges items: fall back to timing.
            if not added.done():
                added.set_result(None)

        self.events.on("response.started", on_started)
        self.events.on("response.completed", on_completed)
        fallback = loop.call_later(ITEM_ACK_FALLBACK_S, assume_added)
        try:
            await self.send_text(text, item_id=item_id)
            return await asyncio.wait_for(answer, timeout)
        finally:
            fallback.cancel()
            self._item_waiters.pop(item_id, None)
            self.events.off(on_started)
            self.events.off(on_completed)

    async def send_audio(
        self, pcm: bytes, *, sample_rate: int = wire.INPUT_SAMPLE_RATE
    ) -> None:
        """Microphone audio you capture yourself: PCM16 mono, sent at the pace it was recorded."""
        self._require_live()
        if self._control_id:
            raise ConfigurationError(
                "A controlled session has no audio here: the player's client streams it."
            )
        if sample_rate != wire.INPUT_SAMPLE_RATE:
            resampler = self._send_resamplers.setdefault(
                sample_rate, Resampler(sample_rate, wire.INPUT_SAMPLE_RATE)
            )
            pcm = resampler.process(pcm)
        if pcm and self._ready.is_set():
            self.metrics.audio_in_seconds += len(pcm) / 2 / wire.INPUT_SAMPLE_RATE
            await self._send(wire.append_audio(pcm))

    async def respond(self, instructions: str | None = None) -> None:
        """Ask the character to speak now (or as soon as its current reply ends), optionally steered."""
        self._require_live()
        await self._request_reply(wire.response_create(instructions))

    async def say(self, line: str) -> None:
        """Have the character say *line* (as nearly verbatim as the model allows), in its voice."""
        self._require_live()
        await self._request_reply(
            wire.response_create(
                f'Say exactly this, in character and in your own voice, and nothing else: "{line}"'
            )
        )

    async def _request_reply(self, event: dict[str, Any]) -> None:
        """Ask for a reply. The runtime holds a request made during a reply until that reply ends.

        There is one speaker: a ``response.create`` while a reply is open waits
        in the runtime's single slot, and a newer request replaces a waiting
        one (``response_request_superseded``), so two quick lines get one
        answer that has heard both.
        """
        await self._send(event)

    async def cue(self, event: str, instructions: str | None = None) -> None:
        """Something happened in the application; the character reacts as soon as the floor is free."""
        self._require_live()
        self.transcript.append(("event", event))
        assert self._initiative is not None
        self._initiative.offer(
            Moment(
                kind=f"cue:{time.monotonic()}",
                what=event,
                instructions=instructions,
                urgent=True,
                ttl=15.0,
            )
        )

    def moment(
        self,
        what: str,
        *,
        kind: str = "moment",
        urgent: bool = False,
        cooldown: float = 0.0,
        ttl: float = 8.0,
        instructions: str | None = None,
    ) -> None:
        """Offer something the character *may* remark on, paced so it never talks over anyone.

        Only the newest moment of each kind waits; at most one unprompted
        remark per few seconds unless ``urgent``.
        """
        self._require_live()
        assert self._initiative is not None
        self._initiative.offer(
            Moment(
                kind=kind,
                what=what,
                urgent=urgent,
                cooldown=cooldown,
                ttl=ttl,
                instructions=instructions,
            )
        )

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
        """Have the character remark when live state turns *condition* true (this session only).

        ``condition`` gets :meth:`state`, every live field of the character's
        and this session's context. It fires once each time it becomes true.
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
        self._triggers.append(trigger)
        return trigger

    def state(self) -> dict[str, Any]:
        """Every live field the character can see: its own context, then this session's."""
        return {**self.character.context.all_fields(), **self.context.all_fields()}

    def _on_state_change(self, _block: str) -> None:
        if not self._ready.is_set() or self._closing or not self._triggers:
            return
        state = self.state()
        for trigger in self._triggers:
            remark = trigger.check(state)
            if remark:
                self.moment(
                    remark,
                    kind=trigger.kind,
                    urgent=trigger.urgent,
                    cooldown=trigger.cooldown,
                    instructions=trigger.instructions,
                )

    def _is_internal(self, event: dict[str, Any]) -> bool:
        response = (
            event.get("response") if isinstance(event.get("response"), dict) else {}
        )
        metadata = (
            response.get("metadata")
            if isinstance(response.get("metadata"), dict)
            else {}
        )
        if metadata.get(INTERNAL_TAG):
            return True
        # A runtime that does not echo metadata still marks an out-of-band reply.
        return (
            self._idle_pending
            and "conversation_id" in response
            and response.get("conversation_id") is None
        )

    def _floor_free(self) -> bool:
        return (
            not self._user_speaking
            and not self._active
            and not self._audible()
            and not self._closing
        )

    async def _idle_watch(self) -> None:
        assert self._proactive is not None and self._proactive.idle_after
        proactive = self._proactive
        while not self._closing:
            await asyncio.sleep(IDLE_POLL_S)
            now = time.monotonic()
            if (
                not self._ready.is_set()
                or self._idle_pending
                or not self._floor_free()
                or now - self._last_activity < proactive.idle_after  # type: ignore[operator]
                or now - self._last_idle_check < proactive.cooldown
                or (
                    proactive.max_remarks is not None
                    and self._idle_remarks >= proactive.max_remarks
                )
            ):
                continue
            await self._idle_check()

    async def _idle_check(self) -> None:
        """Ask the character, out of band, whether a quiet moment is worth a remark."""
        assert self._proactive is not None
        self._idle_pending = True
        self._last_activity = time.monotonic()
        # The cooldown runs between checks, said or not: each one is a model call.
        self._last_idle_check = time.monotonic()
        await self._send(
            {
                "type": "response.create",
                "event_id": wire.event_id(),
                "response": {
                    "conversation": "none",
                    "output_modalities": ["text"],
                    "metadata": {INTERNAL_TAG: "idle_check"},
                    "instructions": self._proactive.prompt(),
                },
            }
        )

    async def _idle_decided(self, text: str) -> None:
        self._idle_pending = False
        self._last_activity = time.monotonic()
        line = text.strip().strip('"').strip()
        if not line or line.upper().rstrip(".!") == IDLE_SKIP or len(line) > 400:
            return
        if not self._floor_free():
            return  # Someone spoke while it was deciding; the moment has passed.
        self._idle_remarks += 1
        await self._send(
            wire.response_create(f'Say this now, in your own voice, as it is: "{line}"')
        )

    async def interrupt(self) -> None:
        """Stop the character mid-sentence."""
        self._require_live()
        if self._current and self._current in self._active:
            self._cut(self._current, "client_cancelled")
        await self._send(wire.response_cancel())

    def mute(self, muted: bool = True) -> None:
        """Stop (or resume) sending the microphone: push-to-talk, a pause menu."""
        self._muted = muted

    async def remember(self, text: str, **options: Any) -> MemoryRecord:
        """Keep something about this player for later sessions."""
        if self.memory is None:
            raise SessionClosedError(
                "This session has no player memory (no player_id, or memory is off)."
            )
        record = await self.memory.remember(text, session_id=self.session_id, **options)
        self._emit(MemoryUpdated(added=(record,), source=options.get("source", "app")))
        await self.refresh_memory()
        return record

    async def set_mood(self, mood: str | None, *, reason: str | None = None) -> None:
        """A passing emotional state that colours how the character speaks; never who it is."""
        if not mood:
            await self.context.set("mood", None)
            return
        because = f" ({reason})" if reason else ""
        await self.context.set(
            "mood",
            f"Your mood right now: {mood}{because}. Let it colour how you speak in this moment; it does not change who you are.",
        )

    async def refresh_memory(self, query: str | None = None) -> list[MemoryRecord]:
        """Recompose the memory block, optionally around *query*."""
        text, records = await self._memory_block(query)
        self._publisher.set_block("memory", text)
        if records:
            self._memory_records = records
        return records

    async def feedback(
        self,
        rating: int | str,
        *,
        note: str | None = None,
        response_id: str | None = None,
    ) -> None:
        """Record how a reply (or the session) went, for your own evaluation. Nothing is trained on it."""
        payload = {
            "rating": rating,
            "note": note,
            "response_id": response_id or self._current,
        }
        if self.trace is not None:
            self.trace.write("feedback", **payload)
        self._emit(
            Feedback(rating=rating, note=note, response_id=payload["response_id"])
        )

    def inspect(self) -> dict[str, Any]:
        """What the character is working from right now: instructions, context blocks, tools, memories."""
        return {
            "character": self.character.name,
            "player_id": self.player_id,
            "session_id": self.session_id,
            "instructions": self._config.get("instructions")
            if hasattr(self, "_config")
            else None,
            "voice": ((self._config.get("audio") or {}).get("output") or {}).get(
                "voice"
            )
            if hasattr(self, "_config")
            else None,
            "turn_taking": self.character.turn_taking,
            "context": self._publisher.current(),
            "context_pending": self._publisher.desired(),
            "tools": [spec["name"] for spec in self._tools.specs()],
            "memories": [record.to_dict() for record in self._memory_records],
            "metrics": self.metrics.summary(),
        }

    @property
    def tools(self) -> ToolRunner:
        return self._tools

    async def set_tools_enabled(self, name: str, enabled: bool) -> None:
        """Offer or withdraw one tool for this session only (a permission that changed mid-game)."""
        if enabled:
            self._tools.disabled.discard(name)
        else:
            self._tools.disabled.add(name)
        specs = self._tools.specs()
        await self._update_session(
            {"tools": specs, "tool_choice": "auto" if specs else "none"}
        )

    async def _update_session(self, patch: dict[str, Any]) -> None:
        """A partial session.update that keeps the character's voice.

        Gateways before October 2026 pin the platform's default voice on a
        session.update that names none, so a speed-only or tools-only frame
        would otherwise change who is speaking there.
        """
        output = dict((self._config.get("audio") or {}).get("output") or {})
        merged = _merge(_merge({}, {"audio": {"output": output}}), patch)
        self._config = _merge(self._config, patch)
        await self._send(wire.session_update(merged))

    # ── connection ────────────────────────────────────────────────────────

    def _url(self) -> str:
        if self._control_id:
            return f"{self.client.realtime_url}/control/{self._control_id}"
        query = {"model": self.character.model, "source": "sdk"}
        if self.client.timezone:
            query["timezone"] = self.client.timezone
        if self._resume:
            query.update(self._resume)
        return f"{self.client.realtime_url}?{urlencode(query)}"

    async def _connection_loop(self) -> None:
        try:
            await self._dial()
        except Exception as error:
            if not self._established:
                raise
            log.warning("realtime connection failed: %s", error)
            self._emit(ErrorEvent(message=str(error), code="connection_failed"))
            self._finish("connection_lost", None, str(error))

    async def _dial(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        attempt = 0
        dial_retries = 0
        readies = 0
        while not self._closing:
            code: int | None = None
            reason = ""
            try:
                async with connect(
                    self._url(),
                    additional_headers=self.client._auth_headers(),
                    user_agent_header=self.client.user_agent,
                    open_timeout=15,
                    max_size=16 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=3,
                ) as ws:
                    self._ws = ws
                    try:
                        async for raw in ws:
                            if isinstance(raw, bytes):
                                continue
                            try:
                                event = json.loads(raw)
                            except ValueError:
                                continue
                            if isinstance(event, dict):
                                try:
                                    await self._handle(event)
                                except Exception:
                                    log.exception(
                                        "handling %s failed", event.get("type")
                                    )
                    except ConnectionClosed:
                        pass
                    code = ws.close_code
                    reason = ws.close_reason or ""
            except InvalidStatus as error:
                status = error.response.status_code
                if status in (401, 403):
                    raise AuthenticationError(
                        f"The gateway refused the API key (HTTP {status}). Check EESI_API_KEY and its organization."
                    ) from None
                if (
                    (status == 429 or status >= 500)
                    and not self._established
                    and dial_retries < 3
                ):
                    dial_retries += 1
                    await asyncio.sleep(_reconnect_delay(dial_retries))
                    continue
                if not self._established:
                    raise RealtimeConnectionError(
                        f"The gateway refused the session (HTTP {status}).",
                        kind="unavailable",
                    ) from None
                code = 1006
            except (OSError, asyncio.TimeoutError) as error:
                if not self._established:
                    if dial_retries < 2:
                        dial_retries += 1
                        await asyncio.sleep(_reconnect_delay(dial_retries))
                        continue
                    raise RealtimeConnectionError(
                        f"Could not reach {self.client.realtime_url}: {error}"
                    ) from None
                code = 1006
            finally:
                self._ws = None
                self._ready.clear()
                self._configured = False
                self._publisher.stop()
                self._drop_audio("connection")
            if self._closing:
                return
            if self._ready_count > readies:
                # This connection came up: a later drop starts a fresh reconnect budget.
                readies = self._ready_count
                attempt = 0
            if not self._established:
                if (
                    code == 1013 and dial_retries < 3
                ):  # a draining replica: dial another
                    dial_retries += 1
                    continue
                kind = (
                    "at-limit"
                    if code == 1008
                    else "unavailable"
                    if code == 1011
                    else "dropped"
                )
                if code == 1008 and "auth" in reason.lower():
                    raise AuthenticationError(
                        f"The gateway refused the credential: {reason}"
                    )
                raise RealtimeConnectionError(
                    f"The session closed before it was ready (code {code}{': ' + reason if reason else ''}).",
                    code=code,
                    kind=kind,
                )
            if self._limit is not None:
                self._finish("limit", code, self._limit.get("message"))
                return
            if self._hangup_requested:
                self._finish("hangup", code)
                return
            if code == 1000 or code in FATAL_CLOSE_CODES:
                self._finish("server_closed", code, reason or None)
                return
            if (
                not self._resume and not self._control_id
            ) or attempt >= MAX_RECONNECT_ATTEMPTS:
                self._finish("connection_lost", code, reason or None)
                return
            delay = _reconnect_delay(attempt)
            attempt += 1
            self.metrics.reconnects += 1
            self._emit(Reconnecting(attempt=attempt, delay=round(delay, 2), code=code))
            await asyncio.sleep(delay)

    async def _send(self, event: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        from websockets.exceptions import ConnectionClosed

        data = json.dumps(event, ensure_ascii=False)
        async with self._send_lock:
            try:
                await ws.send(data)
            except ConnectionClosed:
                pass

    def _spawn(self, coroutine: Any, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(self._guarded(coroutine, name), name=f"eesi-{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _guarded(self, coroutine: Any, name: str) -> None:
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.exception("session task %s failed", name)
            self._emit(
                ErrorEvent(message=f"{name} failed: {error}", code="sdk_task_failed")
            )

    def _stamp(self) -> dict[str, Any]:
        return {
            "at": round(time.monotonic() - self._started_at, 3),
            "session_id": self.session_id,
            "player_id": self.player_id,
        }

    def _emit(self, event: Event) -> None:
        if event.session_id is None and event.at == 0.0:
            event = type(event)(**{**_fields(event), **self._stamp()})
        self.events.emit(event)

    # ── the tasks ─────────────────────────────────────────────────────────

    async def _pump_audio(self) -> None:
        if self.input is None:
            return
        if self.input.sample_rate != wire.INPUT_SAMPLE_RATE:
            self._in_resampler = Resampler(
                self.input.sample_rate, wire.INPUT_SAMPLE_RATE
            )
        async for frame in self.input.frames():
            if self._closing:
                return
            if not self._ready.is_set() or self._muted:
                continue
            if self._in_resampler is not None:
                frame = self._in_resampler.process(frame)
            if frame:
                self.metrics.audio_in_seconds += len(frame) / 2 / wire.INPUT_SAMPLE_RATE
                await self._send(wire.append_audio(frame))

    async def _report_status(self) -> None:
        last_keepalive = time.monotonic()
        while not self._closing:
            await asyncio.sleep(STATUS_INTERVAL_S)
            if not self._ready.is_set():
                continue
            if self.output is not None and self._play_rid:
                buffered = self.output.buffered_seconds()
                pending = round(buffered * self.character.output_sample_rate)
                await self._send(
                    wire.output_status(
                        response_id=self._play_rid,
                        sample_rate=self.character.output_sample_rate,
                        received=self._play_received,
                        rendered=max(
                            0, self._play_received - self._play_discarded - pending
                        ),
                        discarded=self._play_discarded,
                        flushed=self._play_flushed,
                        buffered_ms=round(buffered * 1000),
                    )
                )
            if time.monotonic() - last_keepalive > KEEPALIVE_INTERVAL_S:
                last_keepalive = time.monotonic()
                await self._send(wire.keepalive())

    # ── server events ─────────────────────────────────────────────────────

    async def _handle(self, event: dict[str, Any]) -> None:
        kind = wire.canonical_type(str(event.get("type") or ""))
        now = time.monotonic()
        if kind == "eesi.recording_disclosure":
            disclosure = (
                event.get("disclosure")
                if isinstance(event.get("disclosure"), dict)
                else {}
            )
            self.recorded = disclosure.get("recorded") is True
            self.disclosure = disclosure.get("text") if self.recorded else None
        elif kind == "eesi.session":
            self._on_session_handle(event)
        elif kind == "session.created":
            capabilities = event.get("capabilities")
            self.capabilities = capabilities if isinstance(capabilities, dict) else {}
            if self._control_id:
                return  # The client's session, configured by its secret.
            self._configured = False
            await self._send(wire.session_update(self._config))
            loop = asyncio.get_running_loop()
            self._configure_timer = loop.call_later(
                CONFIGURE_TIMEOUT_S, self._configure_timeout
            )
        elif kind == "eesi.control.attached":
            session_id = event.get("session_id")
            if isinstance(session_id, str) and session_id:
                self.session_id = session_id
            recording = event.get("recording_session_id")
            if isinstance(recording, str) and recording:
                self.recording_session_id = recording
            if event.get("tools") in ("server", "client"):
                self._control_tools = str(event["tools"])
            if not self._ready.is_set():
                await self._on_ready()
        elif kind == "eesi.control.ended":
            self._finish("ended", None, str(event.get("reason") or "") or None)
        elif kind == "session.updated" and not self._control_id:
            if not self._configured:
                self._configured = True
                if self._configure_timer is not None:
                    self._configure_timer.cancel()
                    self._configure_timer = None
                await self._on_ready()
        elif kind == "input_audio_buffer.speech_started":
            self._on_speech_started(event)
        elif kind == "input_audio_buffer.speech_stopped":
            self._user_speaking = False
            self._speech_stopped_at = now
            self._last_activity = now
            end_ms = event.get("audio_end_ms")
            end_ms = end_ms if isinstance(end_ms, int) else None
            self.speech_window = (self.speech_window[0], end_ms)
            if self._initiative is not None:
                self._initiative.user_stopped()
            self._emit(AudioInputEnded(audio_end_ms=end_ms))
        elif kind == "conversation.item.input_audio_transcription.delta":
            text = event.get("delta")
            if isinstance(text, str) and text and text != self._partial:
                self._partial = text
                self._emit(TranscriptPartial(text=text, item_id=event.get("item_id")))
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._partial = ""
            text = event.get("transcript")
            if isinstance(text, str) and text.strip():
                self._on_user_turn(text.strip(), event.get("item_id"))
        elif kind == "response.created":
            rid = wire.response_id_of(event)
            if self._is_internal(event):
                self._internal[rid] = []
                return
            self._last_activity = now
            response = self._responses.setdefault(rid, _Response(rid, now))
            response.turn_ended_at = self._speech_stopped_at
            self._speech_stopped_at = None
            self._current = rid
            self._active.add(rid)
            if self._initiative is not None:
                self._initiative.response_started()
            self._emit(ResponseStarted(response_id=rid))
        elif kind == "response.output_audio.delta":
            self._on_audio(event)
        elif kind in (
            "response.output_audio_transcript.delta",
            "response.output_text.delta",
        ):
            delta = event.get("delta")
            rid = wire.response_id_of(event) or self._current or ""
            if rid in self._internal:
                if isinstance(delta, str):
                    self._internal[rid].append(delta)
                return
            if isinstance(delta, str) and delta and rid not in self._cancelled:
                self._responses.setdefault(rid, _Response(rid, now)).deltas.append(
                    delta
                )
                self._emit(ResponseTextDelta(response_id=rid, text=delta))
        elif kind in (
            "response.output_audio_transcript.done",
            "response.output_text.done",
        ):
            rid = wire.response_id_of(event) or self._current or ""
            if rid in self._internal:
                text = event.get("text") or event.get("transcript")
                if isinstance(text, str) and text.strip():
                    self._internal[rid] = [text]
                return
            response = self._responses.setdefault(rid, _Response(rid, now))
            text = (
                event.get("transcript")
                if kind.endswith("transcript.done")
                else event.get("text")
            )
            event_id = event.get("event_id")
            if isinstance(event_id, str) and event_id in response.segment_ids:
                return
            if isinstance(event_id, str):
                response.segment_ids.add(event_id)
            if isinstance(text, str) and text.strip():
                response.segments.append(text)
        elif kind == "output_audio_buffer.cleared":
            rid = event.get("response_id") or self._current
            if rid not in self._cancelled:
                self._cut(rid, "cleared")
        elif kind == "eesi.output.resume":
            duck = getattr(self.output, "duck", None)
            if callable(duck):
                duck(False)
        elif kind == "response.function_call_arguments.done":
            self._on_tool_call(event)
        elif kind in ("conversation.item.added", "conversation.item.created"):
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            waiter = self._item_waiters.get(str(item.get("id") or ""))
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
        elif kind == "response.done":
            rid = wire.response_id_of(event)
            if rid in self._internal:
                await self._idle_decided("".join(self._internal.pop(rid)).strip())
                return
            await self._on_response_done(event)
        elif kind == "eesi.hangup":
            self._on_hangup(str(event.get("reason") or "model_requested"))
        elif kind == "error":
            message, code = wire.error_of(event)
            if code == "response_request_superseded":
                # A newer request took a waiting one's place: normal turn-taking, not a failure.
                log.debug("reply request superseded: %s", message)
                return
            if (
                code == "invalid_context_update"
                and "eesi.context.update" in message
                and ("unknown" in message.lower())
            ):
                self._publisher.refused()
            self._emit(ErrorEvent(message=message, code=code))

    def _on_session_handle(self, event: dict[str, Any]) -> None:
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            self.session_id = session_id
        recording = event.get("recording_session_id")
        if isinstance(recording, str) and recording:
            self.recording_session_id = recording
        resume = event.get("resume") if isinstance(event.get("resume"), dict) else {}
        token = (
            event.get("resume_token")
            or resume.get("token")
            or resume.get("resume_token")
        )
        if (
            resume.get("supported") is True
            and isinstance(token, str)
            and token
            and self.session_id
        ):
            self._resume = {"resume": self.session_id, "resume_token": token}
        else:
            self._resume = None
        resumed = event.get("resumed")
        self._resumed = isinstance(resumed, dict)
        limit = event.get("limit")
        if isinstance(limit, dict) and limit.get("code"):
            self._limit = {
                "code": str(limit.get("code")),
                "message": str(limit.get("message") or ""),
            }
            self._emit(
                ErrorEvent(
                    message=self._limit["message"] or "The session reached a limit.",
                    code=self._limit["code"],
                )
            )

    def _configure_timeout(self) -> None:
        self._configure_timer = None
        if self._configured or self._closing:
            return
        self._emit(
            ErrorEvent(
                message="The gateway did not accept the character's configuration (no session.updated). "
                "Check the instructions size, the voice and the tool schemas.",
                code="configure_timeout",
            )
        )

    async def _on_ready(self) -> None:
        try:
            await self._publisher.start()
        except ContextError as error:
            self._emit(ErrorEvent(message=str(error), code="invalid_context_update"))
        first = not self._established
        self._established = True
        self._ready_count += 1
        self._last_activity = time.monotonic()
        self._ready.set()
        if first:
            self._emit(
                SessionStarted(
                    model=self.character.model,
                    resumed=self._resumed,
                    recorded=self.recorded,
                    disclosure=self.disclosure,
                )
            )
            if self._memory_records:
                self._emit(
                    MemoryRetrieved(query=None, records=tuple(self._memory_records))
                )
            if self._greeting and not self._resumed:
                known = any(
                    record.player_id is not None for record in self._memory_records
                )
                stranger = self.memory is not None and not known
                await self._send(
                    wire.response_create(
                        self.character._greeting_instructions(
                            self._greeting,
                            known=known,
                            stranger=stranger,
                            name=self.player_name,
                        )
                    )
                )
        else:
            self._emit(Reconnected(resumed=self._resumed))

    def _on_speech_started(self, event: dict[str, Any]) -> None:
        interrupt = event.get("interrupt_response")
        audible = self._audible()
        self._user_speaking = True
        self._speech_stopped_at = None
        self._last_activity = time.monotonic()
        start_ms = event.get("audio_start_ms")
        start_ms = start_ms if isinstance(start_ms, int) else None
        self.speech_window = (start_ms, None)
        if self._reply_timer is not None:
            self._reply_timer.cancel()
            self._reply_timer = None
        if self._initiative is not None:
            self._initiative.user_started()
        self._emit(AudioInputStarted(while_speaking=audible, audio_start_ms=start_ms))
        if interrupt is False and audible:
            # Held, not cut: the runtime is deciding whether that was a
            # backchannel. Duck until it resumes or cuts.
            duck = getattr(self.output, "duck", None)
            if callable(duck):
                duck(True)
            return
        if (
            interrupt is not False
            and not event.get("eesi_late")
            and (audible or (self._current in self._active))
        ):
            self._cut(self._current, "barge_in")

    def _on_user_turn(self, text: str, item_id: Any) -> None:
        self.metrics.turns += 1
        if self._attributor is not None or self._roster is not None:
            attributed = (
                self._attributor(text)
                if self._attributor is not None
                else self._roster.attribute(text)  # type: ignore[union-attr]
            )
            turns = tuple(
                SpeakerTurn(
                    text=a.text,
                    speaker=a.speaker,
                    player_id=self._player_ids.get(a.player or "", a.player),
                )
                for a in attributed
            )
            for turn in turns:
                if turn.player_id:
                    self._last_player = turn.player_id
            if self._roster is not None:
                self._publisher.set_block(
                    "roster", self._roster.render(self._languages)
                )
        else:
            turns = tuple(
                SpeakerTurn(text=words, speaker=speaker, player_id=self.player_id)
                for speaker, words in wire.split_speaker_turns(text)
            )
        for turn in turns:
            label = (
                "player"
                if not self.players
                else (self._name_of(turn.player_id) or f"Speaker {turn.speaker}")
            )
            self.transcript.append((label, turn.text))
            party = self.character.party
            if party is not None:
                party.heard(
                    self._name_of(turn.player_id) or "A player",
                    turn.text,
                    origin=self._origin,
                )
        if self.character.party is not None:
            self.character._refresh_party_blocks()
        self._emit(
            TranscriptFinal(
                text=text,
                item_id=item_id if isinstance(item_id, str) else None,
                turns=turns,
            )
        )
        if (
            self.character.memory is not None
            and self.character.memory.policy.recall_on_turn
            and self._memory_mode != "off"
        ):
            if self._memory_task is None or self._memory_task.done():
                self._memory_task = self._spawn(self._recall_for(text), "memory")
        if self.character.turn_taking == "app":
            loop = asyncio.get_running_loop()
            if self._reply_timer is not None:
                self._reply_timer.cancel()
            self._reply_timer = loop.call_later(
                TURN_SETTLE_S, lambda: self._spawn(self._reply_to_turn(), "reply")
            )

    async def _reply_to_turn(self) -> None:
        self._reply_timer = None
        if self._user_speaking or self._closing:
            return
        if self._memory_task is not None and not self._memory_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._memory_task), RECALL_WAIT_S)
            except (asyncio.TimeoutError, Exception):
                pass
        try:
            await self._publisher.flush()
        except ContextError as error:
            self._emit(ErrorEvent(message=str(error), code="invalid_context_update"))
        if not self._user_speaking:
            await self._send(wire.response_create(tool_choice="auto"))

    def _on_audio(self, event: dict[str, Any]) -> None:
        rid = wire.response_id_of(event) or self._current or ""
        if rid in self._cancelled:
            return
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        pcm = base64.b64decode(delta)
        response = self._responses.setdefault(rid, _Response(rid, time.monotonic()))
        if not response.had_audio:
            response.had_audio = True
            response.first_audio_at = time.monotonic()
            if response.turn_ended_at is not None:
                self.metrics.latencies.append(
                    response.first_audio_at - response.turn_ended_at
                )
        rate = self.character.output_sample_rate
        self.metrics.audio_out_seconds += len(pcm) / 2 / rate
        if rid != self._play_rid:
            self._play_rid = rid
            self._play_received = 0
            self._play_discarded = 0
            self._play_flushed = False
        self._play_received += len(pcm) // 2
        if self.output is not None:
            self.output.play(
                self._out_resampler.process(pcm) if self._out_resampler else pcm
            )
        self._emit(AudioOutputChunk(response_id=rid, pcm=pcm, sample_rate=rate))

    def _audible(self) -> bool:
        if self.output is not None:
            return self.output.buffered_seconds() > 0.0
        return self._current in self._active

    def _drop_audio(self, reason: str) -> int:
        dropped = 0
        if self.output is not None:
            dropped = round(
                self.output.buffered_seconds() * self.character.output_sample_rate
            )
            self.output.flush()
        if self._out_resampler is not None:
            self._out_resampler.reset()
        if dropped:
            self._play_discarded += min(
                dropped, max(0, self._play_received - self._play_discarded)
            )
            self._play_flushed = True
        return dropped

    def _cut(self, rid: str | None, reason: str) -> None:
        """The character stopped talking because someone took the floor: drop its audio."""
        audible = self._audible()
        self._drop_audio(reason)
        if rid:
            if rid not in self._cancelled:
                self._cancelled.append(rid)
            response = self._responses.get(rid)
            if response is not None:
                response.cancelled = True
        if (
            audible or (rid and rid in self._active)
        ) and rid not in self._interrupted_ids:
            if rid:
                self._interrupted_ids.add(rid)
            self.metrics.interruptions += 1
            self._emit(SpeechInterrupted(response_id=rid))

    async def _on_response_done(self, event: dict[str, Any]) -> None:
        rid = wire.response_id_of(event)
        status, reason = wire.response_status(event)
        self._last_activity = time.monotonic()
        response = self._responses.setdefault(rid, _Response(rid, time.monotonic()))
        self._active.discard(rid)
        response.done = True
        cut = status == "cancelled" or reason == "turn_detected"
        text = response.text()
        if cut:
            self._cut(rid, reason or status or "cancelled")
            self._emit(
                ResponseCancelled(response_id=rid, reason=reason or status, text=text)
            )
        elif status in (None, "completed", "incomplete"):
            self.metrics.replies += 1
            latency = None
            if (
                response.first_audio_at is not None
                and response.turn_ended_at is not None
            ):
                latency = round(response.first_audio_at - response.turn_ended_at, 3)
            self._emit(ResponseCompleted(response_id=rid, text=text, latency=latency))
        else:
            self._emit(
                ErrorEvent(
                    message=f"The reply ended with status {status}"
                    + (f" ({reason})" if reason else ""),
                    code=reason or status,
                )
            )
        if text:
            self.transcript.append(
                (self.character.name, text + (" [cut off]" if cut else ""))
            )
            party = self.character.party
            if party is not None and not cut:
                party.said(
                    self.character.name,
                    self._name_of(self._last_player) or "a player",
                    text,
                    origin=self._origin,
                )
                self.character._refresh_party_blocks()
        if self._initiative is not None:
            self._initiative.response_done()
        await self._continue_if_settled(rid)
        for pending in list(self._pending_continue):
            await self._continue_if_settled(pending)
        if len(self._responses) > 64:
            for old in list(self._responses)[:-32]:
                entry = self._responses[old]
                if entry.done and not entry.outstanding:
                    del self._responses[old]

    # ── tools ─────────────────────────────────────────────────────────────

    @property
    def _origin(self) -> str:
        return str(id(self))

    def _name_of(self, player_id: str | None) -> str | None:
        if player_id is None:
            return None
        for name, pid in self._player_ids.items():
            if pid == player_id:
                return name
        if player_id == self.player_id:
            return self.player_name or player_id
        return player_id

    def _tool_context(self, call_id: str, arguments: dict[str, Any]) -> ToolContext:
        return ToolContext(
            session=self,
            character=self.character,
            player_id=self._last_player or self.player_id,
            call_id=call_id,
            arguments=arguments,
        )

    def _tool_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "tool.started":
            self.metrics.tool_calls += 1
            self._emit(ToolStarted(**data))
        elif kind == "tool.completed":
            self._emit(ToolCompleted(**data))
        elif kind == "tool.failed":
            self.metrics.tool_failures += 1
            self._emit(ToolFailed(**data))

    def _on_tool_call(self, event: dict[str, Any]) -> None:
        call_id = event.get("call_id")
        name = event.get("name")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
            return
        executor = event.get("eesi_executor")
        if self._control_id:
            # The server's tools run here; with tools left to the client, the client runs them.
            if executor == "gateway" or self._control_tools != "server":
                return
        elif executor in ("gateway", "server"):
            return
        if self._tools.seen(call_id):
            return
        rid = wire.response_id_of(event) or self._current or ""
        response = self._responses.setdefault(rid, _Response(rid, time.monotonic()))
        response.calls += 1
        response.outstanding.add(call_id)
        arguments = (
            event.get("arguments") if isinstance(event.get("arguments"), str) else "{}"
        )
        self._spawn(self._run_tool(rid, call_id, name, arguments), f"tool:{name}")

    async def _run_tool(
        self, rid: str, call_id: str, name: str, arguments: str
    ) -> None:
        outcome = await self._tools.run(call_id, name, arguments)
        if outcome.later is not None:
            self._follow(name, outcome.later)
        await self._send(wire.tool_output(call_id, outcome.output))
        response = self._responses.get(rid)
        if response is None:
            return
        response.outstanding.discard(call_id)
        response.speak_after = response.speak_after or outcome.speak_after
        await self._continue_if_settled(rid)

    def _follow(self, name: str, later: Any) -> None:
        async def wait() -> None:
            try:
                result = await later
                from .tools import render_result

                news = (
                    f"The {name} you started earlier finished: {render_result(result)}"
                )
            except asyncio.CancelledError:
                news = f"The {name} you started earlier was stopped before it finished."
            except Exception as error:
                news = f"The {name} you started earlier failed: {error}"
            if not self._closing:
                self.transcript.append(("event", news))
                assert self._initiative is not None
                self._initiative.offer(
                    Moment(kind=f"tool:{name}", what=news, urgent=True, ttl=30.0)
                )

        self._spawn(wait(), f"follow:{name}")

    async def _continue_if_settled(self, rid: str) -> None:
        """One continuation after every result of a reply is in and that reply is over.

        A continuation per result would talk over itself when the model asked
        for two tools at once; one before ``response.done`` would be refused;
        one after a reply that already said what the tool did makes the
        character say it twice.
        """
        response = self._responses.get(rid)
        if (
            response is None
            or not response.done
            or response.outstanding
            or response.continued
        ):
            self._pending_continue.discard(rid)
            return
        if self._active:
            self._pending_continue.add(rid)
            return
        self._pending_continue.discard(rid)
        response.continued = True
        if (
            response.cancelled
            or response.calls == 0
            or self._user_speaking
            or self._closing
        ):
            return
        if response.had_audio and not response.speak_after:
            return
        # Plain on purpose: a tool the runtime's thinker asked for belongs to
        # that request, and only a plain create resumes it.
        await self._send(wire.response_create())

    def _speak_moment(self, moment: Moment) -> None:
        self._spawn(
            self._send(wire.event_reply(moment.what, moment_instructions(moment))),
            "moment",
        )

    def _on_hangup(self, reason: str) -> None:
        self._hangup_requested = True
        self._emit(Hangup(reason=reason))
        if not self.character.end_on_hangup:
            return

        async def after_playout() -> None:
            await asyncio.sleep(0.3)
            while self.output is not None and self.output.buffered_seconds() > 0.05:
                await asyncio.sleep(0.1)
            self._end_reason = ("hangup", None, None)
            await self.close()

        self._spawn(after_playout(), "hangup")

    # ── context and memory ────────────────────────────────────────────────

    async def _send_context(self, key: str, text: str) -> None:
        await self._send(wire.context_update(key, text))

    def _context_sent(self, key: str, text: str) -> None:
        self._emit(ContextUpdated(block=key, text=text))

    def _refresh_party(self) -> None:
        party = self.character.party
        if party is not None:
            self._publisher.set_block("party", party.render(exclude=self._origin))

    @property
    def _memory_writable(self) -> bool:
        return self._memory_mode == "read_write"

    async def _memory_block(
        self, query: str | None = None
    ) -> tuple[str, list[MemoryRecord]]:
        memory = self.character.memory
        if memory is None or self._memory_mode == "off":
            return "", []
        if self._player_ids and self.player_id is None:
            parts: list[str] = []
            used: list[MemoryRecord] = []
            for name, pid in self._player_ids.items():
                text, records = await memory.block(
                    player_id=pid, player_name=name, query=query
                )
                if text:
                    parts.append(text)
                used.extend(records)
            return "\n\n".join(parts), used
        return await memory.block(
            player_id=self.player_id,
            player_name=self.player_name,
            query=query,
            session_preferences=self.preferences.session_only,
        )

    async def _load_memory(self) -> None:
        situation = " ".join(self._publisher.desired().values())
        try:
            text, records = await self._memory_block(situation or None)
        except Exception as error:
            log.exception("could not read memory; starting without it")
            self._emit(
                ErrorEvent(
                    message=f"Memory could not be read: {error}",
                    code="memory_unavailable",
                )
            )
            return
        self._memory_records = records
        self._publisher.set_block("memory", text)

    async def _recall_for(self, text: str) -> None:
        """Memories this turn brings to mind, in the ``recall`` block.

        Kept apart from the ``memory`` block, which is chosen once per session
        and stays byte-identical, so a turn's recall never invalidates the
        cached prompt in front of it.
        """
        memory = self.character.memory
        if memory is None:
            return
        shown = {record.id for record in self._memory_records}
        found: list[MemoryRecord] = []
        pids = [
            pid
            for pid in (
                [self.player_id] if self.player_id else list(self._player_ids.values())
            )
        ]
        for pid in pids or [None]:
            for record, score in await memory.search(text, player_id=pid, limit=6):
                if (
                    score >= RECALL_MIN_SCORE
                    and record.id not in shown
                    and all(r.id != record.id for r in found)
                ):
                    found.append(record)
        found = found[:RECALL_MAX]
        lines = [f"- {' '.join(record.text.split())}" for record in found]
        self._publisher.set_block(
            "recall",
            (
                "What the player just said reminds you of (use it only if it matters):\n"
                + "\n".join(lines)
            )
            if lines
            else "",
        )
        if found:
            await memory.store.touch([record.id for record in found], time.time())
            self._emit(MemoryRetrieved(query=text, records=tuple(found)))

    async def _learn(self) -> None:
        memory = self.character.memory
        if memory is None or not self._memory_writable or not memory.policy.extract:
            return
        if not any(
            speaker not in ("event", self.character.name)
            for speaker, _ in self.transcript
        ):
            return
        targets: list[tuple[str, str | None]] = []
        if self.player_id is not None:
            targets.append((self.player_id, self.player_name))
        else:
            targets.extend((pid, name) for name, pid in self._player_ids.items())
        for player_id, name in targets:
            lines = [
                (
                    "player" if (speaker == "player" or speaker == name) else speaker,
                    text,
                )
                for speaker, text in self.transcript
            ]
            try:
                added, superseded, forgotten = await asyncio.wait_for(
                    memory.learn(
                        lines,
                        player_id=player_id,
                        player_name=name,
                        session_id=self.session_id,
                        skip_kinds=()
                        if self.character.personalize
                        else ("preference",),
                    ),
                    LEARN_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                self._emit(
                    ErrorEvent(
                        message="Learning from the conversation took too long; nothing was kept.",
                        code="memory_extraction_timeout",
                    )
                )
                continue
            except Exception as error:
                log.warning("memory extraction failed: %s", error)
                self._emit(
                    ErrorEvent(
                        message=f"Learning from the conversation failed: {error}",
                        code="memory_extraction_failed",
                    )
                )
                continue
            if added or superseded or forgotten:
                event = MemoryUpdated(
                    added=tuple(added),
                    updated=tuple(superseded),
                    removed=tuple(forgotten),
                    source="extraction",
                    at=round(time.monotonic() - self._started_at, 3),
                    session_id=self.session_id,
                    player_id=player_id,
                )
                self.events.emit(event)


def _fields(event: Event) -> dict[str, Any]:
    return {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
