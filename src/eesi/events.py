"""What happens in a session, as typed events.

Every event has a ``type`` string (``"transcript.final"``, ``"tool.completed"``)
and ``at``, the seconds since the session started. Subscribe with a callback
or iterate::

    @session.on("transcript.final")
    def heard(event):
        print(event.player_id, event.text)

    async for event in session.events():
        ...

Callbacks may be plain functions or coroutines. A plain callback runs inline
and must be quick; a coroutine runs as its own task, so a slow handler never
holds up the audio. An exception in a handler is logged and the session goes
on. ``on("tool.*")`` matches every tool event, ``on("*")`` every event.

Audio chunks (``audio.output.chunk``) reach callbacks registered for them by
name, and reach :meth:`EventBus.stream` only with ``include_audio=True``:
they arrive fifty times a second and would crowd everything else out.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .memory import MemoryRecord

log = logging.getLogger("eesi")

__all__ = [
    "AudioInputEnded",
    "AudioInputStarted",
    "AudioOutputChunk",
    "ContextUpdated",
    "Event",
    "EventBus",
    "ErrorEvent",
    "Feedback",
    "Hangup",
    "MemoryRetrieved",
    "MemoryUpdated",
    "Reconnected",
    "Reconnecting",
    "ResponseCancelled",
    "ResponseCompleted",
    "ResponseStarted",
    "ResponseTextDelta",
    "SessionEnded",
    "SessionStarted",
    "SpeakerTurn",
    "SpeechInterrupted",
    "ToolCompleted",
    "ToolFailed",
    "ToolStarted",
    "TranscriptFinal",
    "TranscriptPartial",
]


@dataclass(frozen=True, kw_only=True)
class Event:
    type: ClassVar[str] = "event"
    #: Seconds since the session started.
    at: float = 0.0
    session_id: str | None = None
    player_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class SessionStarted(Event):
    type: ClassVar[str] = "session.started"
    model: str | None = None
    #: Whether this session continues an earlier one (a resumed conversation).
    resumed: bool = False
    #: Whether the organization records this session. Show ``disclosure`` to people when it does.
    recorded: bool = False
    disclosure: str | None = None


@dataclass(frozen=True, kw_only=True)
class SessionEnded(Event):
    type: ClassVar[str] = "session.ended"
    #: ``closed`` (you closed it), ``hangup`` (the character said goodbye), ``limit``
    #: (time or credits), ``connection_lost`` or ``server_closed``.
    reason: str = "closed"
    code: int | None = None
    message: str | None = None


@dataclass(frozen=True, kw_only=True)
class AudioInputStarted(Event):
    """The player started talking."""

    type: ClassVar[str] = "audio.input.started"
    #: The character was speaking at the time.
    while_speaking: bool = False
    #: Where the speech starts in the audio sent so far, in ms, when the server says.
    audio_start_ms: int | None = None


@dataclass(frozen=True, kw_only=True)
class AudioInputEnded(Event):
    type: ClassVar[str] = "audio.input.ended"
    #: Where the speech ends in the audio sent so far, in ms, when the server says.
    audio_end_ms: int | None = None


@dataclass(frozen=True)
class SpeakerTurn:
    """One person's part of a transcript several voices share."""

    text: str
    #: ``Speaker N`` as the transcript tagged it; None when there was one voice.
    speaker: int | None = None
    #: The player the SDK attributes it to, when it knows.
    player_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptPartial(Event):
    """What the player is saying, so far. Each one replaces the last."""

    type: ClassVar[str] = "transcript.partial"
    text: str = ""
    item_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptFinal(Event):
    """A finished player turn."""

    type: ClassVar[str] = "transcript.final"
    text: str = ""
    item_id: str | None = None
    turns: tuple[SpeakerTurn, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ResponseStarted(Event):
    type: ClassVar[str] = "response.started"
    response_id: str = ""


@dataclass(frozen=True, kw_only=True)
class ResponseTextDelta(Event):
    """The character's words as they are generated, for subtitles."""

    type: ClassVar[str] = "response.text.delta"
    response_id: str = ""
    text: str = ""


@dataclass(frozen=True, kw_only=True)
class ResponseCompleted(Event):
    """A reply finished generating. Its audio may still be playing."""

    type: ClassVar[str] = "response.completed"
    response_id: str = ""
    text: str = ""
    #: Seconds from the end of the player's turn to the first audio of this reply, when measured.
    latency: float | None = None


@dataclass(frozen=True, kw_only=True)
class ResponseCancelled(Event):
    type: ClassVar[str] = "response.cancelled"
    response_id: str = ""
    #: ``turn_detected`` (the player spoke over it), ``client_cancelled`` or the server's reason.
    reason: str | None = None
    #: What was generated before the cut; the player may have heard less.
    text: str = ""


@dataclass(frozen=True, kw_only=True)
class AudioOutputChunk(Event):
    """PCM16 mono of the character's voice."""

    type: ClassVar[str] = "audio.output.chunk"
    response_id: str = ""
    pcm: bytes = field(default=b"", repr=False)
    sample_rate: int = 24_000


@dataclass(frozen=True, kw_only=True)
class SpeechInterrupted(Event):
    """The player cut the character off; its queued audio was dropped."""

    type: ClassVar[str] = "speech.interrupted"
    response_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class MemoryRetrieved(Event):
    type: ClassVar[str] = "memory.retrieved"
    query: str | None = None
    records: tuple[MemoryRecord, ...] = ()


@dataclass(frozen=True, kw_only=True)
class MemoryUpdated(Event):
    type: ClassVar[str] = "memory.updated"
    added: tuple[MemoryRecord, ...] = ()
    updated: tuple[MemoryRecord, ...] = ()
    removed: tuple[str, ...] = ()
    #: ``app``, ``player``, ``extraction`` or ``correction``.
    source: str = "app"


@dataclass(frozen=True, kw_only=True)
class ContextUpdated(Event):
    type: ClassVar[str] = "context.updated"
    block: str = ""
    #: The block's text as the model now reads it; empty when the block was removed.
    text: str = ""


@dataclass(frozen=True, kw_only=True)
class ToolStarted(Event):
    type: ClassVar[str] = "tool.started"
    call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ToolCompleted(Event):
    type: ClassVar[str] = "tool.completed"
    call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    #: What the function returned.
    result: Any = None
    #: What the character was told.
    output: str = ""
    duration: float = 0.0


@dataclass(frozen=True, kw_only=True)
class ToolFailed(Event):
    type: ClassVar[str] = "tool.failed"
    call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    #: ``exception``, ``invalid_arguments``, ``unknown_tool``, ``not_allowed``,
    #: ``not_confirmed``, ``timeout`` or ``duplicate``.
    reason: str = "exception"
    error: str = ""


@dataclass(frozen=True, kw_only=True)
class ErrorEvent(Event):
    type: ClassVar[str] = "error"
    message: str = ""
    code: str | None = None


@dataclass(frozen=True, kw_only=True)
class Reconnecting(Event):
    type: ClassVar[str] = "reconnecting"
    attempt: int = 0
    delay: float = 0.0
    code: int | None = None


@dataclass(frozen=True, kw_only=True)
class Reconnected(Event):
    type: ClassVar[str] = "reconnected"
    #: Whether the conversation was resumed; False means a fresh one began.
    resumed: bool = True


@dataclass(frozen=True, kw_only=True)
class Feedback(Event):
    """A rating your application recorded with :meth:`Session.feedback`."""

    type: ClassVar[str] = "feedback"
    rating: Any = None
    note: str | None = None
    response_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class Hangup(Event):
    """The character ended the conversation (said goodbye). The session closes once its audio has played."""

    type: ClassVar[str] = "session.hangup"
    reason: str = ""


Handler = Callable[[Event], Any]


def _matches(pattern: str, kind: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        return kind.startswith(pattern[:-1])
    return pattern == kind


class EventBus:
    """Fan-out of events to callbacks and async iterators."""

    def __init__(self, parent: EventBus | None = None) -> None:
        self._handlers: list[tuple[str, Handler]] = []
        self._streams: list[tuple[asyncio.Queue[Event | None], bool]] = []
        self._parent = parent
        self._tasks: set[asyncio.Task[Any]] = set()

    def on(self, kind: str | Handler = "*", handler: Handler | None = None) -> Any:
        """Register *handler* for events of *kind*; usable as a decorator."""
        if callable(kind):
            self._handlers.append(("*", kind))
            return kind
        if handler is not None:
            self._handlers.append((kind, handler))
            return handler

        def register(fn: Handler) -> Handler:
            self._handlers.append((kind, fn))
            return fn

        return register

    def off(self, handler: Handler) -> None:
        self._handlers = [
            (kind, fn) for kind, fn in self._handlers if fn is not handler
        ]

    def emit(self, event: Event) -> None:
        """Deliver *event* now. Never raises; never waits on a handler."""
        for pattern, handler in list(self._handlers):
            if pattern == "*" and event.type == AudioOutputChunk.type:
                continue
            if not _matches(pattern, event.type):
                continue
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    self._spawn(result, handler)
            except Exception:
                log.exception("event handler %r failed on %s", handler, event.type)
        for queue, include_audio in list(self._streams):
            if event.type == AudioOutputChunk.type and not include_audio:
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning(
                    "event stream is full; dropped %s (read events faster)", event.type
                )
        if self._parent is not None:
            self._parent.emit(event)

    async def stream(
        self, *, include_audio: bool = False, maxsize: int = 1000
    ) -> AsyncIterator[Event]:
        """Events from now until :meth:`close`."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=maxsize)
        entry = (queue, include_audio)
        self._streams.append(entry)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            if entry in self._streams:
                self._streams.remove(entry)

    def close(self) -> None:
        """End every :meth:`stream`."""
        for queue, _ in list(self._streams):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                # Make room for the end marker: a full queue's reader is far behind anyway.
                try:
                    queue.get_nowait()
                    queue.put_nowait(None)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for coroutine handlers that are still running."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)

    def _spawn(self, awaitable: Awaitable[Any], handler: Handler) -> None:
        async def run() -> None:
            try:
                await awaitable
            except Exception:
                log.exception("event handler %r failed", handler)

        task = asyncio.ensure_future(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
