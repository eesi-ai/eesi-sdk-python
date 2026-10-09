"""Live context: what is true right now, beside the conversation.

A game knows things nobody said out loud: where the player stands, what
they carry, that the gate just closed. Send them as context and the
character reads them on its next reply::

    await npc.context.update({"time_of_day": "dusk", "weather": "rain"})       # every player's session
    await session.context.update({"location": "Northern Gate", "health": "6/20"})  # this session only

Context is kept in named *blocks*. Each block is sent to the gateway as one
``eesi.context.update`` and placed after the character's instructions. A
block never enters the conversation history and never starts a reply by
itself; when it disagrees with something said earlier, it wins. Application
state stays authoritative: nothing the model says changes a block.

Updates merge field by field (``None`` removes a field). Fields may expire
(``ttl=`` seconds). A block is sent at most once per ``min_interval`` and only
when its text changed, so a game can update every frame without flooding
anything; the latest value always goes out.

Order is precedence and cache: blocks keep the order they were first sent in,
and a change re-sends only the blocks after it in the prompt. So the SDK sends
slow-changing blocks first: ``memory``, then the character's shared ``world``,
then mood and party, then this session's ``scene`` and your own blocks, and
last ``recall``, the memories the latest turn brought to mind.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ._protocol import (
    CONTEXT_KEY,
    MAX_CONTEXT_BLOCK_BYTES,
    MAX_CONTEXT_KEYS,
    MAX_CONTEXT_TOTAL_BYTES,
    byte_len,
)
from .errors import ContextError

__all__ = ["LiveContext", "render_fields"]

#: Blocks the SDK writes itself, in the order they are first sent.
SDK_BLOCK_ORDER = ("memory", "world", "mood", "party", "roster", "scene")
#: Blocks that change every turn go after everything else, so a change never
#: invalidates the cached prompt in front of them.
SDK_LAST_BLOCKS = ("recall",)


@dataclass
class _Field:
    value: Any
    expires_at: float | None = None


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _render_value(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(item) for item in value]
        if all(isinstance(item, (str, int, float, bool)) for item in items):
            return ", ".join(_render_value(item) for item in items) or "none"
        return json.dumps(
            items, ensure_ascii=False, default=str, separators=(", ", ": ")
        )
    if isinstance(value, Mapping):
        return json.dumps(
            value, ensure_ascii=False, default=str, separators=(", ", ": ")
        )
    return str(value)


def render_fields(fields: Mapping[str, Any]) -> str:
    """``key: value`` lines, in insertion order; nested mappings become ``key.sub: value``."""
    lines: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        value = _plain(value)
        if isinstance(value, Mapping) and value:
            for key, inner in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), inner)
            return
        lines.append(f"{prefix}: {_render_value(value)}")

    for key, value in fields.items():
        walk(str(key), value)
    return "\n".join(lines)


class _Block:
    def __init__(self, name: str) -> None:
        self.name = name
        self.fields: dict[str, _Field] = {}
        self.text: str | None = None  # a raw block replaces fields

    def render(self, now: float) -> str:
        if self.text is not None:
            return self.text
        live = {
            key: field.value
            for key, field in self.fields.items()
            if field.expires_at is None or field.expires_at > now
        }
        return render_fields(live)

    def next_expiry(self, now: float) -> float | None:
        times = [
            f.expires_at
            for f in self.fields.values()
            if f.expires_at is not None and f.expires_at > now
        ]
        return min(times) if times else None

    def prune(self, now: float) -> None:
        self.fields = {
            k: f
            for k, f in self.fields.items()
            if f.expires_at is None or f.expires_at > now
        }


Sink = Callable[[str, str], Awaitable[None]]


class LiveContext:
    """Named blocks of live state.

    A character's context is shared by all of its sessions; a session's
    context is its own. Both feed the same blocks list a session sends.
    """

    def __init__(
        self,
        default_block: str,
        *,
        min_interval: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.default_block = default_block
        self.min_interval = min_interval
        self._clock = clock
        self._blocks: dict[str, _Block] = {}
        self._listeners: list[Callable[[str], None]] = []
        self._expiry: asyncio.TimerHandle | None = None

    # ── reading ──────────────────────────────────────────────────────────

    def get(self, block: str | None = None) -> dict[str, Any]:
        """The live fields of *block* (expired ones are gone)."""
        found = self._blocks.get(block or self.default_block)
        if found is None:
            return {}
        now = self._clock()
        return {
            k: f.value
            for k, f in found.fields.items()
            if f.expires_at is None or f.expires_at > now
        }

    def blocks(self) -> list[str]:
        return list(self._blocks)

    def all_fields(self) -> dict[str, Any]:
        """Every live field of every block, later blocks winning a shared name."""
        merged: dict[str, Any] = {}
        for name in self._blocks:
            merged.update(self.get(name))
        return merged

    def render(self, block: str) -> str:
        found = self._blocks.get(block)
        return found.render(self._clock()) if found else ""

    def snapshot(self) -> dict[str, str]:
        """Every block's text exactly as the model reads it."""
        now = self._clock()
        return {
            name: block.render(now)
            for name, block in self._blocks.items()
            if block.render(now)
        }

    # ── writing ──────────────────────────────────────────────────────────

    def seed(self, values: Mapping[str, Any], *, block: str | None = None) -> None:
        """Set starting fields without notifying anyone (before any session exists)."""
        name = self._check_name(block or self.default_block)
        target = self._blocks.setdefault(name, _Block(name))
        for key, value in dict(_plain(values)).items():
            if value is not None:
                target.fields[str(key)] = _Field(value)
        self._check_size(name)

    async def update(
        self,
        values: Mapping[str, Any] | BaseModel | Any | None = None,
        /,
        *,
        block: str | None = None,
        ttl: float | None = None,
        **fields: Any,
    ) -> None:
        """Merge fields into a block. ``None`` removes a field; ``ttl`` expires them."""
        name = self._check_name(block or self.default_block)
        merged: dict[str, Any] = {}
        if values is not None:
            plain = _plain(values)
            if not isinstance(plain, Mapping):
                raise ContextError(
                    "context.update takes a mapping, a pydantic model or a dataclass."
                )
            merged.update(plain)
        merged.update(fields)
        target = self._blocks.setdefault(name, _Block(name))
        target.text = None
        expires = self._clock() + ttl if ttl is not None else None
        for key, value in merged.items():
            if value is None:
                target.fields.pop(str(key), None)
            else:
                target.fields[str(key)] = _Field(value, expires)
        self._check_size(name)
        self._changed(name)

    async def set(
        self, block: str, content: str | Mapping[str, Any] | BaseModel | None
    ) -> None:
        """Replace a whole block: with text as written, with fields, or remove it (None or "")."""
        name = self._check_name(block)
        if content is None or content == "":
            if self._blocks.pop(name, None) is not None:
                self._changed(name)
            return
        target = _Block(name)
        if isinstance(content, str):
            target.text = content.strip()
        else:
            plain = _plain(content)
            if not isinstance(plain, Mapping):
                raise ContextError(
                    "context.set takes text, a mapping, a pydantic model or a dataclass."
                )
            target.fields = {
                str(k): _Field(v) for k, v in plain.items() if v is not None
            }
        previous = self._blocks.get(name)
        self._blocks[name] = target
        try:
            self._check_size(name)
        except ContextError:
            if previous is None:
                self._blocks.pop(name, None)
            else:
                self._blocks[name] = previous
            raise
        self._changed(name)

    async def remove(self, *fields: str, block: str | None = None) -> None:
        name = block or self.default_block
        target = self._blocks.get(name)
        if target is None:
            return
        for key in fields:
            target.fields.pop(key, None)
        self._changed(name)

    async def clear(self, block: str | None = None) -> None:
        """Remove one block, or every block when *block* is None."""
        names = [block] if block else list(self._blocks)
        for name in names:
            if self._blocks.pop(name, None) is not None:
                self._changed(name)

    # ── plumbing ─────────────────────────────────────────────────────────

    def subscribe(self, listener: Callable[[str], None]) -> Callable[[], None]:
        """Call *listener(block)* whenever a block changes; returns an unsubscribe function."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _changed(self, name: str) -> None:
        self._schedule_expiry()
        for listener in list(self._listeners):
            listener(name)

    def _schedule_expiry(self) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
        now = self._clock()
        soonest = [
            t for b in self._blocks.values() if (t := b.next_expiry(now)) is not None
        ]
        if not soonest:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = max(0.0, min(soonest) - now) + 0.01

        def expire() -> None:
            self._expiry = None
            at = self._clock()
            for block in self._blocks.values():
                before = len(block.fields)
                block.prune(at)
                if len(block.fields) != before:
                    for listener in list(self._listeners):
                        listener(block.name)
            self._schedule_expiry()

        self._expiry = loop.call_later(delay, expire)

    def _check_name(self, name: str) -> str:
        if not CONTEXT_KEY.match(name):
            raise ContextError(
                f"Context block name {name!r} must be 1-48 lowercase letters, digits, '_', '.' or '-'."
            )
        return name

    def _check_size(self, name: str) -> None:
        text = self.render(name)
        size = byte_len(text)
        if size > MAX_CONTEXT_BLOCK_BYTES:
            raise ContextError(
                f"Context block {name!r} would be {size:,} bytes; one block holds at most "
                f"{MAX_CONTEXT_BLOCK_BYTES:,}. Keep live state a digest, not a save file."
            )

    def close(self) -> None:
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
        self._listeners.clear()


class ContextPublisher:
    """One session's view of every context source, sent as it changes.

    Sources are LiveContext objects (the character's and the session's) plus
    SDK blocks (memory, party). The publisher throttles each block, sends
    only changed text, keeps blocks in a stable order and re-sends everything
    after a resume, when the gateway has forgotten them.
    """

    def __init__(
        self,
        send: Sink,
        sources: list[LiveContext],
        *,
        min_interval: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
        on_sent: Callable[[str, str], None] | None = None,
    ) -> None:
        self._send = send
        self._sources = sources
        self._extra: dict[str, str] = {}
        self._sent: dict[str, str] = {}
        self._last_at: dict[str, float] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._min_interval = min_interval
        self._clock = clock
        self._on_sent = on_sent
        self._ready = False
        self._refused = False
        self._unsubscribe = [source.subscribe(self._changed) for source in sources]
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task[Any]] = set()

    def set_block(self, name: str, text: str) -> None:
        """An SDK block (memory, party, mood); empty text removes it."""
        if text:
            self._extra[name] = text
        else:
            self._extra.pop(name, None)
        self._changed(name)

    def current(self) -> dict[str, str]:
        """What the model holds now, by block."""
        return dict(self._sent)

    def desired(self) -> dict[str, str]:
        """Every block's latest text, in send order."""
        texts: dict[str, str] = {}
        for source in self._sources:
            for name, text in source.snapshot().items():
                # A session's own block wins over the character's block of the same name.
                texts[name] = text
        texts.update(self._extra)
        return {name: texts[name] for name in self._order(texts)}

    def _order(self, texts: Mapping[str, str]) -> list[str]:
        known = [name for name in SDK_BLOCK_ORDER if name in texts]
        last = [name for name in SDK_LAST_BLOCKS if name in texts]
        sent_first = [
            name
            for name in self._sent
            if name in texts and name not in known and name not in last
        ]
        rest = [
            name
            for name in texts
            if name not in known and name not in sent_first and name not in last
        ]
        return known + sent_first + rest + last

    def validate(self, texts: Mapping[str, str]) -> None:
        if len(texts) > MAX_CONTEXT_KEYS:
            raise ContextError(
                f"A session holds at most {MAX_CONTEXT_KEYS} context blocks; this one would have "
                f"{len(texts)} ({', '.join(texts)}). Group related fields into one block."
            )
        total = sum(byte_len(text) for text in texts.values())
        if total > MAX_CONTEXT_TOTAL_BYTES:
            raise ContextError(
                f"Context blocks together would be {total:,} bytes; a session holds at most "
                f"{MAX_CONTEXT_TOTAL_BYTES:,}. Trim the largest blocks."
            )

    async def start(self) -> None:
        """Send everything (a new or resumed session)."""
        self._ready = True
        self._sent.clear()
        await self.flush()

    def stop(self) -> None:
        self._ready = False
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()

    def refused(self) -> None:
        """The gateway does not take context at all; keep blocks, stop sending."""
        self._refused = True

    async def flush(self, only: str | None = None) -> None:
        """Send every changed block now, ignoring the throttle."""
        if not self._ready or self._refused:
            return
        async with self._lock:
            desired = self.desired()
            self.validate(desired)
            names = (
                [only]
                if only
                else list(desired) + [n for n in self._sent if n not in desired]
            )
            for name in names:
                text = desired.get(name, "")
                if self._sent.get(name, "") == text:
                    continue
                await self._send(name, text)
                self._last_at[name] = self._clock()
                if text:
                    self._sent[name] = text
                else:
                    self._sent.pop(name, None)
                if self._on_sent:
                    self._on_sent(name, text)

    def _changed(self, name: str) -> None:
        if not self._ready or self._refused or name in self._timers:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        since = self._clock() - self._last_at.get(name, float("-inf"))
        delay = max(0.0, self._min_interval - since)

        def fire() -> None:
            self._timers.pop(name, None)
            task = loop.create_task(self._flush_one(name))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

        self._timers[name] = loop.call_later(delay, fire)

    async def _flush_one(self, name: str) -> None:
        try:
            await self.flush(only=name)
        except ContextError:
            # The block that made the set too large stays unsent; the caller
            # saw the same error from validate() when it made the change.
            pass

    async def close(self) -> None:
        self.stop()
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        for task in list(self._pending):
            task.cancel()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
