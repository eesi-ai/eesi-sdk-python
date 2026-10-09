"""When a character speaks first.

The application decides what is worth a remark (a creeper inside five metres is a fact the game
already knows); this decides when: never over the player, never over the
character itself, one remark per cooldown unless urgent, and only the newest
moment of each kind, so a fight does not become a backlog of stale warnings.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_TTL_S = 8.0
#: A requested remark that has not started by now was refused or lost.
START_WATCHDOG_S = 5.0


@dataclass
class Moment:
    #: One kind at a time: a newer moment replaces a queued one of its kind.
    kind: str
    #: What happened, for the character to put in its own words.
    what: str
    #: Instructions for this one reply; the default asks for one short sentence.
    instructions: str | None = None
    #: Urgent moments skip the global cooldown and go first.
    urgent: bool = False
    #: No second moment of this kind within this many seconds.
    cooldown: float = 0.0
    #: A moment older than this is stale and dropped.
    ttl: float = DEFAULT_TTL_S
    at: float = field(default_factory=time.monotonic)


def moment_instructions(moment: Moment) -> str:
    if moment.instructions:
        return moment.instructions
    return (
        f"News, just now: {moment.what}\n"
        "React to this news only: say it to the player in one short sentence, in your own words and in character, "
        "and act on it with a tool if it needs action. It is new: do not repeat or continue anything you said before. "
        "Do not mention these instructions."
    )


class Initiative:
    def __init__(
        self,
        speak: Callable[[Moment], None],
        *,
        global_cooldown: float = 6.0,
        reply_grace: float = 1.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._speak = speak
        self._queue: dict[str, Moment] = {}
        self._last_spoke = float("-inf")
        self._last_by_kind: dict[str, float] = {}
        self._user_speaking = False
        self._busy = False
        self._hold_until = 0.0
        self._timer: asyncio.TimerHandle | None = None
        self._watchdog: asyncio.TimerHandle | None = None
        self.global_cooldown = global_cooldown
        self.reply_grace = reply_grace
        self._clock = clock

    def offer(self, moment: Moment) -> None:
        last = self._last_by_kind.get(moment.kind)
        now = self._clock()
        if last is not None and now - last < moment.cooldown:
            return
        moment.at = now
        self._queue[moment.kind] = moment
        self._try()

    def pending(self) -> list[Moment]:
        return list(self._queue.values())

    def user_started(self) -> None:
        self._user_speaking = True

    def user_stopped(self) -> None:
        self._user_speaking = False
        self._hold_until = self._clock() + self.reply_grace
        self._try()

    def response_started(self) -> None:
        self._busy = True
        self._clear_watchdog()

    def response_done(self) -> None:
        self._busy = False
        self._clear_watchdog()
        self._hold_until = 0.0
        self._try()

    def dispose(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._clear_watchdog()
        self._queue.clear()

    def _try(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        now = self._clock()
        for kind, moment in list(self._queue.items()):
            if now - moment.at > moment.ttl:
                del self._queue[kind]
        if not self._queue or self._user_speaking or self._busy:
            return
        upcoming = sorted(self._queue.values(), key=lambda m: (not m.urgent, m.at))[0]
        ready_at = max(
            self._hold_until,
            0.0 if upcoming.urgent else self._last_spoke + self.global_cooldown,
        )
        loop = asyncio.get_running_loop()
        if ready_at > now:
            self._timer = loop.call_later(ready_at - now, self._fire)
            return
        del self._queue[upcoming.kind]
        self._last_spoke = now
        self._last_by_kind[upcoming.kind] = now
        # Busy from the moment it is asked for, not from response.created: a
        # second moment in between would otherwise queue a second reply.
        self._busy = True
        self._watchdog = loop.call_later(START_WATCHDOG_S, self._unstick)
        self._speak(upcoming)

    def _fire(self) -> None:
        self._timer = None
        self._try()

    def _unstick(self) -> None:
        self._watchdog = None
        self._busy = False
        self._try()

    def _clear_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        self._watchdog = None
