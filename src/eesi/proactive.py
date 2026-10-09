"""Characters that speak first.

Three ways, from most to least scripted:

- ``session.cue("The player drew a sword.")``: something happened; the
  character reacts as soon as nobody is talking.
- ``npc.when(lambda state: state.get("health", 100) < 20, "The player is badly hurt.")``:
  a trigger on live state. It fires once each time the condition becomes true
  (not on every update while it stays true), paced like any remark.
- ``Proactive(idle_after=20)``: after a quiet stretch the character asks
  itself whether something is worth saying (a question, a remark about what
  is happening, a follow-up on something remembered) and says it only if so.
  The decision is a short text-only request kept out of the conversation;
  nothing is said when the answer is no.

Remarks never talk over anyone: they wait for the floor, keep a cooldown, and
only the newest of each kind waits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Proactive", "Trigger", "IDLE_SKIP"]

#: What the decision answers when nothing is worth saying.
IDLE_SKIP = "SKIP"

_IDLE_PROMPT = (
    "Nobody has said anything for a while. Decide whether you, in character, would naturally speak up "
    "right now: to ask how they are doing, to remark on what is happening around you, to follow up on "
    "something you remember about them, or to offer help. Speak up only if it would feel natural and "
    "welcome, not to fill silence, and only with something new: never repeat, rephrase or apologise "
    "again for anything you already said. If you would, reply with exactly what you would say, in one "
    "or two short spoken sentences, in the language of the conversation. If not, reply with the single "
    f"word {IDLE_SKIP}."
)


@dataclass
class Proactive:
    """When a character may speak up on its own.

    ``idle_after``: seconds of quiet before it considers a remark (None turns
    idle remarks off). ``cooldown``: the least time between two considerations,
    whether or not it spoke; each is one short model call. ``max_remarks``: a
    cap on remarks per session. ``guidance``: what kind of remark you
    want ("point out quest hints", "check on their mood"), added to the
    character's own judgement.
    """

    idle_after: float | None = 25.0
    cooldown: float = 60.0
    max_remarks: int | None = None
    guidance: str | None = None

    def prompt(self) -> str:
        if self.guidance:
            return f"{_IDLE_PROMPT} What you might bring up: {self.guidance.strip()}"
        return _IDLE_PROMPT


Condition = Callable[[Mapping[str, Any]], bool]


@dataclass
class Trigger:
    """A remark the character makes when live state turns a condition true."""

    condition: Condition
    #: What happened, for the character to put in its own words; or a function of the state.
    what: str | Callable[[Mapping[str, Any]], str]
    kind: str = ""
    cooldown: float = 30.0
    urgent: bool = False
    #: Fire only the first time in a session.
    once: bool = False
    instructions: str | None = None
    _was_true: bool = field(default=False, repr=False)
    _fired: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = f"trigger:{id(self)}"

    def check(self, state: Mapping[str, Any]) -> str | None:
        """The remark to offer now, if the condition just became true."""
        try:
            now = bool(self.condition(state))
        except Exception:
            now = False
        rising = now and not self._was_true
        self._was_true = now
        if not rising or (self.once and self._fired):
            return None
        self._fired = True
        return self.what(state) if callable(self.what) else self.what

    def fresh(self) -> Trigger:
        """This trigger with no history, for a new session."""
        return Trigger(
            condition=self.condition,
            what=self.what,
            kind=self.kind,
            cooldown=self.cooldown,
            urgent=self.urgent,
            once=self.once,
            instructions=self.instructions,
        )
