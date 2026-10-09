"""What a memory is, and where memories are kept."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "ALL_PLAYERS",
    "KINDS",
    "MemoryRecord",
    "MemoryStore",
    "layer_of",
]

#: What a memory is about, and which layer it belongs to.
#:
#: - semantic: stable facts about the player (``fact``), how they like to be
#:   spoken to (``preference``), how they and the character stand
#:   (``relationship``), anything else worth keeping (``note``).
#: - episodic: things that happened (``event``), promises made (``promise``)
#:   and what each conversation was about (``summary``).
#: - world: knowledge that is not about one player (``lore``, and any record
#:   with no player), shared by every session of the character.
KINDS: dict[str, str] = {
    "fact": "semantic",
    "preference": "semantic",
    "relationship": "semantic",
    "note": "semantic",
    "event": "episodic",
    "promise": "episodic",
    "summary": "episodic",
    "lore": "world",
}

Kind = Literal[
    "fact", "preference", "relationship", "note", "event", "promise", "summary", "lore"
]
Source = Literal["app", "player", "extraction", "correction", "import"]


class _AllPlayers:
    def __repr__(self) -> str:
        return "ALL_PLAYERS"


#: Every player's records (and the character-wide ones), for inspection and erasure.
ALL_PLAYERS: Any = _AllPlayers()


def layer_of(kind: str, player_id: str | None) -> str:
    if player_id is None:
        return "world"
    return KINDS.get(kind, "semantic")


def new_id() -> str:
    return f"mem_{uuid.uuid4().hex[:20]}"


@dataclass
class MemoryRecord:
    """One thing a character remembers.

    ``player_id`` None means character-wide knowledge every player's session
    may use; anything about one person carries their id and is never shown
    to another player's session. ``key`` makes a record the current value of
    something (a preference named ``language``): writing the same key again
    replaces it. ``superseded_by`` marks a memory a correction replaced; it is
    kept for provenance and never recalled.
    """

    text: str
    kind: str = "fact"
    player_id: str | None = None
    character_id: str = ""
    namespace: str = "default"
    #: 0..1: how much this matters to the relationship.
    importance: float = 0.5
    #: 0..1: how sure we are. App writes are 1; extracted memories less.
    confidence: float = 1.0
    #: ``app`` (your code), ``player`` (they asked to be remembered), ``extraction``
    #: (learned from a conversation), ``correction`` or ``import``.
    source: str = "app"
    key: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=time.time)
    updated_at: float = 0.0
    expires_at: float | None = None
    last_recalled_at: float | None = None
    recall_count: int = 0
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = self.created_at
        self.importance = min(1.0, max(0.0, float(self.importance)))
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
        self.tags = tuple(self.tags)

    @property
    def layer(self) -> str:
        return layer_of(self.kind, self.player_id)

    def active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.superseded_by is None and (
            self.expires_at is None or self.expires_at > now
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["layer"] = self.layer
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        values = {k: v for k, v in data.items() if k in known}
        values["tags"] = tuple(values.get("tags") or ())
        return cls(**values)

    def evolve(self, **changes: Any) -> MemoryRecord:
        return replace(self, **changes)


@runtime_checkable
class MemoryStore(Protocol):
    """Where records live. Implement this to keep memory somewhere else.

    Every method is async, and the SDK never awaits one on the audio path.
    Scope arguments are explicit on every call: a store never infers which
    player a read is for.
    """

    async def put(self, record: MemoryRecord) -> None:
        """Insert *record*, or replace the record with its id."""

    async def get(self, record_id: str) -> MemoryRecord | None: ...

    async def delete(self, record_ids: Sequence[str]) -> int:
        """Remove records for good; returns how many there were."""

    async def query(
        self,
        *,
        namespace: str,
        character_id: str | None,
        player_ids: Sequence[str | None] | Any,
        kinds: Sequence[str] | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """Records in scope, newest first. ``player_ids=ALL_PLAYERS`` reads every player."""

    async def search(
        self,
        text: str,
        *,
        namespace: str,
        character_id: str | None,
        player_ids: Sequence[str | None],
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Active records matching *text*, each with a relevance in 0..1."""

    async def touch(self, record_ids: Sequence[str], at: float) -> None:
        """Note that these records were recalled."""

    async def erase(
        self,
        *,
        namespace: str,
        character_id: str | None = None,
        player_id: str | None | Any = ALL_PLAYERS,
    ) -> int:
        """Remove every record in scope for good."""

    async def close(self) -> None: ...
