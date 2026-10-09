"""Long-term memory for characters, local first.

::

    npc = nur.npc(name="Luna", persona="...", memory=nur.memory.local("./luna-memory"))

    await npc.remember("The northern bridge collapsed.")                 # every player's Luna knows
    await npc.remember("They betrayed the northern alliance.", player_id="p42")  # only p42's
    await npc.memory.for_player("p42").list()                             # inspect
    await npc.memory.erase(player_id="p42")                               # forget them completely

Layers: *working* memory is the live conversation, held by the runtime.
*Episodic* memories are what happened (events, promises, a summary of each
conversation). *Semantic* memories are what is true of the player (facts,
preferences, the relationship). *World* memories belong to no player and
every session of the character may use them. A player's memories are only
ever recalled into that player's sessions.

When a session starts, the most relevant memories (by what is happening,
importance and recency) are composed into a ``memory`` context block within
a byte budget. As the player talks, memories matching what they said are
searched off the audio path and the block is refreshed. When the session
ends, what was worth keeping is extracted (see :mod:`.extraction`).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .extraction import (
    PREFERENCE_KEYS,
    ExtractionRequest,
    ExtractionResult,
    Extractor,
    LLMExtractor,
    MemoryCandidate,
)
from .recall import (
    DEFAULT_BLOCK_BYTES,
    RecallWeights,
    compose_block,
    rank,
    relative_age,
)
from .records import ALL_PLAYERS, KINDS, MemoryRecord, MemoryStore, layer_of
from .stores import InMemoryStore, LocalMemoryStore, query_terms

log = logging.getLogger("eesi")

__all__ = [
    "ALL_PLAYERS",
    "CharacterMemory",
    "ExtractionRequest",
    "ExtractionResult",
    "Extractor",
    "InMemoryStore",
    "KINDS",
    "LLMExtractor",
    "LocalMemoryStore",
    "Memory",
    "MemoryCandidate",
    "MemoryPolicy",
    "MemoryRecord",
    "MemoryStore",
    "PREFERENCE_KEYS",
    "PlayerMemory",
    "RecallWeights",
    "layer_of",
    "migrate",
    "relative_age",
]

EXPORT_FORMAT = "eesi.memory"


@dataclass
class MemoryPolicy:
    """What a character may keep, and for how long.

    The defaults learn from every conversation and keep it until erased.
    Set ``extract=False`` to keep only what your code writes.
    """

    #: Learn memories from each finished conversation.
    extract: bool = True
    #: Episodic memories (events, promises, summaries) expire after this many days.
    retention_days: float | None = None
    #: The oldest, least important memories of a player are dropped past this count.
    max_records_per_player: int = 500
    #: Applied to every text before it is stored, e.g. to strip e-mail addresses.
    redact: Callable[[str], str] | None = None
    #: The ``memory`` context block's size in bytes.
    block_bytes: int = DEFAULT_BLOCK_BYTES
    #: Search memories for what the player just said, and refresh the block.
    recall_on_turn: bool = True
    weights: RecallWeights = field(default_factory=RecallWeights)
    #: Your own extraction model; the default asks nur-llm-v1 with your key.
    extractor: Extractor | None = None


class Memory:
    """A store plus a policy: what you pass as a character's ``memory=``."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: MemoryPolicy | None = None,
        namespace: str = "default",
    ) -> None:
        if not re.match(r"^[\w.:-]{1,128}$", namespace):
            raise ValueError(
                "namespace must be 1-128 letters, digits, '_', '.', ':' or '-'."
            )
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.namespace = namespace

    def __repr__(self) -> str:
        return f"Memory({self.store!r}, namespace={self.namespace!r})"

    async def close(self) -> None:
        await self.store.close()


def _similar(a: str, b: str) -> bool:
    left, right = set(query_terms(a, 64)), set(query_terms(b, 64))
    if not left or not right:
        return a.strip().lower() == b.strip().lower()
    return len(left & right) / len(left | right) >= 0.8


class CharacterMemory:
    """One character's memory: every player's, and its own world knowledge."""

    def __init__(
        self,
        memory: Memory,
        *,
        character_id: str,
        character_name: str,
        chat: Callable[[list[dict[str, Any]]], Awaitable[str]] | None = None,
    ) -> None:
        self.memory = memory
        self.store = memory.store
        self.policy = memory.policy
        self.namespace = memory.namespace
        self.character_id = character_id
        self.character_name = character_name
        self._chat = chat

    def __repr__(self) -> str:
        return f"CharacterMemory({self.character_id!r}, {self.memory!r})"

    def for_player(self, player_id: str) -> PlayerMemory:
        return PlayerMemory(self, player_id)

    # ── writing ──────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = " ".join(str(text).split())
        if self.policy.redact is not None:
            text = self.policy.redact(text)
        return text

    def _expiry(self, kind: str, created: float, ttl: float | None) -> float | None:
        if ttl is not None:
            return created + ttl
        if self.policy.retention_days is not None and KINDS.get(kind) == "episodic":
            return created + self.policy.retention_days * 86_400
        return None

    async def remember(
        self,
        text: str,
        *,
        player_id: str | None = None,
        kind: str = "fact",
        importance: float = 0.6,
        key: str | None = None,
        ttl: float | None = None,
        tags: Sequence[str] = (),
        source: str = "app",
        confidence: float = 1.0,
        session_id: str | None = None,
        replaces: str | None = None,
    ) -> MemoryRecord:
        """Keep *text*. With ``key``, it replaces the player's earlier record of that key."""
        cleaned = self._clean(text)
        if not cleaned:
            raise ValueError("A memory needs some text.")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}.")
        now = time.time()
        record = MemoryRecord(
            text=cleaned,
            kind=kind,
            player_id=player_id,
            character_id=self.character_id,
            namespace=self.namespace,
            importance=importance,
            confidence=confidence,
            source=source,
            key=key,
            session_id=session_id,
            tags=tuple(tags),
            created_at=now,
            expires_at=self._expiry(kind, now, ttl),
        )
        if key is not None:
            for old in await self._scope(player_id, kinds=[kind]):
                if old.key == key:
                    replaces = replaces or old.id
        await self.store.put(record)
        if replaces:
            old = await self.store.get(replaces)
            if (
                old is not None
                and old.character_id == self.character_id
                and old.player_id == player_id
            ):
                await self.store.put(
                    old.evolve(superseded_by=record.id, updated_at=now)
                )
        return record

    async def update(
        self,
        record_id: str,
        *,
        text: str | None = None,
        importance: float | None = None,
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        ttl: float | None = None,
    ) -> MemoryRecord | None:
        """Edit a record in place (your own correction of what is stored)."""
        record = await self._owned(record_id)
        if record is None:
            return None
        changes: dict[str, Any] = {"updated_at": time.time()}
        if text is not None:
            changes["text"] = self._clean(text)
        if importance is not None:
            changes["importance"] = min(1.0, max(0.0, importance))
        if kind is not None:
            if kind not in KINDS:
                raise ValueError(f"kind must be one of {', '.join(KINDS)}.")
            changes["kind"] = kind
        if tags is not None:
            changes["tags"] = tuple(tags)
        if ttl is not None:
            changes["expires_at"] = time.time() + ttl
        updated = record.evolve(**changes)
        await self.store.put(updated)
        return updated

    async def correct(
        self, record_id: str, text: str, *, source: str = "correction"
    ) -> MemoryRecord | None:
        """Replace a memory that was wrong, keeping the old one as history."""
        old = await self._owned(record_id)
        if old is None:
            return None
        return await self.remember(
            text,
            player_id=old.player_id,
            kind=old.kind,
            importance=old.importance,
            key=old.key,
            tags=old.tags,
            source=source,
            replaces=old.id,
        )

    async def forget(self, *record_ids: str) -> int:
        """Delete records for good."""
        owned = [
            record.id
            for record_id in record_ids
            if (record := await self._owned(record_id)) is not None
        ]
        return await self.store.delete(owned)

    async def erase(self, *, player_id: str | None | Any = ALL_PLAYERS) -> int:
        """Delete everything about one player (or every player and the character's world memory)."""
        return await self.store.erase(
            namespace=self.namespace,
            character_id=self.character_id,
            player_id=player_id,
        )

    # ── reading ──────────────────────────────────────────────────────────

    async def _owned(self, record_id: str) -> MemoryRecord | None:
        record = await self.store.get(record_id)
        if (
            record is None
            or record.character_id != self.character_id
            or record.namespace != self.namespace
        ):
            return None
        return record

    async def _scope(
        self, player_id: str | None, *, kinds: Sequence[str] | None = None
    ) -> list[MemoryRecord]:
        return await self.store.query(
            namespace=self.namespace,
            character_id=self.character_id,
            player_ids=[player_id],
            kinds=kinds,
        )

    async def get(self, record_id: str) -> MemoryRecord | None:
        return await self._owned(record_id)

    async def list(
        self,
        *,
        player_id: str | None | Any = ALL_PLAYERS,
        kinds: Sequence[str] | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """Records, newest first. ``player_id=None`` lists the character's world memory."""
        player_ids = player_id if player_id is ALL_PLAYERS else [player_id]
        return await self.store.query(
            namespace=self.namespace,
            character_id=self.character_id,
            player_ids=player_ids,
            kinds=kinds,
            include_inactive=include_inactive,
            limit=limit,
        )

    async def search(
        self, query: str, *, player_id: str | None = None, limit: int = 10
    ) -> list[tuple[MemoryRecord, float]]:
        """Records matching *query*: the player's and the character's world memory."""
        player_ids = [player_id, None] if player_id is not None else [None]
        return await self.store.search(
            query,
            namespace=self.namespace,
            character_id=self.character_id,
            player_ids=player_ids,
            limit=limit,
        )

    async def recall(
        self, *, player_id: str | None = None, query: str | None = None, limit: int = 12
    ) -> list[tuple[MemoryRecord, float]]:
        """What the character would bring to mind for *player_id*: ranked by relevance to *query*, importance and recency."""
        player_ids = [player_id, None] if player_id is not None else [None]
        candidates = await self.store.query(
            namespace=self.namespace,
            character_id=self.character_id,
            player_ids=player_ids,
            limit=400,
        )
        relevance: dict[str, float] = {}
        if query:
            for record, score in await self.store.search(
                query,
                namespace=self.namespace,
                character_id=self.character_id,
                player_ids=player_ids,
                limit=40,
            ):
                relevance[record.id] = score
        return rank(candidates, relevance, weights=self.policy.weights)[:limit]

    async def block(
        self,
        *,
        player_id: str | None,
        player_name: str | None = None,
        query: str | None = None,
        session_preferences: dict[str, str] | None = None,
    ) -> tuple[str, list[MemoryRecord]]:
        """The ``memory`` context block for a session, and the records in it."""
        player_ids = [player_id, None] if player_id is not None else [None]
        candidates = await self.store.query(
            namespace=self.namespace,
            character_id=self.character_id,
            player_ids=player_ids,
            limit=400,
        )
        relevance: dict[str, float] = {}
        if query:
            for record, score in await self.store.search(
                query,
                namespace=self.namespace,
                character_id=self.character_id,
                player_ids=player_ids,
                limit=40,
            ):
                relevance[record.id] = score
        ranked = rank(candidates, relevance, weights=self.policy.weights)
        text, used = compose_block(
            ranked,
            character_name=self.character_name,
            player_name=player_name,
            first_meeting=player_id is not None,
            budget_bytes=self.policy.block_bytes,
            session_preferences=session_preferences,
        )
        if used:
            await self.store.touch([record.id for record in used], time.time())
        return text, used

    # ── learning ─────────────────────────────────────────────────────────

    def _extractor(self) -> Extractor | None:
        if self.policy.extractor is not None:
            return self.policy.extractor
        if self._chat is not None:
            return LLMExtractor(self._chat)
        return None

    async def learn(
        self,
        transcript: Sequence[tuple[str, str]],
        *,
        player_id: str,
        player_name: str | None = None,
        session_id: str | None = None,
        skip_kinds: Sequence[str] = (),
    ) -> tuple[list[MemoryRecord], list[MemoryRecord], list[str]]:
        """Extract and keep what a conversation is worth remembering.

        *transcript* is ``(speaker, text)`` lines: ``"player"`` for the player,
        ``"event"`` for something that happened in the game, anything else for
        the character. Returns (added, superseded, forgotten ids). Use it for a
        conversation the SDK did not hold itself (a client that connected
        with a client secret and sent you its transcript). ``skip_kinds``
        drops candidates of those kinds (``("preference",)`` when the
        character does not personalize).
        """
        extractor = self._extractor()
        if extractor is None or not any(text.strip() for _, text in transcript):
            return [], [], []
        existing = await self._scope(player_id)
        existing = [record for record in existing if record.kind != "summary"][:60]
        result = await extractor(
            ExtractionRequest(
                character_name=self.character_name,
                player_id=player_id,
                player_name=player_name,
                transcript=list(transcript),
                existing=existing,
            )
        )
        if skip_kinds:
            result = ExtractionResult(
                summary=result.summary,
                memories=tuple(m for m in result.memories if m.kind not in skip_kinds),
                forget=result.forget,
                raw=result.raw,
            )
        return await self.apply(
            result, player_id=player_id, session_id=session_id, existing=existing
        )

    async def apply(
        self,
        result: ExtractionResult,
        *,
        player_id: str,
        session_id: str | None = None,
        existing: Sequence[MemoryRecord] | None = None,
    ) -> tuple[list[MemoryRecord], list[MemoryRecord], list[str]]:
        """Write an extraction's memories: dedupe, supersede corrections, upsert keyed preferences."""
        if existing is None:
            existing = await self._scope(player_id)
        added: list[MemoryRecord] = []
        superseded: list[MemoryRecord] = []
        forgotten = [
            record_id
            for record_id in result.forget
            if any(r.id == record_id for r in existing)
        ]
        if forgotten:
            await self.store.delete(forgotten)
        live = [record for record in existing if record.id not in forgotten]
        for candidate in result.memories:
            if (
                candidate.replaces is None
                and candidate.key is None
                and any(_similar(candidate.text, record.text) for record in live)
            ):
                continue
            replaces = (
                candidate.replaces
                if any(r.id == candidate.replaces for r in live)
                else None
            )
            record = await self.remember(
                candidate.text,
                player_id=player_id,
                kind=candidate.kind if candidate.kind in KINDS else "fact",
                importance=candidate.importance,
                key=candidate.key,
                source="extraction",
                confidence=0.8,
                session_id=session_id,
                replaces=replaces,
            )
            added.append(record)
            live.append(record)
            if replaces:
                old = next(r for r in existing if r.id == replaces)
                superseded.append(old)
        if result.summary:
            added.append(
                await self.remember(
                    result.summary,
                    player_id=player_id,
                    kind="summary",
                    importance=0.5,
                    source="extraction",
                    confidence=0.8,
                    session_id=session_id,
                )
            )
        await self._prune(player_id)
        return added, superseded, forgotten

    async def _prune(self, player_id: str) -> None:
        limit = self.policy.max_records_per_player
        records = await self.store.query(
            namespace=self.namespace,
            character_id=self.character_id,
            player_ids=[player_id],
            include_inactive=True,
        )
        if len(records) <= limit:
            return
        inactive = [record for record in records if not record.active()]
        active = rank(
            [record for record in records if record.active()],
            weights=self.policy.weights,
        )
        excess = len(records) - limit
        doomed = [record.id for record in inactive][:excess]
        if len(doomed) < excess:
            doomed += [record.id for record, _ in reversed(active)][
                : excess - len(doomed)
            ]
        await self.store.delete(doomed)

    # ── portability ──────────────────────────────────────────────────────

    async def export(
        self,
        path: str | Path | None = None,
        *,
        player_id: str | None | Any = ALL_PLAYERS,
    ) -> dict[str, Any]:
        """Every record in scope as JSON (written to *path* when given)."""
        records = await self.list(player_id=player_id, include_inactive=True)
        data = {
            "format": EXPORT_FORMAT,
            "version": 1,
            "exported_at": time.time(),
            "namespace": self.namespace,
            "character_id": self.character_id,
            "records": [record.to_dict() for record in records],
        }
        if path is not None:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return data

    async def import_records(
        self, data: dict[str, Any] | Sequence[dict[str, Any]] | str | Path
    ) -> int:
        """Load records exported by :meth:`export` (or a list of record dicts) into this character."""
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        items = data.get("records", []) if isinstance(data, dict) else list(data)
        count = 0
        for item in items:
            record = MemoryRecord.from_dict(item)
            await self.store.put(
                record.evolve(character_id=self.character_id, namespace=self.namespace)
            )
            count += 1
        return count


class PlayerMemory:
    """One player's memories with one character: inspect, edit, delete."""

    def __init__(self, memory: CharacterMemory, player_id: str) -> None:
        self.character = memory
        self.player_id = player_id

    def __repr__(self) -> str:
        return f"PlayerMemory({self.character.character_id!r}, {self.player_id!r})"

    async def remember(self, text: str, **options: Any) -> MemoryRecord:
        return await self.character.remember(text, player_id=self.player_id, **options)

    async def list(
        self, *, kinds: Sequence[str] | None = None, include_inactive: bool = False
    ) -> list[MemoryRecord]:
        return await self.character.list(
            player_id=self.player_id, kinds=kinds, include_inactive=include_inactive
        )

    async def search(
        self, query: str, *, limit: int = 10
    ) -> list[tuple[MemoryRecord, float]]:
        return await self.character.search(query, player_id=self.player_id, limit=limit)

    async def recall(
        self, query: str | None = None, *, limit: int = 12
    ) -> list[tuple[MemoryRecord, float]]:
        return await self.character.recall(
            player_id=self.player_id, query=query, limit=limit
        )

    async def preferences(self) -> dict[str, str]:
        return {
            record.key or record.id: record.text
            for record in await self.list(kinds=["preference"])
        }

    async def update(self, record_id: str, **changes: Any) -> MemoryRecord | None:
        record = await self.character.get(record_id)
        if record is None or record.player_id != self.player_id:
            return None
        return await self.character.update(record_id, **changes)

    async def correct(self, record_id: str, text: str) -> MemoryRecord | None:
        record = await self.character.get(record_id)
        if record is None or record.player_id != self.player_id:
            return None
        return await self.character.correct(record_id, text)

    async def forget(self, *record_ids: str) -> int:
        mine = []
        for record_id in record_ids:
            record = await self.character.get(record_id)
            if record is not None and record.player_id == self.player_id:
                mine.append(record_id)
        return await self.character.forget(*mine)

    async def erase(self) -> int:
        return await self.character.erase(player_id=self.player_id)

    async def export(self, path: str | Path | None = None) -> dict[str, Any]:
        return await self.character.export(path, player_id=self.player_id)


async def migrate(
    source: MemoryStore, target: MemoryStore, *, namespace: str = "default"
) -> int:
    """Copy every record of *namespace* from one store to another (ids kept)."""
    records = await source.query(
        namespace=namespace,
        character_id=None,
        player_ids=ALL_PLAYERS,
        include_inactive=True,
    )
    for record in records:
        await target.put(record)
    return len(records)
