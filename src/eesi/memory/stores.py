"""Memory stores that need nothing hosted: SQLite on disk, and a dict for tests.

:class:`LocalMemoryStore` is one SQLite file. It keeps full-text search in an
FTS5 index when the interpreter's SQLite has it (CPython's does on every
supported platform) and scores words in Python otherwise. Deleted rows are
overwritten on disk (``secure_delete``), so an erased player is gone from the
file, not only hidden. Every call runs in a worker thread; the event loop
never waits on the disk.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import MemoryStoreError
from .records import ALL_PLAYERS, MemoryRecord

__all__ = ["InMemoryStore", "LocalMemoryStore", "query_terms"]

_SCHEMA_VERSION = 1

_STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from had has have he her him his how i if in
    into is it its just me my no not of on or our she so than that the their them then there these they this
    to too us was we were what when where which who why will with would you your yes ok okay oh um uh""".split()
)

_COLUMNS = (
    "id",
    "namespace",
    "character_id",
    "player_id",
    "kind",
    "key",
    "text",
    "importance",
    "confidence",
    "source",
    "session_id",
    "tags",
    "created_at",
    "updated_at",
    "expires_at",
    "last_recalled_at",
    "recall_count",
    "superseded_by",
)


def query_terms(text: str, limit: int = 24) -> list[str]:
    """The words of *text* worth searching for."""
    seen: list[str] = []
    for word in re.findall(r"\w{2,}", text.lower()):
        if word in _STOPWORDS or word.isdigit() and len(word) < 3 or word in seen:
            continue
        seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        namespace=row["namespace"],
        character_id=row["character_id"],
        player_id=row["player_id"],
        kind=row["kind"],
        key=row["key"],
        text=row["text"],
        importance=row["importance"],
        confidence=row["confidence"],
        source=row["source"],
        session_id=row["session_id"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        last_recalled_at=row["last_recalled_at"],
        recall_count=row["recall_count"],
        superseded_by=row["superseded_by"],
    )


def _record_values(record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.id,
        record.namespace,
        record.character_id,
        record.player_id,
        record.kind,
        record.key,
        record.text,
        record.importance,
        record.confidence,
        record.source,
        record.session_id,
        json.dumps(list(record.tags)),
        record.created_at,
        record.updated_at,
        record.expires_at,
        record.last_recalled_at,
        record.recall_count,
        record.superseded_by,
    )


def _player_clause(player_ids: Sequence[str | None] | Any, params: list[Any]) -> str:
    if player_ids is ALL_PLAYERS:
        return "1"
    ids = list(player_ids)
    named = [pid for pid in ids if pid is not None]
    parts: list[str] = []
    if named:
        parts.append(f"m.player_id IN ({','.join('?' * len(named))})")
        params.extend(named)
    if None in ids:
        parts.append("m.player_id IS NULL")
    return f"({' OR '.join(parts)})" if parts else "0"


class LocalMemoryStore:
    """Memories in one SQLite file. Safe to share between characters and sessions in a process."""

    def __init__(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        if target.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
            target = target / "memory.db"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MemoryStoreError(
                f"Cannot create the memory directory {target.parent}: {error}"
            ) from error
        self.path = target
        self._lock = threading.Lock()
        try:
            self._db = sqlite3.connect(
                str(target), check_same_thread=False, isolation_level=None, timeout=5.0
            )
        except sqlite3.Error as error:
            raise MemoryStoreError(
                f"Cannot open memory database {target}: {error}"
            ) from error
        self._db.row_factory = sqlite3.Row
        self.fts = False
        self._migrate()

    def __repr__(self) -> str:
        return f"LocalMemoryStore({str(self.path)!r})"

    # ── schema ───────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        db = self._db
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA secure_delete=ON")
        db.execute("PRAGMA busy_timeout=5000")
        db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                character_id TEXT NOT NULL,
                player_id TEXT,
                kind TEXT NOT NULL,
                key TEXT,
                text TEXT NOT NULL,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                session_id TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                last_recalled_at REAL,
                recall_count INTEGER NOT NULL DEFAULT 0,
                superseded_by TEXT
            )"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS memories_scope ON memories(namespace, character_id, player_id, updated_at)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        db.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        try:
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "text, content='memories', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')"
            )
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN "
                "INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text); END"
            )
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN "
                "INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END"
            )
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text ON memories BEGIN "
                "INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text); "
                "INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text); END"
            )
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False

    # ── sync core (worker thread) ────────────────────────────────────────

    def _run(self, fn: Any, *args: Any) -> Any:
        with self._lock:
            try:
                return fn(*args)
            except sqlite3.Error as error:
                raise MemoryStoreError(f"Memory database error: {error}") from error

    def _put(self, record: MemoryRecord) -> None:
        placeholders = ",".join("?" * len(_COLUMNS))
        updates = ",".join(
            f"{column}=excluded.{column}" for column in _COLUMNS if column != "id"
        )
        self._db.execute(
            f"INSERT INTO memories ({','.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            _record_values(record),
        )

    def _get(self, record_id: str) -> MemoryRecord | None:
        row = self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (record_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def _delete(self, record_ids: Sequence[str]) -> int:
        if not record_ids:
            return 0
        cursor = self._db.execute(
            f"DELETE FROM memories WHERE id IN ({','.join('?' * len(record_ids))})",
            list(record_ids),
        )
        return cursor.rowcount

    def _query(
        self,
        namespace: str,
        character_id: str | None,
        player_ids: Any,
        kinds: Sequence[str] | None,
        include_inactive: bool,
        limit: int | None,
    ) -> list[MemoryRecord]:
        params: list[Any] = [namespace]
        clauses = ["m.namespace = ?"]
        if character_id is not None:
            clauses.append("m.character_id = ?")
            params.append(character_id)
        clauses.append(_player_clause(player_ids, params))
        if kinds:
            clauses.append(f"m.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if not include_inactive:
            clauses.append(
                "m.superseded_by IS NULL AND (m.expires_at IS NULL OR m.expires_at > ?)"
            )
            params.append(time.time())
        sql = f"SELECT m.* FROM memories m WHERE {' AND '.join(clauses)} ORDER BY m.updated_at DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [_row_to_record(row) for row in self._db.execute(sql, params)]

    def _search(
        self,
        text: str,
        namespace: str,
        character_id: str | None,
        player_ids: Sequence[str | None],
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        terms = query_terms(text)
        if not terms:
            return []
        if not self.fts:
            return self._search_python(
                terms, namespace, character_id, player_ids, limit
            )
        match = " OR ".join(
            f'"{term}"*' if len(term) >= 4 else f'"{term}"' for term in terms
        )
        params: list[Any] = [match, namespace]
        clauses = ["memories_fts MATCH ?", "m.namespace = ?"]
        if character_id is not None:
            clauses.append("m.character_id = ?")
            params.append(character_id)
        clauses.append(_player_clause(player_ids, params))
        clauses.append(
            "m.superseded_by IS NULL AND (m.expires_at IS NULL OR m.expires_at > ?)"
        )
        params.append(time.time())
        params.append(limit)
        rows = self._db.execute(
            "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts "
            "JOIN memories m ON m.rowid = memories_fts.rowid "
            f"WHERE {' AND '.join(clauses)} ORDER BY rank LIMIT ?",
            params,
        ).fetchall()
        if not rows:
            return []
        best = max(-row["rank"] for row in rows) or 1.0
        return [
            (_row_to_record(row), max(0.0, min(1.0, -row["rank"] / best)))
            for row in rows
        ]

    def _search_python(
        self,
        terms: list[str],
        namespace: str,
        character_id: str | None,
        player_ids: Sequence[str | None],
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        records = self._query(namespace, character_id, player_ids, None, False, None)
        return score_by_terms(records, terms, limit)

    def _touch(self, record_ids: Sequence[str], at: float) -> None:
        if record_ids:
            self._db.execute(
                f"UPDATE memories SET last_recalled_at = ?, recall_count = recall_count + 1 "
                f"WHERE id IN ({','.join('?' * len(record_ids))})",
                [at, *record_ids],
            )

    def _erase(self, namespace: str, character_id: str | None, player_id: Any) -> int:
        params: list[Any] = [namespace]
        clauses = ["namespace = ?"]
        if character_id is not None:
            clauses.append("character_id = ?")
            params.append(character_id)
        if player_id is not ALL_PLAYERS:
            if player_id is None:
                clauses.append("player_id IS NULL")
            else:
                clauses.append("player_id = ?")
                params.append(player_id)
        cursor = self._db.execute(
            f"DELETE FROM memories WHERE {' AND '.join(clauses)}", params
        )
        if self.fts:
            # Merging the index's segments drops the deleted entries from its pages too.
            self._db.execute(
                "INSERT INTO memories_fts(memories_fts) VALUES ('optimize')"
            )
        return cursor.rowcount

    # ── async API ────────────────────────────────────────────────────────

    async def put(self, record: MemoryRecord) -> None:
        await asyncio.to_thread(self._run, self._put, record)

    async def get(self, record_id: str) -> MemoryRecord | None:
        return await asyncio.to_thread(self._run, self._get, record_id)  # type: ignore[no-any-return]

    async def delete(self, record_ids: Sequence[str]) -> int:
        return await asyncio.to_thread(self._run, self._delete, list(record_ids))  # type: ignore[no-any-return]

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
        return await asyncio.to_thread(  # type: ignore[no-any-return]
            self._run,
            self._query,
            namespace,
            character_id,
            player_ids,
            kinds,
            include_inactive,
            limit,
        )

    async def search(
        self,
        text: str,
        *,
        namespace: str,
        character_id: str | None,
        player_ids: Sequence[str | None],
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(  # type: ignore[no-any-return]
            self._run,
            self._search,
            text,
            namespace,
            character_id,
            list(player_ids),
            limit,
        )

    async def touch(self, record_ids: Sequence[str], at: float) -> None:
        await asyncio.to_thread(self._run, self._touch, list(record_ids), at)

    async def erase(
        self,
        *,
        namespace: str,
        character_id: str | None = None,
        player_id: str | None | Any = ALL_PLAYERS,
    ) -> int:
        return await asyncio.to_thread(
            self._run, self._erase, namespace, character_id, player_id
        )  # type: ignore[no-any-return]

    async def close(self) -> None:
        def shut() -> None:
            with self._lock:
                self._db.close()

        await asyncio.to_thread(shut)


def score_by_terms(
    records: list[MemoryRecord], terms: list[str], limit: int
) -> list[tuple[MemoryRecord, float]]:
    """Word-overlap relevance, for stores without a text index."""
    if not terms:
        return []
    scored: list[tuple[MemoryRecord, float]] = []
    for record in records:
        words = set(re.findall(r"\w{2,}", record.text.lower()))
        hits = sum(
            1
            for term in terms
            if term in words or any(w.startswith(term) for w in words if len(term) >= 4)
        )
        if hits:
            # Shorter memories that match the same words are more on point.
            scored.append(
                (record, hits / len(terms) / (1.0 + math.log1p(len(words)) / 10.0))
            )
    if not scored:
        return []
    best = max(score for _, score in scored) or 1.0
    scored = [(record, score / best) for record, score in scored]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


class InMemoryStore:
    """Records in a dict: for tests, demos and sessions that should leave no trace."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def __repr__(self) -> str:
        return f"InMemoryStore({len(self._records)} records)"

    def _in_scope(
        self,
        record: MemoryRecord,
        namespace: str,
        character_id: str | None,
        player_ids: Any,
    ) -> bool:
        if record.namespace != namespace:
            return False
        if character_id is not None and record.character_id != character_id:
            return False
        return player_ids is ALL_PLAYERS or record.player_id in list(player_ids)

    async def put(self, record: MemoryRecord) -> None:
        self._records[record.id] = record.evolve()

    async def get(self, record_id: str) -> MemoryRecord | None:
        record = self._records.get(record_id)
        return record.evolve() if record else None

    async def delete(self, record_ids: Sequence[str]) -> int:
        return sum(
            1
            for record_id in record_ids
            if self._records.pop(record_id, None) is not None
        )

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
        now = time.time()
        found = [
            record.evolve()
            for record in self._records.values()
            if self._in_scope(record, namespace, character_id, player_ids)
            and (not kinds or record.kind in kinds)
            and (include_inactive or record.active(now))
        ]
        found.sort(key=lambda record: record.updated_at, reverse=True)
        return found[:limit] if limit else found

    async def search(
        self,
        text: str,
        *,
        namespace: str,
        character_id: str | None,
        player_ids: Sequence[str | None],
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        records = await self.query(
            namespace=namespace, character_id=character_id, player_ids=player_ids
        )
        return score_by_terms(records, query_terms(text), limit)

    async def touch(self, record_ids: Sequence[str], at: float) -> None:
        for record_id in record_ids:
            record = self._records.get(record_id)
            if record is not None:
                record.last_recalled_at = at
                record.recall_count += 1

    async def erase(
        self,
        *,
        namespace: str,
        character_id: str | None = None,
        player_id: str | None | Any = ALL_PLAYERS,
    ) -> int:
        doomed = [
            record.id
            for record in self._records.values()
            if record.namespace == namespace
            and (character_id is None or record.character_id == character_id)
            and (player_id is ALL_PLAYERS or record.player_id == player_id)
        ]
        return await self.delete(doomed)

    async def close(self) -> None:
        return None
