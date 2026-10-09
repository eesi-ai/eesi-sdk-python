"""Choosing what a character remembers right now, and how it reads.

A session cannot carry everything a character ever learned about a player:
the memory block has a budget (a few kilobytes of a 24 KB context). Records
compete on relevance to what is being said, importance and recency; a few
kinds always make it in because they change how every reply sounds: the
player's preferences, the relationship, and what happened last time.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from .._protocol import byte_len
from .records import MemoryRecord

__all__ = [
    "DEFAULT_BLOCK_BYTES",
    "RecallWeights",
    "compose_block",
    "rank",
    "relative_age",
]

#: The memory block's default size: about 700 tokens, a fraction of the context.
DEFAULT_BLOCK_BYTES = 3_000


@dataclass(frozen=True)
class RecallWeights:
    relevance: float = 0.55
    importance: float = 0.25
    recency: float = 0.20
    #: A memory this many days old counts half as recent.
    half_life_days: float = 30.0


DEFAULT_WEIGHTS = RecallWeights()


def rank(
    records: Iterable[MemoryRecord],
    relevance: dict[str, float] | None = None,
    *,
    weights: RecallWeights = DEFAULT_WEIGHTS,
    now: float | None = None,
) -> list[tuple[MemoryRecord, float]]:
    """Records with a score in 0..1, best first."""
    now = time.time() if now is None else now
    relevance = relevance or {}
    scored = []
    for record in records:
        age_days = max(0.0, now - record.updated_at) / 86_400
        recency = 0.5 ** (age_days / weights.half_life_days)
        score = (
            weights.relevance * relevance.get(record.id, 0.0)
            + weights.importance * record.importance * (0.5 + 0.5 * record.confidence)
            + weights.recency * recency
        )
        scored.append((record, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def relative_age(then: float, now: float | None = None) -> str:
    now = time.time() if now is None else now
    days = (now - then) / 86_400
    if days < 1 and time.localtime(then).tm_yday == time.localtime(now).tm_yday:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 14:
        return f"{int(days)} days ago"
    if days < 60:
        return f"{int(days // 7)} weeks ago"
    if days < 730:
        return f"{int(days // 30)} months ago"
    return f"{int(days // 365)} years ago"


def _line(record: MemoryRecord, now: float) -> str:
    text = " ".join(record.text.split())
    if record.kind in ("event", "promise") or record.source == "extraction":
        return f"- {text} ({relative_age(record.updated_at, now)})"
    return f"- {text}"


def compose_block(
    ranked: list[tuple[MemoryRecord, float]],
    *,
    character_name: str,
    player_name: str | None,
    first_meeting: bool,
    budget_bytes: int = DEFAULT_BLOCK_BYTES,
    session_preferences: dict[str, str] | None = None,
    now: float | None = None,
) -> tuple[str, list[MemoryRecord]]:
    """The ``memory`` block's text and the records it used."""
    now = time.time() if now is None else now
    who = player_name or "this player"
    used: list[MemoryRecord] = []
    personal = [(r, s) for r, s in ranked if r.player_id is not None]
    world = [(r, s) for r, s in ranked if r.player_id is None]

    preferences: dict[str, str] = {}
    for record, _ in personal:
        if record.kind == "preference":
            key = record.key or record.text
            if key not in preferences:
                preferences[key] = record.text
                used.append(record)
    for key, value in (session_preferences or {}).items():
        preferences[key] = value

    relationship = [r for r, _ in personal if r.kind == "relationship"][:2]
    summaries = sorted(
        (r for r, _ in personal if r.kind == "summary"),
        key=lambda r: r.updated_at,
        reverse=True,
    )[:2]
    used.extend(relationship)
    used.extend(summaries)
    taken = {r.id for r in used}

    header: list[str] = []
    if first_meeting and not personal:
        header.append(
            f"You have not met {who} before: this is your first conversation with them."
        )
    else:
        header.append(
            f"What you remember about {who} from earlier conversations. These are your own memories: "
            "use them naturally when they matter, never recite them as a list, and if the player "
            "corrects one, believe the player."
        )
    lines = list(header)
    if preferences:
        lines.append(
            "How they like to be spoken to: " + "; ".join(preferences.values())
        )
    for record in relationship:
        lines.append(f"Your relationship: {' '.join(record.text.split())}")
    if summaries:
        latest = summaries[0]
        lines.append(
            f"Last conversation ({relative_age(latest.updated_at, now)}): {' '.join(latest.text.split())}"
        )

    def fits(extra: str) -> bool:
        return byte_len("\n".join([*lines, extra])) <= budget_bytes

    for record, _ in personal:
        if record.id in taken or record.kind in (
            "preference",
            "relationship",
            "summary",
        ):
            continue
        line = _line(record, now)
        if not fits(line):
            continue
        lines.append(line)
        used.append(record)
        taken.add(record.id)

    world_lines: list[str] = []
    for record, _ in world:
        if record.id in taken:
            continue
        line = _line(record, now)
        if not fits(
            "\n".join(["What you know from around the world:", *world_lines, line])
        ):
            continue
        world_lines.append(line)
        used.append(record)
        taken.add(record.id)
    if world_lines:
        lines.append("What you know from around the world:")
        lines.extend(world_lines)
    if character_name and not used and not first_meeting:
        return "", []
    return "\n".join(lines), used
