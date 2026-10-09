"""Learning memories from a finished conversation.

After a session the SDK can ask a language model what is worth keeping: who
the player is, what they prefer, what happened, what was promised, how the
relationship changed, and which old memories a correction replaces. The
default extractor asks EESI's own text model (``nur-llm-v1`` over
``/v1/chat/completions``) with your API key; pass ``extractor=`` to use your
own model, or ``MemoryPolicy(extract=False)`` to learn nothing on your behalf.

This updates a memory store. It does not train or fine-tune any model, and
nothing it writes is shared beyond the character and player it belongs to.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .records import MemoryRecord

log = logging.getLogger("eesi")

__all__ = [
    "ExtractionRequest",
    "ExtractionResult",
    "Extractor",
    "LLMExtractor",
    "MemoryCandidate",
    "PREFERENCE_KEYS",
]

#: Preferences the SDK knows how to apply, beside free-form ones.
PREFERENCE_KEYS = ("language", "verbosity", "pace", "formality", "name")
_IMPORTANCE = {"low": 0.3, "medium": 0.6, "high": 0.9}
_KINDS = {"fact", "preference", "event", "promise", "relationship", "note"}
#: The tail of a long conversation the extractor reads, in characters.
MAX_TRANSCRIPT_CHARS = 16_000


@dataclass(frozen=True)
class ExtractionRequest:
    character_name: str
    player_id: str
    player_name: str | None
    #: (speaker, text) lines, oldest first. Speakers are "player", the character's name, or "event".
    transcript: Sequence[tuple[str, str]]
    existing: Sequence[MemoryRecord]


@dataclass(frozen=True)
class MemoryCandidate:
    text: str
    kind: str = "fact"
    importance: float = 0.6
    key: str | None = None
    #: The id of an existing memory this one corrects or updates.
    replaces: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    summary: str | None = None
    memories: tuple[MemoryCandidate, ...] = ()
    #: Ids of existing memories to delete: the player asked, or they proved false.
    forget: tuple[str, ...] = ()
    #: What the model returned, for debugging an extraction that went wrong.
    raw: str | None = field(default=None, repr=False)


class Extractor(Protocol):
    async def __call__(self, request: ExtractionRequest) -> ExtractionResult: ...


Chat = Callable[[list[dict[str, Any]]], Awaitable[str]]

_SYSTEM = """You keep the long-term memory of {character}, a character who talks with a player by voice. Read the conversation and decide what {character} should remember about this player next time they meet.

Answer with JSON only, in exactly this shape:
{{"summary": "...", "memories": [{{"text": "...", "kind": "fact", "importance": "medium", "key": null, "replaces": null}}], "forget": []}}

Rules:
- "summary": one or two sentences on what happened in this conversation, from {character}'s point of view. Use "" if nothing happened.
- "memories": at most 8 things worth remembering next time: facts about the player (their name, background, situation), their preferences, events that happened to or with them, promises either side made, and changes in how the player and {character} stand. Each "text" is one short sentence about the player in the third person ("They ...", or their name).
- "kind" is one of: fact, preference, event, promise, relationship.
- "importance" is one of: low, medium, high.
- For a preference about how to talk with them, "key" is one of: language, verbosity, pace, formality, name. Otherwise "key" is null.
- If a memory corrects or updates an existing memory below, "replaces" is that memory's id (like "m2"). Otherwise null.
- "forget": ids of existing memories the player asked {character} to forget, or that the conversation showed to be wrong.
- Remember only what the conversation shows. Never invent. Skip small talk, and skip what {character} said about itself.
- Lines marked [event] are things that happened in the game, not words the player said."""


def _transcript_text(request: ExtractionRequest) -> str:
    lines = []
    for speaker, text in request.transcript:
        text = " ".join(text.split())
        if not text:
            continue
        if speaker == "player":
            label = request.player_name or "Player"
        elif speaker == "event":
            label = "[event]"
        else:
            label = speaker
        lines.append(f"{label}: {text}")
    joined = "\n".join(lines)
    if len(joined) > MAX_TRANSCRIPT_CHARS:
        joined = "…" + joined[-MAX_TRANSCRIPT_CHARS:]
    return joined


def parse_result(text: str, ids: dict[str, str]) -> ExtractionResult:
    """The model's answer as an ExtractionResult; tolerant of fences and chatter."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return ExtractionResult(raw=text)
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return ExtractionResult(raw=text)
    if not isinstance(data, dict):
        return ExtractionResult(raw=text)
    summary = data.get("summary")
    candidates: list[MemoryCandidate] = []
    for item in data.get("memories") or []:
        if not isinstance(item, dict):
            continue
        memory_text = item.get("text")
        if not isinstance(memory_text, str) or not memory_text.strip():
            continue
        kind = item.get("kind") if item.get("kind") in _KINDS else "fact"
        importance = item.get("importance")
        weight = (
            _IMPORTANCE.get(str(importance).lower(), 0.6)
            if not isinstance(importance, (int, float))
            else float(importance)
        )
        key = (
            item.get("key")
            if isinstance(item.get("key"), str) and item.get("key")
            else None
        )
        replaces = ids.get(str(item.get("replaces"))) if item.get("replaces") else None
        candidates.append(
            MemoryCandidate(
                text=" ".join(memory_text.split())[:500],
                kind=str(kind),
                importance=min(1.0, max(0.0, weight)),
                key=key if kind == "preference" else None,
                replaces=replaces,
            )
        )
    forget = tuple(ids[str(i)] for i in data.get("forget") or [] if str(i) in ids)
    return ExtractionResult(
        summary=" ".join(summary.split())[:600]
        if isinstance(summary, str) and summary.strip()
        else None,
        memories=tuple(candidates[:8]),
        forget=forget,
        raw=text,
    )


#: How a game's own rules are added to the extraction prompt. Shared with the TypeScript SDK.
RULES_SUFFIX = (
    "\n\nWhat {character} may remember, set by the game, which wins over the rules "
    "above where they differ:\n{rules}"
)


class LLMExtractor:
    """Memory extraction with a chat model. *chat* takes messages and returns the reply text.

    *rules* narrow what may be kept, in your words: ``"Only game facts: what
    they build, their favourite items, the name they use in the game. Never
    age, school, location or social accounts."``
    """

    def __init__(self, chat: Chat, *, rules: str | None = None) -> None:
        self._chat = chat
        self._rules = rules.strip() if rules and rules.strip() else None

    async def __call__(self, request: ExtractionRequest) -> ExtractionResult:
        transcript = _transcript_text(request)
        if not transcript.strip():
            return ExtractionResult()
        ids = {
            f"m{index + 1}": record.id for index, record in enumerate(request.existing)
        }
        existing = (
            "\n".join(
                f"m{index + 1}: {' '.join(record.text.split())}"
                for index, record in enumerate(request.existing)
            )
            or "(none)"
        )
        who = request.player_name or "the player"
        messages = [
            {
                "role": "system",
                "content": _SYSTEM.format(character=request.character_name)
                + (
                    RULES_SUFFIX.format(
                        character=request.character_name, rules=self._rules
                    )
                    if self._rules
                    else ""
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The player is {who}.\n\nExisting memories about them:\n{existing}\n\n"
                    f"Conversation, oldest first:\n{transcript}\n\nAnswer with the JSON only."
                ),
            },
        ]
        reply = await self._chat(messages)
        result = parse_result(reply, ids)
        if (
            result.raw is not None
            and not result.memories
            and result.summary is None
            and "{" not in reply
        ):
            log.warning("memory extraction returned no JSON")
        return result
