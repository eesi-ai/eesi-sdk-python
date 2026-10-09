"""The wire: client events the SDK sends and the server events it reads.

Nur Live speaks the OpenAI Realtime protocol plus a few ``eesi.*`` events
(https://docs.eesi.ai/realtime/live). The builders here are plain dicts so a
trace shows exactly what went out. Server event names have aliases across
protocol versions; :func:`canonical_type` collapses them to one name, as every
EESI client does.
"""

from __future__ import annotations

import base64
import re
import uuid
from typing import Any

#: The gateway's native input rate: 16-bit mono PCM unless the session says otherwise.
INPUT_SAMPLE_RATE = 16_000
#: The rate the SDK asks the character's voice at. 16 kHz is also accepted.
OUTPUT_SAMPLE_RATE = 24_000

#: Limits the gateway enforces on context blocks, tool results and instructions.
MAX_CONTEXT_KEYS = 8
MAX_CONTEXT_BLOCK_BYTES = 8_000
MAX_CONTEXT_TOTAL_BYTES = 24_000
MAX_TOOL_RESULT_BYTES = 16_000
MAX_INSTRUCTIONS_BYTES = 64_000
CONTEXT_KEY = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,47}$")

_ALIASES = {
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
}


def canonical_type(kind: str) -> str:
    """One name per server event, whichever protocol version sent it."""
    return _ALIASES.get(kind, kind)


def event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:20]}"


# ── client events ────────────────────────────────────────────────────────


def session_update(session: dict[str, Any]) -> dict[str, Any]:
    return {"type": "session.update", "session": {"type": "realtime", **session}}


def context_update(key: str, text: str) -> dict[str, Any]:
    return {"type": "eesi.context.update", "key": key, "text": text}


def append_audio(pcm16: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm16).decode("ascii"),
    }


def user_text(text: str, *, item_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    return {"type": "conversation.item.create", "item": item}


def tool_output(call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def response_create(
    instructions: str | None = None,
    *,
    tool_choice: str | None = None,
    input_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A reply. *instructions* steer this one reply; the persona still stands.

    The gateway composes per-response instructions after the session's own,
    so they add to the character rather than replacing it.
    """
    response: dict[str, Any] = {}
    if instructions:
        response["instructions"] = instructions
    if tool_choice:
        response["tool_choice"] = tool_choice
    if input_items:
        response["input"] = input_items
    event: dict[str, Any] = {"type": "response.create", "event_id": event_id()}
    if response:
        event["response"] = response
    return event


def event_reply(event: str, instructions: str) -> dict[str, Any]:
    """A reply to something that happened in the application.

    The event joins the conversation as its newest turn, labelled as the
    application's, and *instructions* say what to do with it. With the event
    only in the instructions, the model often answered the player's previous
    question again instead of reacting to the event.
    """
    return response_create(
        instructions,
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"[Game event, not the player speaking] {event}",
                    }
                ],
            }
        ],
    )


def response_cancel() -> dict[str, Any]:
    return {"type": "response.cancel"}


def keepalive() -> dict[str, Any]:
    return {"type": "eesi.keepalive"}


def output_status(
    *,
    response_id: str,
    sample_rate: int,
    received: int,
    rendered: int,
    discarded: int,
    flushed: bool,
    buffered_ms: int,
    muted: bool = False,
) -> dict[str, Any]:
    """What the speaker has played, so the runtime keeps only what was heard on a cut.

    An estimate from the playout clock, never a claim that a person heard it.
    """
    active = buffered_ms > 0 and not muted
    return {
        "type": "eesi.output.status",
        "active": active,
        "available": True,
        "buffered_ms": buffered_ms,
        "playback": {
            "response_id": response_id,
            "sample_rate": sample_rate,
            "received_samples": received,
            "rendered_samples": rendered,
            "discarded_samples": discarded,
            "flushed": flushed,
            "clock_running": active,
            "muted": muted,
        }
        if response_id
        else None,
    }


# ── server events ────────────────────────────────────────────────────────


def response_id_of(event: dict[str, Any]) -> str:
    """The response an event belongs to, from wherever this event type keeps it."""
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    value = event.get("response_id")
    return value if isinstance(value, str) else ""


def error_of(event: dict[str, Any]) -> tuple[str, str | None]:
    error = event.get("error")
    error = error if isinstance(error, dict) else {}
    message = error.get("message")
    code = error.get("code") or error.get("type")
    return (
        message if isinstance(message, str) and message else "Unknown realtime error",
        code if isinstance(code, str) and code else None,
    )


def response_status(event: dict[str, Any]) -> tuple[str | None, str | None]:
    response = event.get("response")
    response = response if isinstance(response, dict) else {}
    details = response.get("status_details")
    details = details if isinstance(details, dict) else {}
    status = response.get("status")
    reason = details.get("reason")
    return (
        status if isinstance(status, str) else None,
        reason if isinstance(reason, str) else None,
    )


SPEAKER_LINE = re.compile(r"^\s*speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def split_speaker_turns(text: str) -> list[tuple[int | None, str]]:
    """``Speaker N:`` tagged lines back into (speaker, text) runs.

    An untagged line continues whoever spoke last, and a transcript with no
    tags is one unattributed turn. Nur Live tags lines once a session has
    heard a second voice.
    """
    trimmed = text.strip()
    if not trimmed:
        return []
    lines = trimmed.split("\n")

    def speaker_of(line: str) -> int | None:
        match = SPEAKER_LINE.match(line)
        if not match:
            return None
        number = int(match.group(1))
        return number if number >= 1 else None

    if not any(speaker_of(line) is not None for line in lines):
        return [(None, trimmed)]
    turns: list[list[Any]] = []
    for line in lines:
        match = SPEAKER_LINE.match(line)
        speaker = speaker_of(line)
        body = (match.group(2) if match and speaker is not None else line).strip()
        if match and speaker is not None:
            if turns and turns[-1][0] == speaker:
                turns[-1][1] = f"{turns[-1][1]} {body}".strip()
            else:
                turns.append([speaker, body])
            continue
        if not body:
            continue
        if turns:
            turns[-1][1] = f"{turns[-1][1]} {body}".strip()
        else:
            turns.append([None, body])
    return [(speaker, words) for speaker, words in turns if words]


def clip_bytes(text: str, limit: int) -> str:
    """*text* cut to at most *limit* UTF-8 bytes on a character boundary, marked with an ellipsis."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    cut = encoded[: max(0, limit - 3)].decode("utf-8", errors="ignore")
    return cut + "…"


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))
