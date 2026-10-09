"""What a session heard, remembered, decided, said and did, on one timeline.

::

    async with npc.connect(player_id="p42", trace="./traces/kael.jsonl") as session: ...

    python -m eesi trace ./traces/kael.jsonl

Each line of the file is one event with ``t`` (seconds since the session
started). By default a trace holds no words: transcripts, replies, tool
arguments and results and memory texts are replaced by their length, so a
trace can be kept and shared without keeping what people said. Pass
``Trace(path, content=True)`` to record the words while you debug.

Latency here is measured at the client: from the end of the player's turn
(``audio.input.ended``) to the first audio of the reply reaching the SDK. It
does not include your speaker's own buffering.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .events import AudioOutputChunk, Event

__all__ = ["SessionMetrics", "Trace", "format_trace"]

_CONTENT_FIELDS = {
    "text",
    "arguments",
    "result",
    "output",
    "disclosure",
    "error",
    "message",
    "query",
}


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return {"chars": len(value)}
    if isinstance(value, dict):
        return {"keys": sorted(str(k) for k in value)}
    return {"redacted": True}


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Trace:
    """A JSONL timeline of one session."""

    def __init__(
        self, path: str | Path | None = None, *, content: bool = False
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.content = content
        self.lines: list[dict[str, Any]] = []
        self._file: IO[str] | None = None
        self._t0 = time.monotonic()

    def open(self) -> None:
        if self.path is not None and self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()
        self.write("trace.start", wall=time.time(), content=self.content)

    def write(self, kind: str, **data: Any) -> None:
        line = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **data}
        self.lines.append(line)
        if len(self.lines) > 5_000:
            del self.lines[:1_000]
        if self._file is not None:
            self._file.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            self._file.flush()

    def event(self, event: Event) -> None:
        if isinstance(event, AudioOutputChunk):
            return
        data: dict[str, Any] = {}
        for f in dataclasses.fields(event):
            if f.name in ("at", "session_id", "player_id"):
                continue
            value = getattr(event, f.name)
            if f.name == "records":
                value = [
                    {
                        "id": r.id,
                        "kind": r.kind,
                        "player": bool(r.player_id),
                        **({"text": r.text} if self.content else {}),
                    }
                    for r in value
                ]
            elif f.name in ("added", "updated"):
                value = [
                    {
                        "id": r.id,
                        "kind": r.kind,
                        **({"text": r.text} if self.content else {}),
                    }
                    for r in value
                ]
            elif f.name == "turns":
                value = [
                    {
                        "speaker": t.speaker,
                        "player": t.player_id,
                        **(
                            {"text": t.text} if self.content else {"chars": len(t.text)}
                        ),
                    }
                    for t in value
                ]
            elif (
                f.name in _CONTENT_FIELDS
                and not self.content
                and value not in (None, "", {})
            ):
                value = _redact(value)
            else:
                value = _plain(value)
            data[f.name] = value
        self.write(event.type, **data)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


@dataclass
class SessionMetrics:
    """Counts and client-side latency for one session."""

    turns: int = 0
    replies: int = 0
    interruptions: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    reconnects: int = 0
    #: Seconds from the end of each player turn to the first audio of its reply.
    latencies: list[float] = field(default_factory=list)
    audio_in_seconds: float = 0.0
    audio_out_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        latencies = sorted(self.latencies)

        def pct(p: float) -> float | None:
            if not latencies:
                return None
            return round(
                latencies[min(len(latencies) - 1, int(p * (len(latencies) - 1) + 0.5))],
                3,
            )

        return {
            "turns": self.turns,
            "replies": self.replies,
            "interruptions": self.interruptions,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "reconnects": self.reconnects,
            "latency_p50": pct(0.5),
            "latency_p95": pct(0.95),
            "latency_mean": round(statistics.fmean(latencies), 3)
            if latencies
            else None,
            "audio_in_seconds": round(self.audio_in_seconds, 1),
            "audio_out_seconds": round(self.audio_out_seconds, 1),
        }


def _who(turn: dict[str, Any]) -> str:
    if turn.get("player"):
        return str(turn["player"])
    if turn.get("speaker"):
        return f"Speaker {turn['speaker']}"
    return "player"


def format_trace(lines: list[dict[str, Any]]) -> str:
    """A trace as a readable timeline."""
    out: list[str] = []
    for line in lines:
        t = f"{line.get('t', 0):8.2f}s"
        kind = line.get("kind", "")
        text = line.get("text")
        words = text if isinstance(text, str) else ""
        if kind == "transcript.final":
            turns = line.get("turns") or []
            heard = (
                " / ".join(f"{_who(turn)}: {turn.get('text', '…')}" for turn in turns)
                or words
                or "…"
            )
            out.append(f"{t}  heard     {heard}")
        elif kind == "response.completed":
            if not words and not isinstance(text, dict):
                continue  # a reply that was only tool calls; the tool line says what happened
            latency = line.get("latency")
            lat = f" ({latency:.2f}s)" if isinstance(latency, (int, float)) else ""
            out.append(f"{t}  said{lat:<9} {words or '…'}")
        elif kind == "response.cancelled":
            out.append(f"{t}  cut off   {words or '…'} [{line.get('reason')}]")
        elif kind == "speech.interrupted":
            out.append(f"{t}  interrupted by the player")
        elif kind in ("tool.completed", "tool.failed"):
            output = (
                line.get("output") if kind == "tool.completed" else line.get("error")
            )
            detail = output if isinstance(output, str) else "…"
            args = line.get("arguments")
            shown_args = (
                json.dumps(args, ensure_ascii=False)
                if isinstance(args, dict) and "keys" not in args
                else "…"
            )
            label = "tool" if kind == "tool.completed" else "tool ✗"
            out.append(f"{t}  {label:<9} {line.get('name')}({shown_args}) → {detail}")
        elif kind == "context.updated":
            size = text.get("chars", 0) if isinstance(text, dict) else len(words)
            shown = words.replace("\n", " | ") if words else f"({size} chars)"
            out.append(f"{t}  context   [{line.get('block')}] {shown}")
        elif kind == "memory.retrieved":
            records = line.get("records") or []
            out.append(
                f"{t}  recalled  {len(records)} memories"
                + "".join(
                    f"\n{'':12}- {r.get('text', r.get('id'))}" for r in records[:6]
                )
            )
        elif kind == "memory.updated":
            added = line.get("added") or []
            out.append(
                f"{t}  learned   {len(added)} memories"
                + "".join(f"\n{'':12}+ {r.get('text', r.get('id'))}" for r in added[:8])
            )
        elif kind in (
            "session.started",
            "session.ended",
            "reconnecting",
            "reconnected",
            "session.hangup",
            "error",
        ):
            detail = {k: v for k, v in line.items() if k not in ("t", "kind")}
            out.append(
                f"{t}  {kind:<9} {json.dumps(detail, ensure_ascii=False) if detail else ''}"
            )
        elif kind == "metrics":
            out.append(
                f"{t}  metrics   {json.dumps({k: v for k, v in line.items() if k not in ('t', 'kind')})}"
            )
    return "\n".join(out)
