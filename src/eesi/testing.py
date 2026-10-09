"""Scripted players, for regression tests, benchmarks and repeatable demos.

::

    from eesi.testing import run_script

    result = await run_script(kael, ["Hi, I'm Sam.", "Could you open the north gate?"], player_id="sam")
    assert any(call.name == "open_gate" for call in result.tool_calls)
    print(result.metrics["latency_p50"])

A scripted player speaks lines synthesized by EESI's own text to speech, with
pauses between them, into a real Nur Live session, in real time. Everything
the character does is collected. This exercises the whole path (speech
recognition, turn-taking, the model, tools, memory), so it costs usage like
any session and its results vary the way a model's do: assert on what must
hold (a tool was called, a fact was recalled), not on exact words.
"""

from __future__ import annotations

import array
import asyncio
import io
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .audio import WavInput, WavOutput
from .events import Event

if TYPE_CHECKING:
    from .character import Character
    from .client import NurClient
    from .memory import MemoryRecord

__all__ = ["ScriptResult", "run_script", "scripted_player"]

#: Seconds of quiet after a line, long enough for the character to answer it.
DEFAULT_PAUSE_S = 9.0


async def _speak(
    nur: NurClient, text: str, voice: str, language: str
) -> tuple[array.array, int]:
    voice_id = await nur.resolve_voice(voice)
    response = await nur._request(
        "POST",
        "/audio/speech",
        json={
            "model": "nur-tts-v1",
            "input": text,
            "voice": voice_id,
            "language": language,
            "response_format": "wav",
        },
    )
    with wave.open(io.BytesIO(response.content)) as clip:
        rate, channels = clip.getframerate(), clip.getnchannels()
        samples = array.array("h")
        samples.frombytes(clip.readframes(clip.getnframes()))
    if channels > 1:
        samples = array.array(
            "h",
            (
                sum(samples[i : i + channels]) // channels
                for i in range(0, len(samples), channels)
            ),
        )
    return samples, rate


async def scripted_player(
    nur: NurClient,
    turns: Sequence[str | tuple[str, float]],
    *,
    voice: str = "Nova",
    language: str = "en",
    path: str | Path | None = None,
    lead_in: float = 1.0,
) -> WavInput:
    """A microphone that says *turns*: each a line, or ``(line, pause_after_seconds)``.

    Give the player an English-trained voice (Nova, Atlas) for English lines.
    """
    rate = 24_000
    pieces: list[tuple[array.array, float]] = []
    for turn in turns:
        text, pause = (turn, DEFAULT_PAUSE_S) if isinstance(turn, str) else turn
        samples, rate = await _speak(nur, text, voice, language)
        pieces.append((samples, pause))
    audio = array.array("h", bytes(int(rate * lead_in) * 2))
    for samples, pause in pieces:
        audio.extend(samples)
        audio.extend(array.array("h", bytes(int(rate * pause) * 2)))
    target = (
        Path(path)
        if path is not None
        else Path(tempfile.mkstemp(prefix="eesi-player-", suffix=".wav")[1])
    )
    with wave.open(str(target), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(audio.tobytes())
    return WavInput(target, tail=2.0)


@dataclass
class ScriptResult:
    """What happened in a scripted session."""

    events: list[Event] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    tool_calls: list[Any] = field(default_factory=list)
    tool_failures: list[Any] = field(default_factory=list)
    recalled: list[MemoryRecord] = field(default_factory=list)
    learned: list[MemoryRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    recording: Path | None = None

    @property
    def said_text(self) -> str:
        return " ".join(self.said)


async def run_script(
    character: Character,
    turns: Sequence[str | tuple[str, float]],
    *,
    player_id: str | None = "scripted-player",
    player_name: str | None = None,
    voice: str = "Nova",
    language: str = "en",
    lead_in: float = 4.0,
    settle: float = 6.0,
    record: str | Path | None = None,
    **connect: Any,
) -> ScriptResult:
    """Run one scripted conversation with *character* and collect everything it did.

    The player waits ``lead_in`` seconds before speaking, enough for a
    greeting; set it lower to have the player talk over the greeting.
    """
    source = await scripted_player(
        character.client, turns, voice=voice, language=language, lead_in=lead_in
    )
    output = WavOutput(record) if record is not None else _Discard()
    result = ScriptResult(recording=Path(record) if record is not None else None)
    session = character.connect(
        player_id, player_name=player_name, input=source, output=output, **connect
    )

    @session.on("*")
    def collect(event: Event) -> None:
        result.events.append(event)
        if event.type == "transcript.final":
            result.heard.append(event.text)  # type: ignore[attr-defined]
        elif event.type == "response.completed" and event.text:  # type: ignore[attr-defined]
            result.said.append(event.text)  # type: ignore[attr-defined]
        elif event.type == "tool.completed":
            result.tool_calls.append(event)
        elif event.type == "tool.failed":
            result.tool_failures.append(event)
        elif event.type == "memory.retrieved":
            result.recalled.extend(event.records)  # type: ignore[attr-defined]
        elif event.type == "memory.updated":
            result.learned.extend(event.added)  # type: ignore[attr-defined]

    async with session:
        result.session_id = session.session_id
        await source.finished.wait()
        await asyncio.sleep(settle)
        result.metrics = session.metrics.summary()
    try:
        Path(source.path).unlink()
    except OSError:
        pass
    return result


class _Discard:
    """A speaker nobody listens to, keeping real-time playout so barge-in behaves."""

    sample_rate = 24_000

    def __init__(self) -> None:
        from .audio import PlayoutClock

        self._clock = PlayoutClock(self.sample_rate)

    def play(self, pcm: bytes) -> None:
        self._clock.push(len(pcm) // 2)

    def flush(self) -> None:
        self._clock.flush()

    def buffered_seconds(self) -> float:
        return self._clock.buffered_seconds()
