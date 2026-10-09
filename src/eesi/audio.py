"""Where a session's audio comes from and goes to.

A session needs neither. Without audio it is a text conversation with a
voice you can still receive as ``audio.output.chunk`` events, and you can
feed it audio yourself with :meth:`Session.send_audio` (a game server relaying
a player's microphone). With audio objects it runs the microphone and the
speaker for you:

- :func:`local` (``pip install "eesi-sdk[audio]"``): this computer's
  microphone and speaker.
- :class:`WavInput` / :class:`WavOutput`: a recording heard as if live, and
  the character's voice written to a file. Repeatable runs and tests.

Audio is 16-bit mono PCM. The session sends 16 kHz and receives 24 kHz;
anything else is resampled here, chunk by chunk, without clicks at the joins.

Python has no acoustic echo canceller. On open speakers the character would
hear itself and stop mid-sentence, so :class:`LocalAudio` guards the
microphone while the character speaks (``echo="auto"``). Use headphones and
``echo="off"`` for the best barge-in; the browser and phone SDKs cancel echo
properly through WebRTC.
"""

from __future__ import annotations

import array
import asyncio
import math
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from ._protocol import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE
from .errors import AudioDeviceError

__all__ = [
    "AudioInput",
    "AudioOutput",
    "LocalAudio",
    "PlayoutClock",
    "Resampler",
    "WavInput",
    "WavOutput",
    "local",
    "rms",
]

FRAME_SECONDS = 0.02

try:  # numpy makes resampling and levels cheap; the SDK works without it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised where numpy is absent
    _np = None  # type: ignore[assignment]


@runtime_checkable
class AudioInput(Protocol):
    """A microphone: PCM16 mono chunks at ``sample_rate``, in real time."""

    sample_rate: int

    def frames(self) -> AsyncIterator[bytes]: ...


@runtime_checkable
class AudioOutput(Protocol):
    """A speaker. ``play`` must return at once; ``flush`` drops everything queued."""

    sample_rate: int

    def play(self, pcm: bytes) -> None: ...

    def flush(self) -> None: ...

    def buffered_seconds(self) -> float: ...


def _samples(pcm: bytes) -> array.array:
    values = array.array("h")
    values.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":  # pragma: no cover - PCM on the wire is little-endian
        values.byteswap()
    return values


def _pcm(values: array.array) -> bytes:
    if sys.byteorder == "big":  # pragma: no cover
        values = array.array("h", values)
        values.byteswap()
    return values.tobytes()


def rms(pcm: bytes) -> float:
    """Root-mean-square level of PCM16, 0..1."""
    if len(pcm) < 2:
        return 0.0
    if _np is not None:
        data = (
            _np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(
                _np.float32
            )
            / 32768.0
        )
        return float(_np.sqrt(_np.mean(data * data))) if data.size else 0.0
    values = _samples(pcm)
    return (
        math.sqrt(sum(v * v for v in values) / len(values)) / 32768.0 if values else 0.0
    )


class Resampler:
    """Linear resampling of a PCM16 stream whose output does not depend on how it is chunked.

    The next output position is carried across calls as an exact rational
    offset with the last input sample, so resampling a stream in pieces gives
    the same samples as resampling it whole.
    Linear interpolation is enough for speech between 16 and 48 kHz.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("Sample rates must be positive.")
        self.src = src_rate
        self.dst = dst_rate
        self._pos = 0
        self._last: list[int] = []

    def reset(self) -> None:
        self._pos = 0
        self._last = []

    def process(self, pcm: bytes) -> bytes:
        if self.src == self.dst:
            return pcm[: len(pcm) - len(pcm) % 2]
        data = self._last + list(_samples(pcm))
        if len(data) < 2:
            self._last = data
            return b""
        limit = (len(data) - 1) * self.dst
        count = max(0, -(-(limit - self._pos) // self.src))
        if _np is not None:
            positions = self._pos + _np.arange(count, dtype=_np.int64) * self.src
            index = positions // self.dst
            frac = (positions % self.dst).astype(_np.float64) / self.dst
            source = _np.asarray(data, dtype=_np.float64)
            out_values = _np.round(
                source[index] * (1.0 - frac) + source[index + 1] * frac
            )
            out = _np.clip(out_values, -32768, 32767).astype("<i2").tobytes()
        else:
            result = array.array("h")
            for k in range(count):
                position = self._pos + k * self.src
                i, rem = divmod(position, self.dst)
                frac = rem / self.dst
                value = round(data[i] * (1.0 - frac) + data[i + 1] * frac)
                result.append(max(-32768, min(32767, value)))
            out = _pcm(result)
        next_pos = self._pos + count * self.src
        keep_from = min(next_pos // self.dst, len(data))
        self._last = data[keep_from:]
        self._pos = next_pos - keep_from * self.dst
        return out


class PlayoutClock:
    """When queued audio will have played, for outputs that play on their own.

    An estimate, never a claim that a person heard it: audio handed to a
    speaker is assumed to play back to back in real time from when it was
    handed over.
    """

    def __init__(self, sample_rate: int, clock: Any = time.monotonic) -> None:
        self.rate = sample_rate
        self._clock = clock
        self._ends_at = 0.0

    def push(self, samples: int) -> None:
        now = self._clock()
        self._ends_at = max(self._ends_at, now) + samples / self.rate

    def buffered_seconds(self) -> float:
        return max(0.0, self._ends_at - self._clock())

    def flush(self) -> None:
        self._ends_at = self._clock()


class WavInput:
    """A WAV file heard as if it were a live microphone: 20 ms chunks on the wall clock.

    After the recording, ``tail`` seconds of silence keep the stream going so
    the turn can end and the reply can be heard.
    """

    def __init__(
        self, path: str | Path, *, tail: float = 15.0, loop: bool = False
    ) -> None:
        self.path = Path(path)
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError(f"{path}: expected 16-bit PCM")
            channels = wav.getnchannels()
            self.sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if channels > 1:
            values = _samples(frames)
            mono = array.array(
                "h",
                (
                    sum(values[i : i + channels]) // channels
                    for i in range(0, len(values), channels)
                ),
            )
            frames = _pcm(mono)
        self.pcm = frames
        self.tail = tail
        self.loop = loop
        self.finished = asyncio.Event()

    @property
    def duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate

    async def frames(self) -> AsyncIterator[bytes]:
        step = int(self.sample_rate * FRAME_SECONDS) * 2
        silence = b"\x00" * step
        start = time.monotonic()
        index = 0
        while True:
            for offset in range(0, len(self.pcm), step):
                await asyncio.sleep(
                    max(0.0, start + index * FRAME_SECONDS - time.monotonic())
                )
                chunk = self.pcm[offset : offset + step]
                yield chunk + silence[len(chunk) :]
                index += 1
            if not self.loop:
                break
        self.finished.set()
        for _ in range(int(self.tail / FRAME_SECONDS)):
            await asyncio.sleep(
                max(0.0, start + index * FRAME_SECONDS - time.monotonic())
            )
            yield silence
            index += 1


class WavOutput:
    """The character's voice, written to a WAV file as it would have played.

    Gaps between replies are kept as silence, so the file lines up with the
    conversation; audio dropped by a barge-in is cut where the cut happened.
    """

    def __init__(
        self, path: str | Path, *, sample_rate: int = OUTPUT_SAMPLE_RATE
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._clock = PlayoutClock(sample_rate)
        self._chunks: list[tuple[float, bytes]] = []
        self._start = time.monotonic()

    def play(self, pcm: bytes) -> None:
        start = max(time.monotonic(), time.monotonic() + self._clock.buffered_seconds())
        self._chunks.append((start - self._start, pcm))
        self._clock.push(len(pcm) // 2)

    def flush(self) -> None:
        cut = time.monotonic() - self._start
        kept: list[tuple[float, bytes]] = []
        for at, pcm in self._chunks:
            if at >= cut:
                continue
            playable = int((cut - at) * self.sample_rate) * 2
            kept.append((at, pcm[:playable]))
        self._chunks = kept
        self._clock.flush()

    def buffered_seconds(self) -> float:
        return self._clock.buffered_seconds()

    def close(self) -> None:
        end = max(
            (at + len(pcm) / 2 / self.sample_rate for at, pcm in self._chunks),
            default=0.0,
        )
        frames = bytearray(int(end * self.sample_rate) * 2)
        for at, pcm in self._chunks:
            offset = int(at * self.sample_rate) * 2
            frames[offset : offset + len(pcm)] = pcm
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(bytes(frames))


Echo = Literal["auto", "gate", "off"]


class LocalAudio:
    """This computer's microphone and speaker, through PortAudio (``sounddevice``).

    ``echo``:
      - ``"auto"``: while the character is audible, the microphone passes only
        sound clearly louder than the character's own voice would be at the
        microphone, so you can still cut in by speaking up.
      - ``"gate"``: the microphone is silent while the character speaks
        (half-duplex; no barge-in, never hears itself).
      - ``"off"``: everything passes. Use with headphones.
    """

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        echo: Echo = "auto",
        volume: float = 1.0,
    ) -> None:
        try:
            import sounddevice
        except (ImportError, OSError) as error:
            raise AudioDeviceError(
                'Local audio needs sounddevice and PortAudio: pip install "eesi-sdk[audio]" '
                f"(and on Linux, apt install libportaudio2). ({error})"
            ) from error
        self._sd = sounddevice
        self.input_device = input_device
        self.output_device = output_device
        self.echo = echo
        self.volume = max(0.0, min(1.0, volume))
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.input_rate = INPUT_SAMPLE_RATE
        self._out_lock = threading.Lock()
        self._out_buffer = bytearray()
        self._out_stream: Any = None
        self._in_stream: Any = None
        self._device_out_rate = OUTPUT_SAMPLE_RATE
        self._device_in_rate = INPUT_SAMPLE_RATE
        self._out_resampler: Resampler | None = None
        self._in_resampler: Resampler | None = None
        self._levels: deque[tuple[float, float]] = deque(maxlen=200)
        self._ducked = False
        self._muted = False

    # ── output ───────────────────────────────────────────────────────────

    def _open_output(self) -> None:
        if self._out_stream is not None:
            return
        frame = int(OUTPUT_SAMPLE_RATE * FRAME_SECONDS)
        try:
            self._out_stream = self._sd.RawOutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=frame,
                device=self.output_device,
                callback=self._fill,
                latency="low",
            )
        except Exception:
            info = self._sd.query_devices(self.output_device, "output")
            self._device_out_rate = int(info["default_samplerate"])
            self._out_resampler = Resampler(OUTPUT_SAMPLE_RATE, self._device_out_rate)
            try:
                self._out_stream = self._sd.RawOutputStream(
                    samplerate=self._device_out_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=int(self._device_out_rate * FRAME_SECONDS),
                    device=self.output_device,
                    callback=self._fill,
                    latency="low",
                )
            except Exception as error:
                raise AudioDeviceError(f"Cannot open the speaker: {error}") from error
        self._out_stream.start()

    def _fill(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None:
        wanted = frames * 2
        with self._out_lock:
            chunk = bytes(self._out_buffer[:wanted])
            del self._out_buffer[:wanted]
        if self.volume < 1.0 or self._ducked:
            gain = self.volume * (0.25 if self._ducked else 1.0)
            values = _samples(chunk)
            chunk = _pcm(array.array("h", (int(v * gain) for v in values)))
        outdata[: len(chunk)] = chunk
        if len(chunk) < wanted:
            outdata[len(chunk) :] = b"\x00" * (wanted - len(chunk))

    def play(self, pcm: bytes) -> None:
        self._open_output()
        if self._out_resampler is not None:
            pcm = self._out_resampler.process(pcm)
        now = time.monotonic()
        level = rms(pcm)
        with self._out_lock:
            ahead = len(self._out_buffer) / 2 / self._device_out_rate
            self._out_buffer.extend(pcm)
        self._levels.append((now + ahead, level))

    def flush(self) -> None:
        with self._out_lock:
            self._out_buffer.clear()
        if self._out_resampler is not None:
            self._out_resampler.reset()
        self._levels.clear()
        self._ducked = False

    def duck(self, ducked: bool = True) -> None:
        """Lower the character's voice while the floor is being decided."""
        self._ducked = ducked

    def buffered_seconds(self) -> float:
        with self._out_lock:
            return len(self._out_buffer) / 2 / self._device_out_rate

    def _output_level(self) -> float:
        """The character's level around now (what the microphone may be hearing)."""
        now = time.monotonic()
        recent = [level for at, level in self._levels if now - 0.3 <= at <= now + 0.05]
        return max(recent, default=0.0)

    # ── input ────────────────────────────────────────────────────────────

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    async def frames(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        def captured(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            data = bytes(indata)
            try:
                loop.call_soon_threadsafe(queue.put_nowait, data)
            except (RuntimeError, asyncio.QueueFull):
                pass

        frame = int(INPUT_SAMPLE_RATE * FRAME_SECONDS)
        try:
            self._in_stream = self._sd.RawInputStream(
                samplerate=INPUT_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=frame,
                device=self.input_device,
                callback=captured,
                latency="low",
            )
        except Exception:
            info = self._sd.query_devices(self.input_device, "input")
            self._device_in_rate = int(info["default_samplerate"])
            self._in_resampler = Resampler(self._device_in_rate, INPUT_SAMPLE_RATE)
            try:
                self._in_stream = self._sd.RawInputStream(
                    samplerate=self._device_in_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=int(self._device_in_rate * FRAME_SECONDS),
                    device=self.input_device,
                    callback=captured,
                    latency="low",
                )
            except Exception as error:
                raise AudioDeviceError(
                    f"Cannot open the microphone: {error}"
                ) from error
        self._in_stream.start()
        try:
            while True:
                data = await queue.get()
                if self._in_resampler is not None:
                    data = self._in_resampler.process(data)
                yield self._guard(data)
        finally:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None

    def _guard(self, pcm: bytes) -> bytes:
        """The microphone frame the session may hear, given what the speaker is playing."""
        if self._muted:
            return b"\x00" * len(pcm)
        if self.echo == "off" or self.buffered_seconds() <= 0.0:
            return pcm
        if self.echo == "gate":
            return b"\x00" * len(pcm)
        played = self._output_level()
        heard = rms(pcm)
        # Speaker-to-microphone coupling on a laptop is rarely above half the
        # played level; a person speaking up is well above it.
        return pcm if heard > max(0.02, 0.9 * played) else b"\x00" * len(pcm)

    def close(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._in_stream = None
        self._out_stream = None


def local(
    *,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    echo: Echo = "auto",
    volume: float = 1.0,
) -> LocalAudio:
    """This computer's microphone and speaker."""
    return LocalAudio(
        input_device=input_device, output_device=output_device, echo=echo, volume=volume
    )
