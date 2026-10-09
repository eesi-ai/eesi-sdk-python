"""Several players, one character, one conversation.

When your server already receives every player's microphone (your game's
voice chat runs through it), a room mixes them into one session with the
character. Everyone hears the same character and the character hears
everyone, so it can talk to the group, answer two people at once and greet
whoever walks up.

Who said what comes from the audio itself. The room knows which player was
loudest while each turn was spoken, and binds Nur Live's ``Speaker N``
labels to players by that, so it does not depend on people introducing
themselves::

    room = kael.room([Player("Ana", "p1", language="ar"), Player("Ben", "p2")])
    async with room:
        room.on("audio.output.chunk", lambda e: game.voice_to_everyone(e.pcm))
        room.on("speech.interrupted", lambda e: game.stop_voice_for_everyone())
        ...
        room.feed("p1", pcm)  # each player's microphone, as it arrives

With players speaking different languages, the character answers each in
their own (give each :class:`Player` a ``language``).
"""

from __future__ import annotations

import array
import asyncio
import math
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from ._protocol import INPUT_SAMPLE_RATE, split_speaker_turns
from .audio import Resampler
from .events import Event
from .multiplayer import AttributedTurn, Player, as_players

if TYPE_CHECKING:
    from .character import Character
    from .session import Session

__all__ = ["Room"]

FRAME_MS = 20
_FRAME_SAMPLES = INPUT_SAMPLE_RATE * FRAME_MS // 1000
_FRAME_BYTES = _FRAME_SAMPLES * 2
#: A player's level (RMS, 0..1) below this does not count toward who spoke.
SPEECH_FLOOR = 0.01
#: The most audio buffered per player; older audio is dropped, never delayed.
MAX_BUFFER_S = 1.0
#: How much level history attribution can look back on.
TIMELINE_S = 300


def _rms(samples: array.array) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768.0


class Room:
    """A character and a group of players in one conversation. Make one with :meth:`Character.room`."""

    def __init__(
        self,
        character: Character,
        players: Any,
        *,
        greet_joiners: bool = True,
        **connect: Any,
    ) -> None:
        self.character = character
        self.players: list[Player] = as_players(players)
        self.greet_joiners = greet_joiners
        self.session: Session = character.connect(players=self.players, **connect)
        self.session._attributor = self._attribute
        self._buffers: dict[str, bytearray] = {
            player.key: bytearray() for player in self.players
        }
        self._resamplers: dict[tuple[str, int], Resampler] = {}
        self._timeline: deque[tuple[int, dict[str, float]]] = deque(
            maxlen=TIMELINE_S * 1000 // FRAME_MS
        )
        self._frame = 0
        self._bound: dict[int, str] = {}
        self._task: asyncio.Task[None] | None = None
        self._last_speaker: str | None = None
        self.session.on("reconnected", self._restart_timeline)
        self.session.on("transcript.final", self._after_turn)

    def __repr__(self) -> str:
        return (
            f"Room({self.character.name!r}, players={[p.name for p in self.players]!r})"
        )

    async def __aenter__(self) -> Room:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        await self.session.start()
        self._task = asyncio.create_task(self._mix(), name="eesi-room-mixer")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self.session.close()

    async def run(self) -> None:
        await self.session.run()

    def on(self, kind: Any = "*", handler: Any = None) -> Any:
        return self.session.on(kind, handler)

    # ── players ──────────────────────────────────────────────────────────

    def feed(
        self, player_id: str, pcm: bytes, *, sample_rate: int = INPUT_SAMPLE_RATE
    ) -> None:
        """A player's microphone audio (16-bit mono PCM), as it arrives. Never blocks."""
        buffer = self._buffers.get(player_id)
        if buffer is None:
            return  # Not in the room.
        if sample_rate != INPUT_SAMPLE_RATE:
            resampler = self._resamplers.setdefault(
                (player_id, sample_rate), Resampler(sample_rate, INPUT_SAMPLE_RATE)
            )
            pcm = resampler.process(pcm)
        buffer.extend(pcm)
        limit = int(MAX_BUFFER_S * INPUT_SAMPLE_RATE) * 2
        if len(buffer) > limit:
            del buffer[: len(buffer) - limit]

    async def add_player(self, player: Player | str) -> None:
        """Someone joined. The character greets them unless ``greet_joiners`` is off."""
        made = player if isinstance(player, Player) else Player(name=str(player))
        if made.key in self._buffers:
            return
        self.players.append(made)
        self._buffers[made.key] = bytearray()
        session = self.session
        session.players.append(made)
        session._player_ids[made.name] = made.key
        if made.language:
            from .languages import describe

            session._languages[made.name] = describe(made.language)[1]
        if session._roster is not None:
            session._roster.expect(made.name)
        if self.greet_joiners and session.ready:
            await session.cue(f"{made.name} just joined the group.")

    def remove_player(self, player_id: str) -> None:
        self._buffers.pop(player_id, None)
        self.players = [player for player in self.players if player.key != player_id]
        for speaker, key in list(self._bound.items()):
            if key == player_id:
                del self._bound[speaker]

    # ── mixing ───────────────────────────────────────────────────────────

    async def _mix(self) -> None:
        """Every 20 ms: one frame from each player, summed, sent; levels kept for attribution."""
        started = time.monotonic()
        tick = 0
        while True:
            tick += 1
            await asyncio.sleep(
                max(0.0, started + tick * FRAME_MS / 1000 - time.monotonic())
            )
            if not self.session.ready:
                continue
            mixed = array.array("h", bytes(_FRAME_BYTES))
            levels: dict[str, float] = {}
            for key, buffer in self._buffers.items():
                chunk = bytes(buffer[:_FRAME_BYTES])
                del buffer[:_FRAME_BYTES]
                if not chunk:
                    continue
                samples = array.array("h")
                samples.frombytes(chunk[: len(chunk) - len(chunk) % 2])
                levels[key] = _rms(samples)
                for index, value in enumerate(samples):
                    mixed[index] = max(-32768, min(32767, mixed[index] + value))
            self._timeline.append((self._frame, levels))
            self._frame += 1
            try:
                await self.session.send_audio(mixed.tobytes())
            except Exception:
                return

    def render(self) -> str:
        """The ``roster`` block: who is here, what they speak, who spoke last."""
        if not self.players:
            return ""
        languages = self.session._languages

        def who(player: Player) -> str:
            spoken = languages.get(player.name)
            return f"{player.name} (speaks {spoken})" if spoken else player.name

        lines = [
            "You are in one conversation with a group: "
            + ", ".join(who(player) for player in self.players)
            + "."
        ]
        bound = [
            f"Speaker {speaker} is {self._name(key)}."
            for speaker, key in sorted(self._bound.items())
            if self._name(key)
        ]
        if bound:
            lines.append(" ".join(bound))
        if self._last_speaker:
            lines.append(f"{self._last_speaker} spoke last.")
        lines.append(
            "Answer the person who spoke to you, by name when it would otherwise be unclear. "
            "When two people ask different things, answer both, each by name."
        )
        if len(set(languages.values())) > 1:
            lines.append("Answer each person in their own language.")
        return " ".join(lines)

    def _after_turn(self, event: Event) -> None:
        turns = getattr(event, "turns", ())
        named = [
            self._name(turn.player_id) or turn.player_id
            for turn in turns
            if turn.player_id
        ]
        if named:
            self._last_speaker = named[-1]
        self.session._publisher.set_block("roster", self.render())

    def _restart_timeline(self, _event: Event) -> None:
        # A resumed conversation counts its audio from zero again.
        self._frame = 0
        self._timeline.clear()

    # ── who said what ────────────────────────────────────────────────────

    def loudest(self, start_ms: int | None, end_ms: int | None) -> list[str]:
        """Players by how much they spoke between two points of the audio sent, loudest first."""
        first = 0 if start_ms is None else start_ms // FRAME_MS
        last = self._frame if end_ms is None else max(first, end_ms // FRAME_MS)
        totals: dict[str, float] = {}
        for frame, levels in self._timeline:
            if first <= frame <= last:
                for key, level in levels.items():
                    if level >= SPEECH_FLOOR:
                        totals[key] = totals.get(key, 0.0) + level
        return [
            key
            for key, _ in sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
        ]

    def _name(self, key: str | None) -> str | None:
        for player in self.players:
            if player.key == key:
                return player.name
        return None

    def _attribute(self, text: str) -> list[AttributedTurn]:
        ranking = self.loudest(*self.session.speech_window)
        roster = self.session._roster
        turns: list[AttributedTurn] = []
        for speaker, words in split_speaker_turns(text):
            if speaker is None:
                key = ranking[0] if ranking else None
            else:
                key = self._bound.get(speaker)
                if key is None:
                    taken = set(self._bound.values())
                    free = [k for k in ranking if k not in taken]
                    key = (free or ranking or [None])[0]
                    if key is not None:
                        self._bound[speaker] = key
                        name = self._name(key)
                        if roster is not None and name:
                            roster.assign(speaker, name)
            turns.append(
                AttributedTurn(text=words, speaker=speaker, player=self._name(key))
            )
        return turns
