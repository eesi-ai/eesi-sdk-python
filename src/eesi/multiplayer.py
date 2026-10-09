"""Several players, one character.

Two shapes:

- **One session per player** (separate machines). Each player's session is a
  separate conversation, so without help the character forgets what it told
  the others. With ``party=True`` the character keeps a :class:`PartyLog` of
  who said what to whom, shared by its sessions as the ``party`` block.
- **One microphone, several people** (a couch, a table). Nur Live tags each
  line ``Speaker N:`` once it has heard a second voice. Pass ``players=`` to
  ``connect`` and a :class:`SpeakerRoster` binds those numbers to names, from
  introductions ("I'm Layla") and the order voices were first heard. It is a
  best effort that inherits every diarization mistake; transcripts carry the
  raw speaker number beside the guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ._protocol import split_speaker_turns

__all__ = ["AttributedTurn", "PartyLog", "Player", "SpeakerRoster", "as_players"]


@dataclass(frozen=True)
class Player:
    """Someone the character talks with.

    ``id`` is your own player id: the key of their memories (their name when
    left out). ``language`` is what they speak, as a code (``"ar"``, ``"es-MX"``)
    or a name (``"Egyptian Arabic"``); with players speaking different
    languages, the character answers each in their own.
    """

    name: str
    id: str | None = None
    language: str | None = None

    @property
    def key(self) -> str:
        return self.id or self.name


def as_players(players: Any) -> list[Player]:
    """Players from names, :class:`Player` objects, or a ``{name: id}`` mapping."""
    if not players:
        return []
    if isinstance(players, dict):
        return [Player(name=str(name), id=str(pid)) for name, pid in players.items()]
    made: list[Player] = []
    for item in players:
        made.append(item if isinstance(item, Player) else Player(name=str(item)))
    return made


class PartyLog:
    """What the character said to whom, and what each player said, newest last."""

    def __init__(self, *, max_lines: int = 12, max_chars: int = 2_000) -> None:
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._lines: list[tuple[str | None, str]] = []

    def add(self, line: str, *, origin: str | None = None) -> None:
        """Log *line*; *origin* is the session it happened in, which already knows it."""
        clean = " ".join(line.split())
        if not clean:
            return
        self._lines.append((origin, clean))
        if len(self._lines) > self.max_lines:
            del self._lines[: len(self._lines) - self.max_lines]

    def heard(self, player: str, text: str, *, origin: str | None = None) -> None:
        self.add(f"{player}: {text}", origin=origin)

    def said(
        self, character: str, player: str, text: str, *, origin: str | None = None
    ) -> None:
        self.add(f"{character} to {player}: {text}", origin=origin)

    def render(self, *, exclude: str | None = None) -> str:
        """The ``party`` block for a session: what happened in the others."""
        kept: list[str] = []
        size = 0
        for origin, line in reversed(self._lines):
            if exclude is not None and origin == exclude:
                continue
            size += len(line) + 1
            if size > self.max_chars:
                break
            kept.insert(0, line)
        if not kept:
            return ""
        return (
            "Your recent conversations with the other players, oldest first. Keep what you tell each player "
            "consistent with this, and pass news between them when it helps:\n"
            + "\n".join(kept)
        )


@dataclass(frozen=True)
class AttributedTurn:
    text: str
    speaker: int | None
    player: str | None


_INTRODUCTION = re.compile(
    r"\b(?:i['’]?m|i am|my name(?:['’]s| is)|this is|it['’]?s|call me)\s+([^\W\d_][\w'’-]+)",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"^\s*(?:(?:hey|hi|okay|ok|so|and)\s+)?([^\W\d_][\w'’-]+)\s*[,!?.:]", re.IGNORECASE
)


def _edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        diagonal, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            above = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] != b[j - 1]))
            diagonal = above
    return row[len(b)]


class SpeakerRoster:
    """``Speaker N`` labels bound to the players sharing one microphone."""

    def __init__(self, players: list[str] | None = None) -> None:
        self._by_speaker: dict[int, str] = {}
        self._expected: list[str] = []
        self._waiting: list[str] = []
        self._hello_order: list[str] = []
        for player in players or []:
            self.expect(player)

    def expect(self, player: str) -> None:
        if player not in self._expected:
            self._expected.append(player)
        if player not in self._waiting and player not in self._by_speaker.values():
            self._waiting.append(player)

    def assign(self, speaker: int, player: str) -> None:
        was = self._by_speaker.get(speaker)
        for key in [k for k, v in self._by_speaker.items() if v == player]:
            del self._by_speaker[key]
        self._by_speaker[speaker] = player
        self._waiting = [name for name in self._waiting if name != player]
        if was and was != player and was not in self._waiting:
            self._waiting.append(was)

    def players(self) -> dict[int, str]:
        return dict(self._by_speaker)

    def attribute(self, transcript: str) -> list[AttributedTurn]:
        turns: list[AttributedTurn] = []
        for speaker, text in split_speaker_turns(transcript):
            hello = self._named(text, _INTRODUCTION)
            if hello and speaker is None and hello not in self._hello_order:
                self._hello_order.append(hello)
            if speaker is not None:
                if hello:
                    if self._by_speaker.get(speaker) != hello:
                        self.assign(speaker, hello)
                elif speaker not in self._by_speaker and self._waiting:
                    to = self._named(text, _ADDRESS, first=True)
                    by_hello = (
                        self._hello_order[speaker - 1]
                        if 0 < speaker <= len(self._hello_order)
                        else None
                    )
                    if by_hello and by_hello != to and by_hello in self._waiting:
                        pick = by_hello
                    else:
                        pick = next(
                            (name for name in self._waiting if name != to),
                            self._waiting[0],
                        )
                    self.assign(speaker, pick)
            player = self._solo() if speaker is None else self._by_speaker.get(speaker)
            turns.append(AttributedTurn(text=text, speaker=speaker, player=player))
        return turns

    def render(self, languages: dict[str, str] | None = None) -> str:
        """The ``roster`` block. *languages* maps a player's name to what they speak."""
        if not self._by_speaker:
            return ""
        languages = languages or {}

        def who(name: str) -> str:
            spoken = languages.get(name)
            return f"{name} (speaks {spoken})" if spoken else name

        known = " ".join(
            f"Speaker {n} is {who(name)}."
            for n, name in sorted(self._by_speaker.items())
        )
        several = (
            len({languages.get(name) for name in self._by_speaker.values()} - {None})
            > 1
        )
        return (
            f"Several players share this conversation. {known} Answer the person who spoke to you, by name when "
            "it would otherwise be unclear. When two people ask different things, answer both, each by name. "
            + ("Answer each person in their own language. " if several else "")
            + "If you cannot tell who said something, ask."
        )

    def _solo(self) -> str | None:
        if len(self._by_speaker) == 1:
            return next(iter(self._by_speaker.values()))
        if not self._by_speaker and len(self._expected) == 1:
            return self._expected[0]
        return None

    def _named(
        self, text: str, pattern: re.Pattern[str], first: bool = False
    ) -> str | None:
        matches = (
            [m for m in [pattern.search(text)] if m]
            if first
            else list(pattern.finditer(text))
        )
        for match in matches:
            said = match.group(1).lower()
            scored = sorted(
                ((name, _edit_distance(name.lower(), said)) for name in self._expected),
                key=lambda p: p[1],
            )
            if not scored:
                continue
            best_name, best = scored[0]
            allowed = 2 if len(best_name) >= 5 else 1 if len(best_name) == 4 else 0
            if best <= allowed and (len(scored) == 1 or scored[1][1] > best):
                return best_name
        return None
