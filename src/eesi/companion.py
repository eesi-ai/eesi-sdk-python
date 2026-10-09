"""Nur as a coding companion: it watches your coding agent and calls you.

Claude Code or Codex works on your code; when it finishes (or after a long
run), Nur phones you with what was done, answers your questions about it, and
hands what you ask for next back to the same agent session, which calls you
again when that is done::

    eesi companion setup --phone +14155550123 --name Amr --api-key sk-eesi-...
    eesi companion confirm 482917            # the code Nur phoned in
    eesi companion hook                      # as a Claude Code Stop hook
    eesi companion codex-notify              # as Codex's notify program
    eesi companion call                      # an update about this repo, now
    eesi companion listen                    # the code line: call Nur to get work done

The hook returns at once and the call runs in the background, so the agent
never waits. What Nur knows on the call is a briefing written from the
session's transcript (what you asked, what the agent said it did, the files it
wrote, the commands it ran) and the repository's numbers (lines added and
removed, files, commits today, languages). After you hang up, what you asked
for is read off the call's transcript and run as a follow-up in the same
session; when it ends, Nur calls back with what it did, or why it could not.

Settings come from the environment or ``~/.eesi/companion.json`` (written by
``setup``, readable by you only): ``phone`` (E.164), ``name`` (what Nur calls
you), ``api_key`` and ``api_url`` (``EESI_API_KEY`` / ``EESI_API_URL`` win),
``call_on`` (``stop``: after every turn; ``long``: after a turn of at least
``min_minutes``, the default; ``never``), ``follow_up`` (run what you ask for,
default true), ``follow_up_minutes`` (how long a follow-up may run, default
60) and ``claude_args`` / ``codex_args`` (flags for follow-up runs).

On the shared EESI line, Nur calls only numbers confirmed with a code it
phoned in (``setup`` asks for one, ``confirm`` enters it).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

HOME = Path(os.environ.get("NUR_COMPANION_HOME") or Path.home() / ".eesi" / "companion")
#: How the call starts in the background, and how a follow-up runs to its end;
#: tests replace both.
_launch = subprocess.Popen
_run = subprocess.run
_TEXT_SUFFIXES = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
    ".jsx": "JavaScript", ".mjs": "JavaScript", ".go": "Go", ".rs": "Rust", ".java": "Java",
    ".kt": "Kotlin", ".swift": "Swift", ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++",
    ".c": "C", ".h": "C", ".css": "CSS", ".html": "HTML", ".sql": "SQL", ".sh": "Shell",
    ".md": "Markdown", ".mdx": "Markdown", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
}  # fmt: skip


# ── settings ──────────────────────────────────────────────────────────────


@dataclass
class Settings:
    phone: str | None = None
    name: str | None = None
    call_on: str = "long"
    min_minutes: float = 3.0
    follow_up: bool = True
    follow_up_minutes: float = 60.0
    claude_args: str = "--permission-mode acceptEdits"
    codex_args: str = "--sandbox workspace-write -c approval_policy='\"never\"'"
    voice: str | None = None
    api_key: str | None = None
    api_url: str | None = None
    #: How the listener tells you a task is done: ``call``, ``text`` or ``both``.
    notify: str = "call"

    @staticmethod
    def path() -> Path:
        return HOME.parent / "companion.json"

    @classmethod
    def file(cls) -> dict[str, Any]:
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def load(cls) -> Settings:
        data = cls.file()
        env = os.environ.get
        return cls(
            phone=env("NUR_COMPANION_PHONE") or data.get("phone"),
            name=env("NUR_COMPANION_NAME") or data.get("name"),
            call_on=env("NUR_COMPANION_CALL_ON") or data.get("call_on") or "long",
            min_minutes=float(
                env("NUR_COMPANION_MIN_MINUTES") or data.get("min_minutes") or 3
            ),
            follow_up=(
                env("NUR_COMPANION_FOLLOW_UP") or str(data.get("follow_up", "true"))
            ).lower()
            not in ("0", "false", "no", "off"),
            claude_args=env("NUR_COMPANION_CLAUDE_ARGS")
            or data.get("claude_args")
            or cls.claude_args,
            codex_args=env("NUR_COMPANION_CODEX_ARGS")
            or data.get("codex_args")
            or cls.codex_args,
            voice=env("NUR_COMPANION_VOICE") or data.get("voice"),
            follow_up_minutes=float(
                env("NUR_COMPANION_FOLLOW_UP_MINUTES")
                or data.get("follow_up_minutes")
                or 60
            ),
            api_key=env("EESI_API_KEY") or data.get("api_key"),
            api_url=env("EESI_API_URL") or data.get("api_url"),
            notify=env("NUR_COMPANION_NOTIFY") or data.get("notify") or "call",
        )

    @classmethod
    def save(cls, changes: dict[str, Any]) -> Path:
        """Merge *changes* into the settings file, readable by this user only: it holds the key."""
        data = cls.file()
        data.update({key: value for key, value in changes.items() if value is not None})
        path = cls.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        os.chmod(path, 0o600)
        return path


# ── what happened in the session ─────────────────────────────────────────


@dataclass
class Digest:
    """One agent session, as much of it as a phone call needs."""

    agent: str  # "Claude Code" or "Codex"
    session_id: str | None
    cwd: str
    asked: list[str] = field(default_factory=list)
    answer: str = ""
    files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: int = 0
    turn_started: float | None = None
    #: On a call back: what the developer asked for on the last call.
    requested: list[str] = field(default_factory=list)
    #: On a call back: why that could not be done ("" when it was).
    failure: str = ""

    @property
    def turn_minutes(self) -> float | None:
        return (time.time() - self.turn_started) / 60 if self.turn_started else None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _when(stamp: Any) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_claude_session(
    transcript_path: str, *, session_id: str | None, cwd: str
) -> Digest:
    """A Claude Code session from its transcript (JSONL, one event a line)."""
    digest = Digest(agent="Claude Code", session_id=session_id, cwd=cwd)
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return digest
    for raw in lines:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content")
        if event.get("type") == "user" and not event.get("isMeta"):
            if isinstance(content, str) or (
                isinstance(content, list)
                and any(
                    isinstance(b, dict) and b.get("type") == "text" for b in content
                )
            ):
                text = _text_of(content).strip()
                if text and not text.startswith("<"):
                    digest.asked.append(text)
                    digest.turn_started = (
                        _when(event.get("timestamp")) or digest.turn_started
                    )
            if isinstance(content, list):
                digest.errors += sum(
                    1
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("is_error")
                )
        elif event.get("type") == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and str(block.get("text") or "").strip():
                    digest.answer = str(block["text"]).strip()
                elif block.get("type") == "tool_use":
                    tool_input = (
                        block.get("input")
                        if isinstance(block.get("input"), dict)
                        else {}
                    )
                    path = tool_input.get("file_path") or tool_input.get(
                        "notebook_path"
                    )
                    if (
                        block.get("name")
                        in ("Write", "Edit", "MultiEdit", "NotebookEdit")
                        and path
                    ):
                        relative = (
                            os.path.relpath(str(path), cwd)
                            if os.path.isabs(str(path))
                            else str(path)
                        )
                        if relative not in digest.files:
                            digest.files.append(relative)
                    elif block.get("name") == "Bash" and tool_input.get("command"):
                        digest.commands.append(
                            str(tool_input["command"]).splitlines()[0][:160]
                        )
                    elif (
                        block.get("name") == "SubagentHandback"
                        and str(tool_input.get("message") or "").strip()
                    ):
                        # A session started by another hands its final report
                        # back as a tool call, not as text.
                        digest.answer = str(tool_input["message"]).strip()
    return digest


def read_codex_turn(payload: dict[str, Any], *, cwd: str) -> Digest:
    """A Codex turn from its notify payload."""
    asked = payload.get("input-messages") or payload.get("input_messages") or []
    return Digest(
        agent="Codex",
        session_id=payload.get("thread-id")
        or payload.get("thread_id")
        or payload.get("session-id"),
        cwd=payload.get("cwd") or cwd,
        asked=[str(text) for text in asked if str(text).strip()],
        answer=str(
            payload.get("last-assistant-message")
            or payload.get("last_assistant_message")
            or ""
        ),
    )


# ── the repository's numbers ─────────────────────────────────────────────


def _git(cwd: str, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


@dataclass
class CodeStats:
    repo: str
    branch: str = ""
    added: int = 0
    removed: int = 0
    changed_files: int = 0
    new_files: int = 0
    commits_today: int = 0
    tracked_files: int = 0
    languages: list[tuple[str, int]] = field(default_factory=list)


def code_stats(cwd: str) -> CodeStats:
    """Lines added and removed since the last commit, files, commits today, languages by lines."""
    top = _git(cwd, "rev-parse", "--show-toplevel").strip() or cwd
    stats = CodeStats(
        repo=Path(top).name, branch=_git(cwd, "branch", "--show-current").strip()
    )
    for line in _git(cwd, "diff", "HEAD", "--numstat").splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            stats.changed_files += 1
            stats.added += int(parts[0]) if parts[0].isdigit() else 0
            stats.removed += int(parts[1]) if parts[1].isdigit() else 0
    for name in _git(cwd, "ls-files", "--others", "--exclude-standard").splitlines():
        stats.new_files += 1
        try:
            stats.added += (
                Path(top, name).read_text(encoding="utf-8", errors="ignore").count("\n")
            )
        except OSError:
            pass
    stats.commits_today = len(
        [
            line
            for line in _git(cwd, "log", "--since=midnight", "--oneline").splitlines()
            if line.strip()
        ]
    )
    counts: dict[str, int] = {}
    tracked = _git(cwd, "ls-files").splitlines()
    stats.tracked_files = len(tracked)
    untracked = _git(cwd, "ls-files", "--others", "--exclude-standard").splitlines()
    for name in (tracked + untracked)[:4000]:
        language = _TEXT_SUFFIXES.get(Path(name).suffix.lower())
        if not language:
            continue
        try:
            counts[language] = counts.get(language, 0) + Path(top, name).read_text(
                encoding="utf-8", errors="ignore"
            ).count("\n")
        except OSError:
            continue
    stats.languages = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return stats


# ── the call ─────────────────────────────────────────────────────────────


#: The directories in front of a file name, which nobody wants read aloud.
_DIRECTORIES = re.compile(
    r"(?:~|\.{1,2})?(?:/[\w.@+-]+)+/"
    r"|(?<![\w/])(?:[\w.@+-]+/)+(?=[\w@+-]+\.[A-Za-z0-9]{1,8}\b)"
)


def _spoken(text: str) -> str:
    """Text as it should be said: file names without their folders, no markdown marks."""
    text = _DIRECTORIES.sub("", text)
    for mark in ("`", "**", "__", "#"):
        text = text.replace(mark, "")
    return " ".join(text.split())


def _headline(digest: Digest) -> str:
    text = _spoken(digest.answer)
    for stop in (". ", "! ", "? ", "\n"):
        if stop in text[:200]:
            text = text[: text.index(stop) + 1]
            break
    return text[:200].rstrip(" ,;:")


def briefing(digest: Digest, stats: CodeStats, *, name: str | None) -> tuple[str, str]:
    """Nur's instructions for the call, and its exact opening line."""
    person = name or "the developer"
    hello = f"Hi{' ' + name if name else ''}, it's Nur"
    languages = (
        ", ".join(f"{language} {lines:,} lines" for language, lines in stats.languages)
        or "unknown"
    )
    facts = [
        f"Agent: {digest.agent}, working in the repository {stats.repo}"
        + (f" on branch {stats.branch}" if stats.branch else "")
        + ".",
    ]
    if digest.requested:
        facts.append(
            f"What {person} asked for on the last call: " + " ".join(digest.requested)
        )
    else:
        asked = digest.asked[-1] if digest.asked else "(no request recorded)"
        facts.append(f"What {person} asked for last: {asked}")
        earlier = digest.asked[:-1][-3:]
        if earlier:
            facts.append("Earlier in the session they asked: " + " | ".join(earlier))
    if digest.failure:
        facts.append(f"It could not be done: {digest.failure}")
    else:
        facts.append(
            f"What the agent said when it finished: {digest.answer or '(nothing)'}"
        )
    if digest.files:
        facts.append("Files it wrote or edited: " + ", ".join(digest.files[:20]))
    if digest.commands:
        facts.append("Commands it ran: " + " | ".join(digest.commands[-8:]))
    if digest.errors:
        facts.append(f"{digest.errors} tool steps failed along the way.")
    facts.append(
        f"Code since the last commit: +{stats.added} / -{stats.removed} lines across "
        f"{stats.changed_files} changed and {stats.new_files} new files; {stats.commits_today} "
        f"commits today; {stats.tracked_files} tracked files; languages: {languages}."
    )
    if digest.turn_minutes is not None:
        minutes = round(digest.turn_minutes)
        facts.append(
            "This run took "
            + (
                "under a minute."
                if minutes < 1
                else f"about {minutes} minute{'s' if minutes != 1 else ''}."
            )
        )
    headline = _headline(digest)
    if digest.failure:
        why = (
            f"what they asked for on the last call could not be handed to {digest.agent}. "
            "Say why plainly, and that nothing was changed by it"
        )
        greeting = (
            f"{hello} again. I couldn't get {digest.agent} to do what you asked: "
            f"{_spoken(digest.failure)[:200].rstrip(' .')}."
        )
    elif digest.requested:
        why = f"{digest.agent} did what they asked for on the last call"
        greeting = (
            f"{hello} again. {digest.agent} did what you asked in {stats.repo}"
            + (f": {headline}" if headline else ".")
            + " Anything else?"
        )
    else:
        why = "their coding agent just finished"
        greeting = (
            f"{hello}. {digest.agent} just finished in {stats.repo}"
            + (f": {headline}" if headline else ".")
            + " Want me to walk you through it?"
        )
    instructions = (
        f"You are Nur, {person}'s coding companion. You are calling {person} on the phone "
        f"because {why}. Give them the update and help them decide what happens next.\n\n"
        "How to talk: this is a phone call. Speak naturally, a sentence or two at a time, no "
        "lists, no markdown, no file paths read character by character (say 'the hello script', "
        "not 'slash src slash hello dot p y'). Lead with the outcome, then offer details. "
        "Answer questions only from the facts below; when you don't know, say so. If they ask "
        "for more work, repeat back exactly what you will tell the agent and confirm it, so it "
        "can be handed over after the call. When they are done, or ask you to hang up, say "
        "goodbye briefly and end the call. If a voicemail greeting or an automated menu "
        "answers instead of them, leave a one-sentence update, say goodbye and end the call.\n\n"
        "The facts:\n- " + "\n- ".join(facts)
    )
    return instructions, greeting


FOLLOW_UP_SYSTEM = (
    "You read the transcript of a phone call between a developer and Nur, their coding "
    "companion, about work their coding agent just did. List the follow-up tasks the developer "
    "clearly asked to be handed to the agent, each as one direct instruction to the agent in "
    "the second person ('Add a test for the greet function.'). Only what they asked for or "
    "agreed to; nothing they declined; nothing you infer. Answer with JSON only: "
    '{"follow_ups": ["..."]}, an empty list when there is nothing.'
)


async def follow_ups(nur: Any, transcript: tuple[tuple[str, str], ...]) -> list[str]:
    """What the developer asked for on the call, as instructions for the agent."""
    if not any(role == "person" for role, _ in transcript):
        return []
    lines = "\n".join(
        f"{'Developer' if role == 'person' else 'Nur'}: {text}"
        for role, text in transcript
    )
    reply = await nur.chat(
        [
            {"role": "system", "content": FOLLOW_UP_SYSTEM},
            {"role": "user", "content": f"The call:\n{lines}\n\nThe JSON:"},
        ]
    )
    try:
        start = reply.index("{")
        data = json.loads(reply[start : reply.rindex("}") + 1])
    except ValueError:
        return []
    return [
        str(item).strip() for item in data.get("follow_ups") or [] if str(item).strip()
    ][:5]


@dataclass
class FollowUp:
    """What became of the work asked for on a call."""

    ok: bool
    #: What the agent said when it finished, or why it could not run.
    report: str
    session_id: str | None = None


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def run_follow_up(
    digest: Digest, instructions: list[str], settings: Settings
) -> FollowUp | None:
    """Run what was asked for on the call in the same agent session, and wait for it.

    The run is told not to call by itself (its own hook would): the companion
    calls back when it ends, with what it did or why it could not run.
    """
    if not instructions:
        return None
    prompt = "Follow-up from a phone call with the developer:\n" + "\n".join(
        f"- {line}" for line in instructions
    )
    HOME.mkdir(parents=True, exist_ok=True)
    report_file = HOME / f"follow-up-{os.getpid()}.txt"
    if digest.agent == "Codex":
        resume = (
            ["resume", digest.session_id] if digest.session_id else ["resume", "--last"]
        )
        command = [
            "codex", "exec", *shlex.split(settings.codex_args),
            "--output-last-message", str(report_file), *resume, prompt,
        ]  # fmt: skip
    else:
        resume = (
            ["--resume", digest.session_id] if digest.session_id else ["--continue"]
        )
        command = [
            "claude", "-p", *resume, "--output-format", "json",
            *shlex.split(settings.claude_args), prompt,
        ]  # fmt: skip
    try:
        done = _run(
            command,
            cwd=digest.cwd,
            env={**os.environ, "NUR_COMPANION_CALL_ON": "never"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=settings.follow_up_minutes * 60,
            check=False,
        )
    except FileNotFoundError:
        return FollowUp(
            False, f"the {command[0]} command is not installed on that computer"
        )
    except subprocess.TimeoutExpired:
        return FollowUp(
            False,
            f"it was still running after {settings.follow_up_minutes:.0f} minutes, so it was stopped",
        )
    with (HOME / "follow-ups.log").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "at": time.time(),
                    "command": command[:-1],
                    "returncode": done.returncode,
                    "stdout": (done.stdout or "")[-4000:],
                    "stderr": (done.stderr or "")[-4000:],
                }
            )
            + "\n"
        )
    failed = _last_line(done.stderr) or _last_line(done.stdout)
    if digest.agent == "Codex":
        try:
            report = report_file.read_text(encoding="utf-8").strip()
            report_file.unlink()
        except OSError:
            report = ""
        if done.returncode == 0 and report:
            return FollowUp(True, report, digest.session_id)
        return FollowUp(
            False, failed or f"Codex exited with {done.returncode}", digest.session_id
        )
    try:
        result = json.loads(done.stdout or "{}")
    except ValueError:
        result = {}
    result = result if isinstance(result, dict) else {}
    text = str(result.get("result") or "").strip()
    session_id = str(result.get("session_id") or "") or digest.session_id
    if done.returncode == 0 and text and not result.get("is_error"):
        return FollowUp(True, text, session_id)
    return FollowUp(
        False,
        (text if result.get("is_error") else "")
        or failed
        or f"Claude Code exited with {done.returncode}",
        session_id,
    )


def _record(event: dict[str, Any]) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    with (HOME / "calls.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps({"at": time.time(), **event}) + "\n")


#: Call backs after follow-ups, at most, before the companion stops on its own.
MAX_ROUNDS = 10


async def call_about(
    digest: Digest, settings: Settings | None = None, *, wait: bool = True
) -> dict[str, Any]:
    """Call the developer about *digest*, run what they ask for, and call back.

    Each call back says how the follow-up went (or why it could not run), and
    may bring the next request; it ends when a call asks for nothing more.
    """
    from .client import NurClient

    settings = settings or Settings.load()
    if not settings.phone:
        raise SystemExit("Run `eesi companion setup --phone +14155550123` first.")
    calls: list[dict[str, Any]] = []
    async with NurClient(settings.api_key, base_url=settings.api_url) as nur:
        for _ in range(MAX_ROUNDS):
            stats = code_stats(digest.cwd)
            instructions, greeting = briefing(digest, stats, name=settings.name)
            placed = await nur.call(
                settings.phone,
                instructions=instructions,
                greeting=greeting,
                voice=settings.voice,
                metadata={
                    "agent": digest.agent,
                    "repo": stats.repo,
                    "session": digest.session_id or "",
                },
            )
            _record(
                {
                    "call": placed.id,
                    "status": placed.status,
                    "agent": digest.agent,
                    "repo": stats.repo,
                }
            )
            if not wait:
                return {"call": placed.id}
            ended = await nur.wait_for_call(placed.id)
            asked = (
                await follow_ups(nur, ended.transcript) if settings.follow_up else []
            )
            outcome: dict[str, Any] = {
                "call": placed.id,
                "status": ended.status,
                "duration_seconds": ended.duration_seconds,
                "transcript": [list(turn) for turn in ended.transcript],
                "follow_ups": asked,
            }
            calls.append(outcome)
            done = await asyncio.to_thread(run_follow_up, digest, asked, settings)
            if done is None:
                _record(outcome)
                break
            outcome["follow_up"] = {"ok": done.ok, "report": done.report}
            _record(outcome)
            digest = Digest(
                agent=digest.agent,
                session_id=done.session_id or digest.session_id,
                cwd=digest.cwd,
                asked=[*digest.asked, *asked],
                answer=done.report if done.ok else "",
                requested=asked,
                failure="" if done.ok else done.report,
            )
    return {"calls": calls}


# ── entry points ─────────────────────────────────────────────────────────


def _detach(argv: list[str], payload: dict[str, Any]) -> None:
    """Do the call in a background process, so the agent never waits on it."""
    HOME.mkdir(parents=True, exist_ok=True)
    log = (HOME / "companion.log").open("ab")
    process = _launch(
        [sys.executable, "-m", "eesi", "companion", *argv],
        stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True,
    )  # fmt: skip
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload).encode())
    process.stdin.close()


def should_call(event: str, digest: Digest, settings: Settings) -> bool:
    if settings.call_on == "never":
        return False
    if event == "SessionEnd":
        return settings.call_on in ("long", "stop", "session_end")
    if settings.call_on == "stop":
        return True
    if settings.call_on == "long":
        minutes = digest.turn_minutes
        return minutes is not None and minutes >= settings.min_minutes
    return False


def claude_hook(stdin_text: str) -> int:
    """A Claude Code hook (Stop, SubagentStop or SessionEnd): decide, then call in the background."""
    try:
        event = json.loads(stdin_text or "{}")
    except ValueError:
        return 0
    if event.get("stop_hook_active"):
        return 0
    payload = {
        "agent": "claude",
        "event": event.get("hook_event_name") or "Stop",
        "session_id": event.get("session_id"),
        "transcript_path": event.get("transcript_path"),
        "cwd": event.get("cwd") or os.getcwd(),
    }
    digest = read_claude_session(
        payload["transcript_path"] or "",
        session_id=payload["session_id"],
        cwd=payload["cwd"],
    )
    if should_call(payload["event"], digest, Settings.load()):
        _detach(["run"], payload)
    return 0


def codex_notify(argument: str) -> int:
    """Codex's notify program: called with the event as JSON after each turn."""
    try:
        event = json.loads(argument or "{}")
    except ValueError:
        return 0
    if event.get("type") not in (None, "agent-turn-complete"):
        return 0
    payload = {
        "agent": "codex",
        "event": "Stop",
        "codex": event,
        "cwd": event.get("cwd") or os.getcwd(),
    }
    if Settings.load().call_on in ("stop", "long"):
        _detach(["run"], payload)
    return 0


def run_detached(stdin_text: str) -> int:
    payload = json.loads(stdin_text or "{}")
    cwd = payload.get("cwd") or os.getcwd()
    if payload.get("agent") == "codex":
        digest = read_codex_turn(payload.get("codex") or {}, cwd=cwd)
    elif payload.get("transcript_path"):
        digest = read_claude_session(
            payload["transcript_path"], session_id=payload.get("session_id"), cwd=cwd
        )
    else:
        digest = Digest(
            agent="your coding agent",
            session_id=None,
            cwd=cwd,
            answer="Here is where the code stands.",
        )
    outcome = asyncio.run(call_about(digest))
    print(json.dumps(outcome, ensure_ascii=False))
    return 0


def latest_claude_transcript(cwd: str) -> Path | None:
    """The newest Claude Code transcript for *cwd*: the session ``/call-me`` runs in."""
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    folder = root / "projects" / re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(cwd))
    try:
        return max(folder.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    except (ValueError, OSError):
        return None


def _option(argv: list[str], name: str) -> str | None:
    for index, argument in enumerate(argv):
        if argument == name and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith(name + "="):
            return argument[len(name) + 1 :]
    return None


_E164 = re.compile(r"^\+[1-9]\d{6,14}$")
CALL_ON = ("stop", "long", "session_end", "never")
NOTIFY = ("call", "text", "both")


def call_now(argv: list[str]) -> int:
    """Nur calls now about this repository and its newest Claude Code session."""
    settings = Settings.load()
    if not settings.phone:
        print("Run `eesi companion setup --phone +14155550123` first.")
        return 1
    cwd = _option(argv, "--cwd") or os.getcwd()
    transcript = _option(argv, "--transcript") or latest_claude_transcript(cwd)
    _detach(
        ["run"],
        {
            "agent": "claude",
            "event": "Call",
            "cwd": cwd,
            "transcript_path": str(transcript) if transcript else None,
            "session_id": Path(transcript).stem if transcript else None,
        },
    )
    print(f"Nur is calling {settings.phone} now.")
    return 0


async def _numbers(settings: Settings, action: str, code: str | None = None) -> Any:
    from .client import NurClient

    async with NurClient(settings.api_key, base_url=settings.api_url) as nur:
        if action == "send":
            return await nur.verify_phone_number(str(settings.phone))
        if action == "confirm":
            return await nur.confirm_phone_number(
                str(settings.phone), str(code), name=settings.name
            )
        return await nur.phone_numbers()


def setup(argv: list[str]) -> int:
    """Save who Nur calls and with which key; Nur phones the number a code to confirm it."""
    from .errors import ApiError

    phone = _option(argv, "--phone")
    if phone is not None:
        phone = re.sub(r"[\s().-]", "", phone)
        if not _E164.match(phone):
            print("Give the number in E.164 form, like +14155550123.")
            return 1
    call_on = _option(argv, "--call-on")
    if call_on is not None and call_on not in CALL_ON:
        print(f"--call-on is one of: {', '.join(CALL_ON)}.")
        return 1
    notify = _option(argv, "--notify")
    if notify is not None and notify not in NOTIFY:
        print(f"--notify is one of: {', '.join(NOTIFY)}.")
        return 1
    path = Settings.save(
        {
            "phone": phone,
            "name": _option(argv, "--name"),
            "api_key": _option(argv, "--api-key"),
            "api_url": _option(argv, "--api-url"),
            "call_on": call_on,
            "notify": notify,
        }
    )
    settings = Settings.load()
    print(f"Saved {path}.")
    if not settings.api_key:
        print(
            "Add your EESI API key (the API keys page of your EESI console): "
            "eesi companion setup --api-key sk-eesi-..."
        )
        return 1
    if not settings.phone:
        print("Add the number Nur calls: eesi companion setup --phone +14155550123")
        return 1
    try:
        number = asyncio.run(_numbers(settings, "send"))
    except ApiError as error:
        print(f"Nur could not call {settings.phone}: {error}")
        return 1
    if number.confirmed:
        print(
            f"{settings.phone} is confirmed: Nur calls it when your coding agent finishes."
        )
    else:
        print(
            f"Nur is calling {settings.phone} with a six-digit code. "
            "Then: eesi companion confirm <code>"
        )
    return 0


def confirm(argv: list[str]) -> int:
    """Enter the code Nur phoned in; from then on Nur may call the number."""
    from .errors import ApiError

    settings = Settings.load()
    code = re.sub(r"\D", "", " ".join(argv))
    if not settings.phone or len(code) != 6:
        print("Usage: eesi companion confirm 482917 (the code Nur phoned in).")
        return 1
    try:
        number = asyncio.run(_numbers(settings, "confirm", code))
    except ApiError as error:
        print(f"Not confirmed: {error}")
        return 1
    print(
        f"{number.number} is confirmed: Nur calls it when your coding agent finishes."
    )
    return 0


def status(argv: list[str]) -> int:
    """Where the settings are, what they say (the key shortened), and the confirmed numbers."""
    from .errors import ApiError, EESIError

    settings = Settings.load()
    key = settings.api_key
    report: dict[str, Any] = {
        "settings": str(Settings.path()),
        "phone": settings.phone,
        "name": settings.name,
        "call_on": settings.call_on,
        "follow_up": settings.follow_up,
        # Whether there is one, never any of it.
        "api_key": "set" if key else None,
        "api_url": settings.api_url or "https://api.eesi.ai",
    }
    if key:
        try:
            report["confirmed_numbers"] = [
                number.number for number in asyncio.run(_numbers(settings, "list"))
            ]
        except (ApiError, EESIError) as error:
            report["confirmed_numbers"] = f"could not ask: {error}"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "help"
    if command == "hook":
        return claude_hook(sys.stdin.read())
    if command == "codex-notify":
        return codex_notify(argv[1] if len(argv) > 1 else sys.stdin.read())
    if command == "run":
        return run_detached(sys.stdin.read())
    if command == "call":
        return call_now(argv[1:])
    if command == "setup":
        return setup(argv[1:])
    if command == "confirm":
        return confirm(argv[1:])
    if command == "status":
        return status(argv[1:])
    if command == "listen":
        from .listener import listen

        return listen(argv[1:])
    if command == "claude-hooks":
        print(json.dumps(CLAUDE_HOOKS, indent=2))
        return 0
    print(__doc__)
    return 0


#: What to add to Claude Code's settings (or what the plugin installs).
CLAUDE_HOOKS = {
    "hooks": {
        event: [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "eesi companion hook",
                        "timeout": 60,
                    }
                ]
            }
        ]
        for event in ("Stop", "SessionEnd")
    }
}
