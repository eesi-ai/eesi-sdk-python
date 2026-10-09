"""Nur's code line: call Nur about your code, and it gets the work done.

``eesi companion listen`` runs beside a repository::

    eesi companion listen            # Claude Code; --agent codex for Codex
    eesi companion listen --notify text   # done? a text instead of a call

Call the code line (the number it prints) from your confirmed phone and tell
Nur what to build. When you hang up, what you asked for is queued and run in
one coding session, task after task, each resuming the last. Call again any
time: Nur knows what is running (and its latest steps), what is queued and
what is done, and how the code moved. When a task finishes, Nur calls you
with the result; what you ask for on that call is queued too.

The listener keeps briefing the code line (``PUT /v1/phone/listener``) every
few seconds, and collects the calls that ended (``GET
/v1/phone/listener/events``), reading each one's transcript for the work it
asked for. Everything about the code stays on this computer until a brief
carries it to a call.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import companion
from .companion import Settings, _headline, _option, _spoken, code_stats, follow_ups

#: How often the brief is refreshed when nothing changes.
PUBLISH_SECONDS = 20
#: Steps of a running task kept for a call that asks how it is going.
ACTIVITY = 6


@dataclass
class Task:
    """One piece of work asked for on a call."""

    text: str
    asked_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    ok: bool | None = None
    report: str = ""
    activity: list[str] = field(default_factory=list)

    def step(self, line: str) -> None:
        line = " ".join(line.split())[:160]
        if line and (not self.activity or self.activity[-1] != line):
            self.activity = [*self.activity, line][-ACTIVITY:]


def _ago(seconds: float) -> str:
    minutes = round(seconds / 60)
    return (
        "under a minute"
        if minutes < 1
        else f"{minutes} minute{'s' if minutes != 1 else ''}"
    )


def _short(text: str, limit: int = 90) -> str:
    text = _spoken(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


class Listener:
    """The code line's view of this repository, and the worker that runs its tasks."""

    def __init__(
        self,
        settings: Settings,
        cwd: str,
        *,
        agent: str = "claude",
        echo: Callable[[str], None] = print,
    ) -> None:
        self.settings = settings
        self.cwd = cwd
        self.agent = "Codex" if agent == "codex" else "Claude Code"
        self.echo = echo
        #: The coding session every task runs in, once the first has started it.
        self.session_id: str | None = None
        self.tasks: list[Task] = []
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.changed = asyncio.Event()
        self.lines: list[dict[str, str]] = []

    # ── what a call knows ────────────────────────────────────────────────

    @property
    def running(self) -> Task | None:
        return next(
            (t for t in self.tasks if t.started_at and t.finished_at is None), None
        )

    def facts(self) -> list[str]:
        """What a call or a text about the code knows, fresh."""
        stats = code_stats(self.cwd)
        now = time.time()
        running = self.running
        queued = [t for t in self.tasks if t.started_at is None]
        finished = [t for t in self.tasks if t.finished_at is not None]
        facts = [
            f"Repository: {stats.repo}"
            + (f" on branch {stats.branch}" if stats.branch else "")
            + f". The agent: {self.agent}."
        ]
        if running:
            facts.append(
                f'Running now, for {_ago(now - (running.started_at or now))}: "{running.text}".'
                + (
                    " Its latest steps: " + "; ".join(running.activity) + "."
                    if running.activity
                    else ""
                )
            )
        else:
            facts.append("Nothing is running right now.")
        if queued:
            facts.append(
                "Queued after it: " + " | ".join(f'"{t.text}"' for t in queued) + "."
            )
        for task in finished[-5:]:
            outcome = "finished" if task.ok else "could not finish"
            facts.append(
                f'Done {_ago(now - (task.finished_at or now))} ago ({outcome}): "{task.text}". '
                f"What the agent said: {task.report or '(nothing)'}"
            )
        languages = ", ".join(
            f"{lang} {lines:,} lines" for lang, lines in stats.languages
        )
        facts.append(
            f"Code since the last commit: +{stats.added} / -{stats.removed} lines across "
            f"{stats.changed_files} changed and {stats.new_files} new files; "
            f"{stats.commits_today} commits today; {stats.tracked_files} tracked files"
            + (f"; languages: {languages}." if languages else ".")
        )
        return facts

    def texting(self) -> str:
        """Nur's instructions for answering a text to the code line."""
        person = self.settings.name or "the developer"
        return (
            f"You are Nur, {person}'s coding companion, texting with {person} about their "
            f"code. {self.agent} works on it on their computer; you can see what it is "
            "doing from the facts below, which are fresh as of this text. Answer questions "
            "only from the facts; when you don't know, say so.\n\n"
            "The facts:\n- " + "\n- ".join(self.facts())
        )

    def brief(self, *, done: Task | None = None) -> tuple[str, str]:
        """Nur's instructions for a call about the code, and its first words.

        *done*: the call that says a task has just finished.
        """
        name = self.settings.name
        person = name or "the developer"
        hello = f"Hi{' ' + name if name else ''}, it's Nur"
        facts = self.facts()
        stats_repo = Path(self.cwd).name
        now = time.time()
        running = self.running
        finished = [t for t in self.tasks if t.finished_at is not None]
        instructions = (
            f"You are Nur, {person}'s coding companion, on the phone with {person} about "
            f"their code. {self.agent} works on it on their computer; you can see what it "
            "is doing from the facts below, which are fresh as of this call.\n\n"
            "How to talk: this is a phone call. Speak naturally, a sentence or two at a "
            "time, no lists, no markdown, no file paths read character by character. Lead "
            "with what matters, then offer details. Answer questions only from the facts; "
            "when you don't know, say so. When they ask for work, repeat back exactly what "
            "you will hand to the agent and confirm it: it starts when the call ends, after "
            "anything already queued, and you will call them when it is done. When they are "
            "done, or ask you to hang up, say goodbye briefly and end the call. If a voicemail "
            "greeting or an automated menu answers instead of them, leave a one-sentence "
            "update, say goodbye and end the call.\n\n"
            "The facts:\n- " + "\n- ".join(facts)
        )
        if done is not None:
            if done.ok:
                headline = _headline_of(done.report)
                greeting = (
                    f"{hello}. {self.agent} finished {_short(done.text, 60)}"
                    + (f": {headline}" if headline else ".")
                    + " Want the details, or what's next?"
                )
            else:
                greeting = (
                    f"{hello}. {self.agent} couldn't finish {_short(done.text, 60)}: "
                    f"{_short(done.report, 140).rstrip(' .')}."
                )
        elif running:
            greeting = (
                f"{hello}. {self.agent} is working on {_short(running.text, 60)}, "
                f"{_ago(now - (running.started_at or now))} in. What would you like?"
            )
        elif finished:
            last = finished[-1]
            greeting = (
                f"{hello}. {self.agent} finished {_short(last.text, 60)}. What's next?"
            )
        else:
            greeting = (
                f"{hello}. Nothing is running in {stats_repo}. What should "
                f"{self.agent} build?"
            )
        return instructions, greeting

    # ── the loops ────────────────────────────────────────────────────────

    async def run(self) -> None:
        from .client import NurClient

        if not self.settings.phone:
            raise SystemExit("Run `eesi companion setup --phone +14155550123` first.")
        async with NurClient(
            self.settings.api_key, base_url=self.settings.api_url
        ) as nur:
            await self.publish(nur)
            code = [line["number"] for line in self.lines if line.get("kind") == "code"]
            self.echo(
                f"Nur is listening for calls about {Path(self.cwd).name}. "
                + (
                    f"Call the code line from {self.settings.phone}: {', '.join(code)}."
                    if code
                    else "This server has no code line."
                )
            )
            await asyncio.gather(
                self._publishing(nur), self._collecting(nur), self._working(nur)
            )

    async def publish(self, nur: Any) -> None:
        instructions, greeting = self.brief()
        state = await nur.publish_listener(
            phone=str(self.settings.phone),
            instructions=instructions,
            greeting=greeting,
            name=self.settings.name,
            repo=Path(self.cwd).name,
            texting=self.texting(),
        )
        self.lines = list(state.get("lines") or [])

    async def _publishing(self, nur: Any) -> None:
        while True:
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=PUBLISH_SECONDS)
            except asyncio.TimeoutError:  # the builtin only from 3.11
                pass
            self.changed.clear()
            try:
                await self.publish(nur)
            except Exception as error:  # noqa: BLE001 - the next round retries
                self.echo(f"Could not brief the code line: {error}")

    async def _collecting(self, nur: Any) -> None:
        while True:
            try:
                events = await nur.listener_events(wait=20)
            except Exception as error:  # noqa: BLE001
                self.echo(f"Could not collect calls: {error}")
                await asyncio.sleep(5)
                continue
            for event in events:
                if event.get("type") == "text.received":
                    asyncio.create_task(
                        self.take_text(
                            nur, event.get("text") or "", event.get("reply") or ""
                        )
                    )
                elif event.get("call_id"):
                    asyncio.create_task(self.take(nur, str(event["call_id"])))

    async def take(self, nur: Any, call_id: str) -> list[Task]:
        """Queue the work a finished call asked for."""
        call = await nur.wait_for_call(call_id)
        asked = await follow_ups(nur, call.transcript)
        tasks = [Task(text=text) for text in asked]
        for task in tasks:
            self.tasks.append(task)
            await self.queue.put(task)
            self.echo(f"Queued from the call: {task.text}")
        if tasks:
            self.changed.set()
        _record({"call": call_id, "status": call.status, "asked": asked})
        return tasks

    async def take_text(self, nur: Any, text: str, reply: str) -> list[Task]:
        """Queue the work a text to the code line asked for (Nur has replied already)."""
        asked = await follow_ups(nur, (("person", text), ("nur", reply)))
        tasks = [Task(text=item) for item in asked]
        for task in tasks:
            self.tasks.append(task)
            await self.queue.put(task)
            self.echo(f"Queued from a text: {task.text}")
        if tasks:
            self.changed.set()
        _record({"text": text, "reply": reply, "asked": asked})
        return tasks

    async def _working(self, nur: Any) -> None:
        while True:
            task = await self.queue.get()
            await self.work(task)
            notify = self.settings.notify
            if notify in ("text", "both"):
                await self.text_about(nur, task)
            if notify in ("call", "both") and self.settings.call_on != "never":
                await self.call_about(nur, task)

    async def text_about(self, nur: Any, task: Task) -> None:
        """Text the developer the result, from the code line: a reply comes back to Nur."""
        if task.ok:
            headline = _headline_of(task.report)
            message = f"{self.agent} finished {_short(task.text, 80)}" + (
                f": {headline}" if headline else "."
            )
        else:
            message = (
                f"{self.agent} couldn't finish {_short(task.text, 80)}: "
                f"{_short(task.report, 300)}"
            )
        try:
            await nur.send_text(str(self.settings.phone), message[:600])
        except Exception as error:  # noqa: BLE001
            self.echo(f"Could not text: {error}")
            return
        self.echo(f"Texted {self.settings.phone}: {message[:120]}")

    async def work(self, task: Task) -> None:
        """Run *task* in the session, streaming its steps into the brief."""
        task.started_at = time.time()
        self.changed.set()
        self.echo(f"{self.agent} is working on: {task.text}")
        command, report_file = self._command(task)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.cwd,
                env={**os.environ, "NUR_COMPANION_CALL_ON": "never"},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1 << 22,
            )
        except FileNotFoundError:
            task.ok, task.report = (
                False,
                f"the {command[0]} command is not installed here",
            )
        else:
            result: dict[str, Any] = {}
            assert process.stdout is not None and process.stderr is not None
            async for raw in process.stdout:
                event = _json(raw)
                if event is None:
                    continue
                result = self._observe(task, event) or result
                self.changed.set()
            errors = (await process.stderr.read()).decode(errors="replace")
            await process.wait()
            task.ok, task.report = self._outcome(
                process.returncode, result, errors, report_file
            )
        task.finished_at = time.time()
        self.changed.set()
        self.echo(
            f"{'Done' if task.ok else 'Could not finish'}: {task.text} — {task.report[:200]}"
        )

    async def call_about(self, nur: Any, task: Task) -> None:
        """Call the developer with the result; what they ask for is queued."""
        instructions, greeting = self.brief(done=task)
        try:
            placed = await nur.call(
                str(self.settings.phone),
                instructions=instructions,
                greeting=greeting,
                voice=self.settings.voice,
                metadata={"line": "code", "repo": Path(self.cwd).name},
            )
        except Exception as error:  # noqa: BLE001
            self.echo(f"Could not call: {error}")
            return
        self.echo(f"Calling {self.settings.phone} with the result.")
        asyncio.create_task(self.take(nur, placed.id))

    # ── the agent ────────────────────────────────────────────────────────

    def _command(self, task: Task) -> tuple[list[str], Path | None]:
        prompt = f"From a phone call with the developer: {task.text}"
        if self.agent == "Codex":
            companion.HOME.mkdir(parents=True, exist_ok=True)
            report = companion.HOME / f"listen-{os.getpid()}.txt"
            resume = ["resume", self.session_id] if self.session_id else []
            return [
                "codex", "exec", *shlex.split(self.settings.codex_args), "--json",
                "--output-last-message", str(report), *resume, prompt,
            ], report  # fmt: skip
        resume = ["--resume", self.session_id] if self.session_id else []
        return [
            "claude", "-p", *resume, "--output-format", "stream-json", "--verbose",
            *shlex.split(self.settings.claude_args), prompt,
        ], None  # fmt: skip

    def _observe(self, task: Task, event: dict[str, Any]) -> dict[str, Any] | None:
        """One streamed event: the session's id, a step, or the result."""
        kind = event.get("type")
        if self.agent == "Codex":
            if kind == "thread.started" and event.get("thread_id"):
                self.session_id = str(event["thread_id"])
            item = event.get("item") or {}
            if kind == "item.completed":
                if item.get("type") == "command_execution":
                    task.step(f"ran {item.get('command') or ''}")
                elif item.get("type") == "file_change":
                    paths = [
                        Path(str(change.get("path") or "")).name
                        for change in item.get("changes") or []
                    ]
                    task.step("changed " + ", ".join(p for p in paths if p))
                elif item.get("type") == "agent_message":
                    task.step(f"said: {item.get('text') or ''}")
            return None
        if event.get("session_id"):
            self.session_id = str(event["session_id"])
        if kind == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool = block.get("input") or {}
                    target = tool.get("file_path") or tool.get("notebook_path")
                    if (
                        block.get("name")
                        in ("Write", "Edit", "MultiEdit", "NotebookEdit")
                        and target
                    ):
                        task.step(f"edited {Path(str(target)).name}")
                    elif block.get("name") == "Bash" and tool.get("command"):
                        task.step(f"ran {str(tool['command']).splitlines()[0]}")
                    else:
                        task.step(f"used {block.get('name')}")
                elif (
                    block.get("type") == "text" and str(block.get("text") or "").strip()
                ):
                    task.step(f"said: {block['text']}")
        if kind == "result":
            return event
        return None

    def _outcome(
        self,
        returncode: int | None,
        result: dict[str, Any],
        errors: str,
        report_file: Path | None,
    ) -> tuple[bool, str]:
        last_error = next(
            (line.strip() for line in reversed(errors.splitlines()) if line.strip()), ""
        )
        if report_file is not None:
            try:
                report = report_file.read_text(encoding="utf-8").strip()
                report_file.unlink()
            except OSError:
                report = ""
            if returncode == 0 and report:
                return True, report
            return False, last_error[:300] or f"Codex exited with {returncode}"
        text = str(result.get("result") or "").strip()
        if returncode == 0 and text and not result.get("is_error"):
            return True, text
        return False, (text if result.get("is_error") else "") or last_error[:300] or (
            f"Claude Code exited with {returncode}"
        )


def _headline_of(report: str) -> str:
    return _headline(companion.Digest(agent="", session_id=None, cwd="", answer=report))


def _json(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _record(event: dict[str, Any]) -> None:
    companion.HOME.mkdir(parents=True, exist_ok=True)
    with (companion.HOME / "calls.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps({"at": time.time(), **event}) + "\n")


def listen(argv: list[str]) -> int:
    """``eesi companion listen [--agent claude|codex] [--cwd PATH]``."""
    settings = Settings.load()
    agent = _option(argv, "--agent") or "claude"
    if agent not in ("claude", "codex"):
        print("--agent is claude or codex.")
        return 1
    notify = _option(argv, "--notify")
    if notify is not None:
        if notify not in ("call", "text", "both"):
            print("--notify is call, text or both.")
            return 1
        settings.notify = notify
    cwd = os.path.abspath(_option(argv, "--cwd") or os.getcwd())
    listener = Listener(settings, cwd, agent=agent)
    try:
        asyncio.run(listener.run())
    except KeyboardInterrupt:
        print("Nur stopped listening.", file=sys.stderr)
    return 0
