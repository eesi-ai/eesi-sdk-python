"""One task, run through the Claude Agent SDK: plan, approval, work, verification, report.

- Plan: Claude reads what it needs in plan mode (nothing can change) and
  writes the plan. A ``plan_first`` task stops there until a person approves
  it (or turns it down) in the console or by voice.
- Work: the same session goes on with edits accepted, in the task's own
  worktree. Each step is reported as it happens; a step the policy holds
  (``policy.py``) waits for a person; a cancel stops the run.
- Verification: everything the task changed is committed on its branch, and
  the diff is read from git; the repository's checks run (``checks`` in the
  runner's settings), beside the tests Claude ran itself.
- Report: one more turn of the same session, in plan mode, answers a JSON
  schema: what changed, what was verified, what is left, the decisions made
  and what was turned down, the bugs, what to do next, and how the work
  compares with the plan. It is Claude's account, kept beside the evidence.

A ``question`` task (mode ``ask``) is only the first step: Claude answers in
plan mode, in the repository or the session it continues, and changes
nothing.

Files the API hands over with a task (for one the learning loop proposed,
what the bench recorded around its problem's moments) are put in the
worktree's ``.nur-evidence/`` before Claude's first look, named to it in its
instructions, and never committed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import git, policy, redact
from .config import Config, Repo
from .sessions import FAILED, is_test

#: How long a check may run.
CHECK_SECONDS = 15 * 60
#: How long a person has to answer a step Claude waits on.
TOOL_APPROVAL_SECONDS = 30 * 60
#: How long a plan waits for its approval.
PLAN_APPROVAL_SECONDS = 24 * 3600
#: Steps sent to the API at once, and how often at least.
FLUSH_EVERY = 3.0
#: However quiet the work (a long test run, a wait for a person), the API
#: hears from the task this often, or it decides the runner stopped.
HEARTBEAT_SECONDS = 60.0

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Two or three sentences: what was done and why."},
        "changes": {"type": "array", "items": {"type": "string"}, "description": "Each change, in a line."},
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "result": {"type": "string"}},
                "required": ["command", "result"],
            },
            "description": "Every test or check you ran and its result, exactly.",
        },
        "incomplete": {"type": "array", "items": {"type": "string"}, "description": "What is not done."},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "rejected": {"type": "string", "description": "What you turned down, and why."},
                },
                "required": ["title", "why"],
            },
        },
        "bugs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "status": {"type": "string", "enum": ["fixed", "open"]},
                    "detail": {"type": "string"},
                },
                "required": ["title", "status"],
            },
        },
        "lessons": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}, "description": "The most useful next tasks, as instructions."},
        "risks": {"type": "array", "items": {"type": "string"}, "description": "What could regress."},
        "plan_followed": {"type": "string", "enum": ["yes", "partly", "no"]},
        "deviations": {"type": "array", "items": {"type": "string"}, "description": "Where the work differs from the plan, and why."},
    },
    "required": ["summary", "changes", "tests", "incomplete", "decisions", "next_steps", "plan_followed"],
}

REPORT_PROMPT = (
    "Report on this task for the team's records, from what you actually did in this session. "
    "Be exact about tests: list only what you ran, with the result you saw. Name what is not "
    "done. For each decision, why, and what you turned down. Do not change anything now."
)


def _append(task: dict[str, Any], branch: str | None, runner: str, evidence: list[str] | None = None) -> str:
    lines = [
        "You are working on a task the team handed you through Nur, their voice assistant. "
        f"You run unattended on {runner}"
        + (f", in a git worktree on the branch {branch}" if branch else "")
        + ": nobody watches this terminal, so do not ask questions; decide, and say what you decided.",
        "- Work only in this folder. Do not push, deploy, merge or touch shared services: a person "
        "publishes your branch when they decide to. A step that needs a person is asked for "
        "automatically; if it is turned down, do without it or stop and explain.",
        "- Keep the change to the task. Follow the repository's own instructions (AGENTS.md, CLAUDE.md).",
        "- Run the narrowest meaningful tests for what you change, and say exactly which ran and whether they passed.",
        "- End with a short report: what you changed and why, what you verified and how, what is left "
        "undone, and the decisions you made with what you turned down.",
    ]
    if evidence:
        lines.append(
            f"- What the platform recorded for this task is in {git.EVIDENCE_DIR}/ in this folder "
            + ("(start with README.md)" if "README.md" in evidence else f"({', '.join(evidence[:6])})")
            + ": read it before you settle on a cause, and say what in it your change answers. "
            "It is never committed."
        )
    if task.get("context"):
        lines.append("\nWhat the team knows that bears on this:\n" + str(task["context"]))
    return "\n".join(lines)


def _task_prompt(task: dict[str, Any]) -> str:
    text = f"Task {task['ref']}: {task['title']}\n\n{task['objective']}"
    if task.get("constraints"):
        text += f"\n\nConstraints: {task['constraints']}"
    return text


@dataclass
class Outcome:
    ok: bool = False
    report: str = ""
    plan: str | None = None
    outcome: dict[str, Any] = field(default_factory=dict)
    diff_stat: list[dict[str, Any]] = field(default_factory=list)
    diff: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    head_commit: str | None = None
    session_id: str | None = None
    branch: str | None = None
    cwd: str | None = None
    cost_usd: float = 0.0
    turns: int = 0

    def as_finish(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "report": self.report,
            "outcome": self.outcome,
            "diff_stat": self.diff_stat,
            "diff": self.diff,
            "checks": self.checks,
            "error": self.error,
            "head_commit": self.head_commit,
            "session_id": self.session_id,
            "branch": self.branch,
            "cwd": self.cwd,
            "plan": self.plan,
            "cost_usd": round(self.cost_usd, 4),
            "turns": self.turns,
        }


class Stopped(Exception):
    """Someone cancelled the task: stop now."""


class Api:
    """What a task run needs from the API (the runner's client implements it)."""

    async def report(self, task_id: str, **fields: Any) -> dict[str, Any]: ...  # pragma: no cover

    async def ask(self, task_id: str, **fields: Any) -> dict[str, Any]: ...  # pragma: no cover

    async def decision(self, approval_id: str, *, wait: float) -> dict[str, Any]: ...  # pragma: no cover


class TaskRun:
    """One task on this computer, start to finish."""

    def __init__(
        self,
        task: dict[str, Any],
        repo: Repo | None,
        config: Config,
        api: Api,
        *,
        echo: Callable[[str], None] = print,
        query: Callable[..., Any] | None = None,
        sdk: Any = None,
    ) -> None:
        self.task = task
        #: The Claude Agent SDK module (tests hand in a stand-in).
        self._sdk_module = sdk
        self.repo = repo
        self.config = config
        self.api = api
        self.echo = echo
        self._query = query
        self.result = Outcome()
        self._events: list[dict[str, Any]] = []
        self._flushed = time.monotonic()
        self._stop = asyncio.Event()
        self._pending_tests: dict[str, str] = {}
        self._claude_checks: list[dict[str, Any]] = []
        self._exit_plan: str | None = None
        #: The evidence the API handed over, as written into the worktree.
        self._evidence: list[str] = []

    # ── plumbing ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop.set()

    def event(self, kind: str, text: str, **data: Any) -> None:
        self._events.append({"at": time.time(), "kind": kind, "text": redact.text(" ".join(text.split()))[:600], "data": data})

    async def flush(self, *, force: bool = False, **fields: Any) -> None:
        if not force and not fields and (time.monotonic() - self._flushed < FLUSH_EVERY or not self._events):
            return
        events, self._events = self._events[:200], self._events[200:]
        self._flushed = time.monotonic()
        answer = await self.api.report(self.task["id"], events=events, **fields)
        if answer.get("stop"):
            self._stop.set()
            raise Stopped()

    def _sdk(self) -> Any:
        if self._sdk_module is None:
            import claude_agent_sdk

            self._sdk_module = claude_agent_sdk
        return self._sdk_module

    def _cli_path(self) -> str | None:
        return self.config.claude_path or shutil.which("claude")

    # ── permissions ──────────────────────────────────────────────────────

    def _verdict(self, name: str, arguments: dict[str, Any]) -> tuple[str, str]:
        return policy.tool(
            name, arguments, cwd=self.result.cwd or os.getcwd(), allow=self.config.allow, deny=self.config.deny
        )

    async def _pre_tool_use(self, payload: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """Before every tool call, whatever a user's own permission rules allow."""
        del tool_use_id, context
        name = str((payload or {}).get("tool_name") or "")
        arguments = (payload or {}).get("tool_input") or {}
        verdict, why = self._verdict(name, arguments if isinstance(arguments, dict) else {})
        if verdict == "deny":
            self.event("permission", f"Not allowed: {policy.describe(name, arguments)} ({why})")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Not allowed in a Nur task: {why}.",
                }
            }
        if verdict == "ask":
            if self.config.unattended:
                self.event("permission", f"Went ahead by itself (unattended): {policy.describe(name, arguments)} ({why})")
                return {}
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": why,
                }
            }
        return {}

    async def _can_use_tool(self, name: str, arguments: dict[str, Any], context: Any) -> Any:
        """A step that would prompt: allowed, refused, or asked of a person and waited on."""
        sdk = self._sdk()
        if name == "ExitPlanMode":
            # The plan is recorded; the turn ends in plan mode, and the work
            # starts (or waits for approval) as the runner decides.
            self._exit_plan = str(arguments.get("plan") or "")
            return sdk.PermissionResultDeny(message="Plan recorded. Stop here; the work starts in the next step.")
        if name == "AskUserQuestion":
            return sdk.PermissionResultDeny(
                message="Nobody is at this terminal to answer. Decide yourself, say what you decided and why, and go on."
            )
        del context
        verdict, reason = self._verdict(name, arguments)
        if verdict == "allow":
            return sdk.PermissionResultAllow()
        if verdict == "deny":
            return sdk.PermissionResultDeny(message=f"Not allowed in a Nur task: {reason}.")
        summary = policy.describe(name, arguments)
        if self.config.unattended:
            # Nobody watches this runner: the step is logged and goes ahead.
            self.event("permission", f"Went ahead by itself (unattended): {summary} ({reason})")
            return sdk.PermissionResultAllow()
        self.event("permission", f"Waiting for a person: {summary} ({reason})")
        await self.flush(force=True)
        approval = await self.api.ask(
            self.task["id"],
            kind="tool",
            tool=name,
            summary=f"{summary} — {reason}" if reason else summary,
            detail={"input": redact.deep(arguments), "reason": reason},
        )
        decided = await self._wait(approval["id"], TOOL_APPROVAL_SECONDS)
        if decided.get("status") == "approved":
            return sdk.PermissionResultAllow()
        note = decided.get("note")
        if decided.get("status") == "denied":
            return sdk.PermissionResultDeny(message=f"A person turned this down{': ' + note if note else ''}. Do without it, or stop and explain.")
        return sdk.PermissionResultDeny(message="Nobody answered in time. Do without it, or stop and explain what is needed.")

    async def _wait(self, approval_id: str, seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise Stopped()
            decided = await self.api.decision(approval_id, wait=20)
            if decided.get("status") != "pending":
                return decided
            await self.flush()
        return {"status": "expired"}

    # ── one query ────────────────────────────────────────────────────────

    def _options(self, *, mode: str, resume: str | None, budget: float, schema: dict[str, Any] | None = None, fork: bool = False) -> Any:
        sdk = self._sdk()
        options: dict[str, Any] = {
            "cwd": self.result.cwd,
            "permission_mode": mode,
            "resume": resume,
            "fork_session": fork,
            "system_prompt": {"type": "preset", "preset": "claude_code", "append": _append(self.task, self.result.branch, self.config.name, self._evidence)},
            "setting_sources": ["user", "project", "local"],
            "max_budget_usd": max(0.5, budget),
            "max_turns": 400,
            # The companion plugin's own calls stay quiet (Nur reports this task
            # herself), and the runner's key is not Claude's to see.
            "env": {
                "NUR_COMPANION_CALL_ON": "never",
                "EESI_ENGINEER_TASK": self.task["ref"],
                "EESI_API_KEY": "",
            },
            "can_use_tool": self._can_use_tool,
            "hooks": {"PreToolUse": [sdk.HookMatcher(matcher=None, hooks=[self._pre_tool_use])]},
            "stderr": lambda line: None,
        }
        if self._cli_path():
            options["cli_path"] = self._cli_path()
        if schema is not None:
            options["output_format"] = {"type": "json_schema", "schema": schema}
        return sdk.ClaudeAgentOptions(**options)

    async def _converse(self, prompt: str, options: Any, *, phase: str) -> Any:
        """Run one turn of the session; the result message. Steps become events as they happen."""
        sdk = self._sdk()
        query = self._query or sdk.query
        result = None
        async for message in query(prompt=prompt, options=options):
            if self._stop.is_set():
                raise Stopped()
            if isinstance(message, sdk.SystemMessage) and message.subtype == "init":
                session = (message.data or {}).get("session_id")
                if session and session != self.result.session_id:
                    self.result.session_id = session
                    await self.flush(force=True, session_id=session)
            elif isinstance(message, sdk.AssistantMessage):
                self._assistant(message, sdk, phase)
            elif isinstance(message, sdk.UserMessage):
                self._user(message, sdk)
            elif isinstance(message, sdk.ResultMessage):
                result = message
                self.result.session_id = message.session_id or self.result.session_id
                self.result.cost_usd += float(message.total_cost_usd or 0)
                self.result.turns += int(message.num_turns or 0)
            await self.flush()
        return result

    def _assistant(self, message: Any, sdk: Any, phase: str) -> None:
        for block in message.content:
            if isinstance(block, sdk.TextBlock) and block.text.strip():
                self.event("text", block.text.strip()[:600], phase=phase)
            elif isinstance(block, sdk.ToolUseBlock):
                arguments = block.input if isinstance(block.input, dict) else {}
                name = block.name
                path = arguments.get("file_path") or arguments.get("notebook_path")
                if name in ("Write", "Edit", "MultiEdit", "NotebookEdit") and path:
                    shown = os.path.relpath(path, self.result.cwd) if self.result.cwd and os.path.isabs(path) else path
                    self.event("file", f"edited {shown}", path=shown)
                elif name == "Bash" and arguments.get("command"):
                    command = str(arguments["command"]).strip().splitlines()[0][:300]
                    self.event("command", f"ran {command}")
                    if is_test(command):
                        self._pending_tests[block.id] = command
                elif name == "TodoWrite":
                    todos = arguments.get("todos") or []
                    done = sum(1 for todo in todos if isinstance(todo, dict) and todo.get("status") == "completed")
                    current = next((todo.get("activeForm") or todo.get("content") for todo in todos if isinstance(todo, dict) and todo.get("status") == "in_progress"), None)
                    self.event("plan", f"{done} of {len(todos)} steps done" + (f"; now: {current}" if current else ""), todos=todos[:30])
                elif name in ("Read", "Grep", "Glob", "LS"):
                    target = arguments.get("file_path") or arguments.get("pattern") or arguments.get("path") or ""
                    self.event("tool", f"{name.lower()} {target}"[:200])
                elif name in ("Task", "Agent"):
                    self.event("tool", f"asked a helper: {arguments.get('description') or arguments.get('prompt', '')[:120]}")
                else:
                    self.event("tool", f"used {name}")

    def _user(self, message: Any, sdk: Any) -> None:
        content = message.content if isinstance(message.content, list) else []
        for block in content:
            if not isinstance(block, sdk.ToolResultBlock):
                continue
            if isinstance(block.content, str):
                output = block.content
            else:
                output = json.dumps(block.content)[:8000] if block.content else ""
            command = self._pending_tests.pop(block.tool_use_id, None)
            failed = bool(block.is_error)
            if command is not None:
                passed = not failed and not FAILED.search(str(output)[-4000:])
                self._claude_checks.append(
                    {"command": command, "exit_code": None, "passed": passed, "source": "claude", "output": redact.text(str(output))[-2000:]}
                )
                self.event("check", f"{'passed' if passed else 'FAILED'}: {command}", passed=passed)
            elif failed:
                self.event("error", redact.text(str(output))[:300])

    # ── the phases ───────────────────────────────────────────────────────

    @property
    def ours(self) -> bool:
        """The task continues one of this runner's own tasks, in its worktree."""
        cwd = self.task.get("cwd")
        return bool(
            cwd
            and self.task.get("branch")
            and Path(cwd).exists()
            and Path(cwd).resolve().is_relative_to(Path(self.config.home).resolve())
        )

    async def prepare(self) -> None:
        task = self.task
        if task["mode"] == "ask":
            self.result.cwd = task.get("cwd") if task.get("cwd") and Path(task["cwd"]).exists() else (self.repo.path if self.repo else os.getcwd())
            return
        if self.ours:
            # Continuing a task of ours: its own worktree and branch.
            self.result.cwd, self.result.branch = task["cwd"], task["branch"]
            base = git.out(task["cwd"], "merge-base", "HEAD", f"origin/{git.default_branch(task['cwd'])}") or git.head(task["cwd"])
        else:
            if self.repo is None:
                raise RuntimeError(f"this runner has no repository called {task.get('repo')!r}")
            # Continuing someone's own session: from where its branch is, in a
            # worktree of its own, so their checkout is never touched.
            start = task.get("branch") if task.get("continues") and task.get("branch") else None
            folder, branch, base = await asyncio.to_thread(
                git.create_worktree,
                self.repo.path,
                self.config.home,
                task["ref"],
                task["title"],
                start=start,
                links=self.repo.links,
            )
            self.result.cwd, self.result.branch = folder, branch
        self._base = base
        self.event("status", f"Working in {self.result.cwd} on {self.result.branch}")
        if task.get("files"):
            self._evidence = await asyncio.to_thread(git.hand_over, self.result.cwd, task["files"])
            if self._evidence:
                self.event("status", f"Put {len(self._evidence)} files of evidence in {git.EVIDENCE_DIR}/")
        await self.flush(force=True, cwd=self.result.cwd, branch=self.result.branch, base_commit=base)

    async def plan(self) -> str:
        task = self.task
        budget = min(2.0, float(task.get("max_budget_usd") or 10) / 4)
        prompt = _task_prompt(task) + (
            "\n\nAnswer this from the code; change nothing."
            if task["mode"] == "ask"
            else "\n\nMake a plan first: read what you need, then give the plan as short steps — the files to "
            "change, the approach, how you will verify it, the risks. Change nothing yet."
        )
        resume = task.get("continues")
        # Someone's own session is forked, never written to: it stays theirs.
        fork = bool(resume) and (task["mode"] == "ask" or not self.ours)
        try:
            result = await self._converse(prompt, self._options(mode="plan", resume=resume, budget=budget, fork=fork), phase="plan")
        except Exception as exc:  # noqa: BLE001 - a session that cannot be resumed starts fresh
            if not resume or isinstance(exc, Stopped):
                raise
            self.event("note", f"Could not resume {resume[:8]} ({type(exc).__name__}); starting fresh with what was known")
            result = await self._converse(prompt, self._options(mode="plan", resume=None, budget=budget), phase="plan")
        text = (self._exit_plan or (result.result if result else "") or "").strip()
        if result is not None and result.is_error and not text:
            raise RuntimeError("; ".join(result.errors or []) or "Claude could not plan it")
        return text

    async def work(self, plan: str, note: str | None) -> Any:
        task = self.task
        if plan:
            prompt = "The plan " + ("was approved" + (f", with this note: {note}" if note else "") if task["mode"] == "plan_first" else "stands") + ". Carry it out now, then report."
        else:
            prompt = _task_prompt(task)
        await self.flush(force=True, status="running")
        return await self._converse(
            prompt,
            self._options(mode="acceptEdits", resume=self.result.session_id, budget=float(task.get("max_budget_usd") or 10)),
            phase="work",
        )

    async def verify(self) -> None:
        cwd = self.result.cwd
        assert cwd is not None
        await self.flush(force=True, status="verifying")
        message = f"{self.task['title']}\n\n{self.task['objective'][:800]}\n\nNur-Task: {self.task['ref']}"
        try:
            head = await asyncio.to_thread(
                git.commit_all, cwd, message, links=self.repo.links if self.repo else ()
            )
        except git.GitError as exc:
            self.event("error", str(exc))
            head = None
        self.result.head_commit = head or git.head(cwd)
        base = getattr(self, "_base", None) or self.result.head_commit
        self.result.diff_stat = await asyncio.to_thread(git.diff_stat, cwd, base)
        self.result.diff = await asyncio.to_thread(git.diff_patch, cwd, base)
        checks = list(self._claude_checks)
        if self.result.diff_stat and self.repo is not None:
            for command in self.repo.checks:
                check = await asyncio.to_thread(run_check, command, cwd)
                if not check["passed"] and base:
                    # Did this change break it, or was it already failing?
                    earlier = await asyncio.to_thread(self._before, command, cwd, base)
                    if earlier is not None:
                        check["before"] = "passed" if earlier["passed"] else "failed"
                        # Tests failing now that did not before: a regression
                        # even when the suite was already red.
                        check["new_failures"] = sorted(
                            failing_tests(check["output"]) - failing_tests(earlier["output"])
                        )[:20]
                checks.append(check)
                verdict = "passed" if check["passed"] else "FAILED" + (
                    f" (newly failing: {', '.join(check['new_failures'][:3])})"
                    if check.get("new_failures")
                    else {
                        "passed": " (it passed before this change)",
                        "failed": " (it already failed before this change)",
                    }.get(str(check.get("before")), "")
                )
                self.event("check", f"{verdict}: {command}", passed=check["passed"], before=check.get("before"))
                await self.flush()
        self.result.checks = checks

    def _before(self, command: str, cwd: str, base: str) -> dict[str, Any] | None:
        """The same check at the base commit, or None when it could not run there."""
        place = Path(self.config.home) / "baseline" / f"{self.task['ref']}-{abs(hash(command)) % 10_000}"
        folder = git.at_commit(cwd, base, place, links=self.repo.links if self.repo else ())
        if folder is None:
            return None
        try:
            return run_check(command, folder)
        finally:
            git.drop_worktree(cwd, place)

    async def report(self) -> None:
        try:
            result = await self._converse(
                REPORT_PROMPT,
                self._options(mode="plan", resume=self.result.session_id, budget=1.0, schema=REPORT_SCHEMA),
                phase="report",
            )
        except Stopped:
            raise
        except Exception as exc:  # noqa: BLE001 - the evidence stands without Claude's account
            self.event("note", f"No report from Claude ({type(exc).__name__})")
            return
        data = getattr(result, "structured_output", None) if result is not None else None
        if not isinstance(data, dict) and result is not None and result.result:
            try:
                start = result.result.index("{")
                data = json.loads(result.result[start : result.result.rindex("}") + 1])
            except ValueError:
                data = None
        if isinstance(data, dict):
            self.result.outcome = data
        elif result is not None and result.result:
            self.result.outcome = {"summary": result.result[:2000]}

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                answer = await self.api.report(self.task["id"], events=[])
            except Exception:  # noqa: BLE001 - the next beat tries again
                continue
            if answer.get("stop"):
                self._stop.set()

    async def run(self) -> Outcome:
        """The whole task. Never raises: what happened is always in the outcome."""
        beating = asyncio.create_task(self._heartbeat())
        try:
            return await self._run()
        finally:
            beating.cancel()

    async def _run(self) -> Outcome:
        task = self.task
        try:
            await self.prepare()
            plan = await self.plan()
            self.result.plan = plan or None
            if task["mode"] == "ask":
                self.result.ok = bool(plan)
                self.result.report = plan
                self.result.outcome = {"summary": plan[:2000]}
                return self.result
            if plan:
                self.event("plan", "Plan written", plan=plan[:6000])
            await self.flush(force=True, plan=plan or None)
            note = None
            if task["mode"] == "plan_first":
                approval = await self.api.ask(
                    task["id"],
                    kind="plan",
                    summary=f"Approve the plan for {task['ref']}: {task['title']}",
                    detail={"plan": plan},
                )
                decided = await self._wait(approval["id"], PLAN_APPROVAL_SECONDS)
                if decided.get("status") != "approved":
                    self.result.error = "The plan was " + ("turned down" if decided.get("status") == "denied" else "not approved in time") + (f": {decided['note']}" if decided.get("note") else ".")
                    return self.result
                note = decided.get("note")
            result = await self.work(plan, note)
            self.result.report = (result.result if result else "") or ""
            failed_run = result is None or result.is_error
            await self.verify()
            await self.report()
            self.result.ok = not failed_run
            if failed_run:
                self.result.error = "; ".join((result.errors or []) if result else []) or (result.subtype if result else "Claude stopped without a result")
        except Stopped:
            self.result.error = "Cancelled."
            if self.result.cwd and task["mode"] != "ask":
                try:
                    await self.verify()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            self.result.error = f"{type(exc).__name__}: {redact.text(str(exc))[:600]}"
        return self.result


#: A failed test as pytest and vitest name it in their summaries.
FAILED_TEST = re.compile(r"(?m)^\s*(?:FAILED|FAIL|×|✗)\s+(\S[^\n]*?)(?:\s+-\s[^\n]*)?$")


def failing_tests(output: str) -> set[str]:
    return {name.strip() for name in FAILED_TEST.findall(output or "") if name.strip()}


def run_check(command: str, cwd: str) -> dict[str, Any]:
    """One of the repository's checks, in the task's worktree: its exit code and the end of its output."""
    started = time.monotonic()
    try:
        done = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=CHECK_SECONDS,
            check=False,
            env={**os.environ, "NUR_COMPANION_CALL_ON": "never"},
        )
        code, output = done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired:
        code, output = 124, f"stopped after {CHECK_SECONDS // 60} minutes"
    return {
        "command": command,
        "exit_code": code,
        "passed": code == 0,
        "source": "runner",
        "output": redact.text(output)[-3000:],
        "seconds": round(time.monotonic() - started, 1),
    }

