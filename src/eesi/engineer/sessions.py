"""The Claude Code sessions on this computer, as digests, through the Claude Agent SDK.

``list_sessions`` (the SDK's supported listing: metadata from a stat and the
head and tail of each transcript) finds the sessions in each repository and
its git worktrees; ``get_session_messages`` reads one that changed since the
runner last looked. A digest keeps what a person would want to hear about
and nothing more: what was asked, Claude's last words, the files it wrote,
the commands it ran, the tests among them and whether they passed, how many
steps failed. Redacted here, before it leaves the computer.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from . import redact
from .config import Repo

#: A first look reports sessions this recent; later looks, whatever changed.
FIRST_LOOK_DAYS = 14
#: The Bash commands that are tests, whose outcome a digest keeps.
TESTS = re.compile(
    r"(?:^|[\s/;&|(])(?:pytest|py\.test|vitest|jest|mocha|playwright\s+test|go\s+test|cargo\s+test"
    r"|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:\S+\s+)?test(?::\S*)?\b|make\s+test[\w-]*|tox\b"
    r"|python\d?(?:\.\d+)?\s+-m\s+(?:pytest|unittest)|rspec|phpunit|swift\s+test|xcodebuild\s+test"
    r"|scripts/test[\w./-]*|(?:pnpm|npm|yarn)\s+(?:\S+\s+)?(?:typecheck|lint)\b|ruff\s+check|mypy\b|tsc\b)"
)
_EDITS = ("Write", "Edit", "MultiEdit", "NotebookEdit")


def is_test(command: str) -> bool:
    return bool(TESTS.search(command))


#: A test run's own summary saying something failed, whatever the exit code
#: (a test piped through ``tail`` exits 0).
FAILED = re.compile(
    r"\b[1-9]\d* (?:failed|failing|errors?)\b|\bFAILED\b|Tests?:\s+[1-9]\d* failed"
    r"|Test Files\s+[1-9]\d* failed|\bERROR:|error TS\d+|Found [1-9]\d* errors?"
)
_INJECTED = re.compile(r"<(system-reminder|local-command-\w+|command-\w+)>.*?</\1>", re.DOTALL)


def _said(content: Any) -> str:
    """What a person typed in a user message: its text blocks, without what the app injected."""
    if isinstance(content, str):
        blocks = [content]
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return ""
        blocks = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
    else:
        return ""
    kept = []
    for block in blocks:
        block = _INJECTED.sub("", block).strip()
        # A compacted session restates itself as a long summary: not their words.
        if block and not block.startswith("<") and not block.startswith(
            "This session is being continued from a previous conversation"
        ):
            kept.append(block)
    return "\n".join(kept)


def _result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return ""


def _relative(path: str, cwd: str | None) -> str:
    if cwd and os.path.isabs(path):
        try:
            return os.path.relpath(path, cwd)
        except ValueError:
            return path
    return path


def digest(info: Any, messages: list[Any], repo: Repo, *, seen: int = 0) -> dict[str, Any]:
    """One session as the API keeps it. *seen*: messages already reported (edits after it count as new)."""
    cwd = getattr(info, "cwd", None) or repo.path
    prompts: list[str] = []
    answer = ""
    files: list[str] = []
    edited: dict[str, float] = {}
    commands: list[str] = []
    pending: dict[str, str] = {}
    tests: list[dict[str, Any]] = []
    errors = 0
    modified = float(getattr(info, "last_modified", 0) or 0) / 1000
    for index, item in enumerate(messages):
        message = getattr(item, "message", None) or {}
        content = message.get("content") if isinstance(message, dict) else None
        if getattr(item, "type", None) == "user":
            text = _said(content)
            if text:
                prompts.append(text)
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                failed = bool(block.get("is_error"))
                errors += failed
                command = pending.pop(str(block.get("tool_use_id") or ""), None)
                if command is not None:
                    output = _result_text(block)[-4000:]
                    tests.append({"command": command, "ok": not failed and not FAILED.search(output)})
        elif getattr(item, "type", None) == "assistant":
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and str(block.get("text") or "").strip():
                    answer = str(block["text"]).strip()
                elif block.get("type") == "tool_use":
                    arguments = block.get("input") if isinstance(block.get("input"), dict) else {}
                    name = block.get("name")
                    path = arguments.get("file_path") or arguments.get("notebook_path")
                    if name in _EDITS and path:
                        relative = _relative(str(path), cwd)
                        if relative not in files:
                            files.append(relative)
                        if index >= seen:
                            edited[relative] = modified
                    elif name == "Bash" and arguments.get("command"):
                        command = str(arguments["command"]).strip().splitlines()[0][:300]
                        commands.append(command)
                        if is_test(command):
                            pending[str(block.get("id") or "")] = command
                    elif name == "SubagentHandback" and str(arguments.get("message") or "").strip():
                        answer = str(arguments["message"]).strip()
    title = getattr(info, "custom_title", None) or getattr(info, "summary", None) or (prompts[0] if prompts else "")
    if str(title).lstrip().startswith("<"):
        title = prompts[0] if prompts else ""
    first = getattr(info, "first_prompt", None) or ""
    if not first or first.lstrip().startswith("<"):
        first = prompts[0] if prompts else ""
    created = getattr(info, "created_at", None)
    return {
        "session_id": info.session_id,
        "repo": repo.name,
        "cwd": cwd,
        "branch": getattr(info, "git_branch", None),
        "title": redact.text(" ".join(str(title).split()))[:300],
        "first_prompt": redact.text(first)[:4000],
        "last_prompt": redact.text(prompts[-1] if prompts else "")[:4000],
        "last_answer": redact.text(answer)[:6000],
        "prompts": redact.lines(prompts[-6:], limit=6, width=600),
        "files": files[:200],
        "edited": edited,
        "commands": redact.lines(commands[-30:], limit=30, width=300),
        "tests": [{"command": redact.text(test["command"]), "ok": test["ok"]} for test in tests[-20:]],
        "errors": errors,
        "messages": len(messages),
        "started_at": (created / 1000) if created else None,
        "last_activity_at": modified,
    }


class Discovery:
    """What changed among the sessions in the runner's repositories since it last looked."""

    def __init__(self, home: str | Path) -> None:
        self.path = Path(home) / "sessions.json"
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        self.state: dict[str, dict[str, Any]] = state if isinstance(state, dict) else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state), encoding="utf-8")

    def changed(self, repos: list[Repo], *, limit: int = 200) -> list[dict[str, Any]]:
        """Digests of the sessions that changed; their new state is kept by ``commit``."""
        from claude_agent_sdk import get_session_messages, list_sessions

        first_look = time.time() - FIRST_LOOK_DAYS * 86_400
        digests: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}
        for repo in repos:
            try:
                infos = list_sessions(directory=repo.path, limit=limit, include_worktrees=True)
            except Exception:  # noqa: BLE001 - one repository's trouble is not the others'
                continue
            for info in infos:
                known = self.state.get(info.session_id) or {}
                if known.get("modified") == info.last_modified:
                    continue
                if not known and info.last_modified / 1000 < first_look:
                    continue
                messages = get_session_messages(info.session_id, directory=info.cwd or repo.path)
                if not messages:
                    continue
                digests.append(digest(info, messages, repo, seen=int(known.get("count") or 0)))
                self._pending[info.session_id] = {"modified": info.last_modified, "count": len(messages)}
        return digests

    def commit(self) -> None:
        """The digests ``changed`` returned reached the API: do not send them again."""
        self.state.update(getattr(self, "_pending", {}))
        self._pending = {}
        self.save()
