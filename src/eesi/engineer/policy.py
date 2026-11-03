"""What Claude may do in a task by itself, what waits for a person, and what it never does.

A task runs in its own worktree with edits accepted, so the work itself
(reading, editing, building, testing, committing on its branch) goes ahead.
Three kinds of step do not:

- never: anything that leaves this computer or changes shared things
  (pushing, deploying, cloud and cluster commands, publishing packages,
  merging pull requests), privilege, and reading credentials (this runner's
  own key among them). Publishing a finished task is a separate, explicit
  request a person makes (``decide`` publish, or the console's button).
- ask a person: deleting outside build output, rewriting history, adding new
  dependencies, editing files outside the worktree, sending requests that
  change something on another server. The runner asks the API and waits;
  Nur tells them, and they answer in the console or by voice.
- everything else: allowed.

The checks run as a ``PreToolUse`` hook (which permission rules in a user's
own settings cannot skip) and again in ``can_use_tool``.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from . import redact

Verdict = Literal["allow", "ask", "deny"]

#: Folders whose contents are credentials: never read, never written.
_PRIVATE = (
    "~/.eesi",
    "~/.ssh",
    "~/.aws",
    "~/.azure",
    "~/.config/gcloud",
    "~/.kube",
    "~/.docker/config.json",
    "~/.netrc",
    "~/.npmrc",
    "~/.pypirc",
    "~/.git-credentials",
    "~/.claude/.credentials.json",
)

_NEVER: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), why)
    for pattern, why in (
        (r"^git\s+push\b", "pushing is a separate request a person makes (publish)"),
        (r"^git\s+.*\bpush\b", "pushing is a separate request a person makes (publish)"),
        (r"^gh\s+pr\s+(merge|close|review)\b", "merging, closing or reviewing pull requests is for people"),
        (r"^gh\s+(release|repo\s+(delete|edit|rename|archive)|secret|variable|workflow\s+run|run\s+(rerun|cancel))\b", "changes the shared repository"),
        (r"^gh\s+api\b.*(-X|--method)\s*(POST|PUT|PATCH|DELETE)\b", "changes the shared repository"),
        (r"^(kubectl|helm|kustomize\s+build\s+.*\|\s*kubectl|k9s|skaffold)\b", "touches a cluster"),
        (r"^(gcloud|gsutil|bq|aws|az|doctl|flyctl|fly|vercel|netlify|heroku|wrangler)\b", "touches cloud resources"),
        (r"^terraform\s+(apply|destroy|import|state|taint)\b", "changes infrastructure"),
        (r"^docker\s+(push|login)\b", "publishes an image"),
        (r"^(npm|pnpm|yarn)\s+publish\b|^(twine|uv)\s+(upload|publish)\b|^cargo\s+publish\b|^poetry\s+publish\b", "publishes a package"),
        (r"^make\s+(deploy|release|publish|nur-live-up|nur-live-down)\b", "deploys"),
        (r"(^|/)(deploy|release)[\w.-]*\.(sh|py)\b", "deploys"),
        (r"^sudo\b|^su\s|^doas\b", "needs privilege"),
        (r"^(ssh|scp|sftp|rsync\s+.*:)\b", "reaches another computer"),
        (r"\b(curl|wget)\b.*\|\s*(ba|z)?sh\b", "runs code from the internet"),
        (r"^(shutdown|reboot|halt|poweroff|mkfs|dd)\b", "touches the computer itself"),
        (r"^rm\s+(-\w*r\w*\s+)+(/|~|\$HOME|\.\.)(\s|/?$)", "deletes far outside the task"),
        (r"^(printenv|env)\s*$|^export\s+-p\b|^set\s*$", "prints the environment, which holds credentials"),
        (r"^telnyx\b|^stripe\b", "touches a live service"),
    )
)

_ASK: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), why)
    for pattern, why in (
        (r"^rm\s+.*-\w*[rf]", "deletes files recursively"),
        (r"^git\s+(reset\s+--hard|clean\s+-\w*f|rebase|filter-branch|filter-repo|update-ref\s+-d|branch\s+-D|tag\s+-d|stash\s+(drop|clear))\b", "rewrites or throws away history"),
        (r"^git\s+(checkout|switch)\s+.*\b(main|master)\b", "moves off the task's own branch"),
        (r"^(pip|pip3|uv\s+pip)\s+install\s+(?!-r\b|-e\s+\.|--requirement)", "adds a new dependency"),
        (r"^uv\s+add\b|^poetry\s+add\b|^cargo\s+add\b|^go\s+get\b", "adds a new dependency"),
        (r"^(npm|pnpm|yarn)\s+(add|install|i)\s+[^-\s]", "adds a new dependency"),
        (r"\b(curl|wget|http|httpie)\b.*(-X\s*(POST|PUT|PATCH|DELETE)|--data\b|-d\s|--request\s+(POST|PUT|PATCH|DELETE))", "sends a change to another server"),
        (r"^(psql|mysql|mongo|mongosh|redis-cli)\b.*(\bDROP\b|\bDELETE\b|\bTRUNCATE\b|\bFLUSHALL\b|\bFLUSHDB\b|\bUPDATE\b)", "changes a database"),
        (r"\balembic\s+(downgrade|stamp)\b", "changes a database's migrations"),
        (r"^chmod\s+-R\b|^chown\b", "changes permissions"),
        (r"^crontab\b|^systemctl\b|^launchctl\b", "changes what runs on this computer"),
    )
)

_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\n)\s*")


def _expand(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


#: Files that are credentials by name: environment files, keys, credential stores.
_CREDENTIAL_FILE = re.compile(
    r"(?i)(?:^|/)(?:\.env(?:\.(?!example$|sample$|template$)[\w.-]+)?"
    r"|[\w.-]+\.(?:pem|key|p12|pfx|jks|keystore)|id_(?:rsa|dsa|ecdsa|ed25519)"
    r"|credentials(?:\.json)?|\.npmrc|\.pypirc|\.netrc|\.git-credentials"
    r"|service[-_]account[\w.-]*\.json)$"
)


def _inside(path: str, folder: str) -> bool:
    real, root = _expand(path), _expand(folder).rstrip("/")
    return real == root or real.startswith(root + "/")


def _in_private_folder(path: str, cwd: str | None = None) -> bool:
    # The task's own worktree is never a credentials folder, even where it
    # lives under one (the runner's home defaults to ~/.eesi/engineer).
    if cwd and _inside(path, cwd):
        return False
    real = _expand(path)
    for folder in _PRIVATE:
        root = _expand(folder)
        if real == root or real.startswith(root.rstrip("/") + "/"):
            return True
    return False


def private(path: str, cwd: str | None = None) -> bool:
    """A credentials folder (outside the task's worktree *cwd*), or a
    credentials file (an .env, a key) anywhere."""
    return _in_private_folder(path, cwd) or bool(_CREDENTIAL_FILE.search(path.strip()))


def _claude_plan(path: str) -> bool:
    """Claude Code's own plan file: plan mode writes the plan it is asked for
    to ``~/.claude/plans/<name>.md``, which is Claude's, not a person's."""
    root = os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
    return path.endswith(".md") and _inside(path, os.path.join(root, "plans"))


def _segments(command: str) -> list[str]:
    parts: list[str] = []
    for chunk in _SPLIT.split(command.strip()):
        chunk = chunk.strip()
        # A subshell or a pipe's first command counts on its own too.
        chunk = re.sub(r"^\(+|\)+$", "", chunk).strip()
        if chunk:
            parts.append(chunk)
            for piece in chunk.split("|")[1:]:
                if piece.strip():
                    parts.append(piece.strip())
    return parts or [command.strip()]


def _strip_prefixes(segment: str) -> str:
    """``FOO=1 time nice cmd`` → ``cmd``: what actually runs."""
    words = segment.split()
    while words and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]) or words[0] in ("time", "nice", "nohup", "command", "builtin", "exec", "timeout")):
        if words[0] == "timeout" and len(words) > 1:
            words = words[1:]
        words = words[1:]
    return " ".join(words)


def bash(command: str, *, cwd: str, allow: Iterable[str] = (), deny: Iterable[str] = ()) -> tuple[Verdict, str]:
    """The verdict on a shell command, and why."""
    text = command.strip()
    for path in re.findall(r"(?:~|\$HOME|/)[^\s'\"|;&<>]*", text):
        candidate = path.replace("$HOME", "~")
        if _in_private_folder(candidate, cwd):
            return "deny", f"reads or writes credentials ({candidate})"
    for word in re.findall(r"[^\s'\"|;&<>]+", text):
        if _CREDENTIAL_FILE.search(word) and not word.startswith("-"):
            return "deny", f"{word} holds credentials"
    verdict: Verdict = "allow"
    reason = ""
    for segment in _segments(text):
        segment = _strip_prefixes(segment)
        if any(segment.startswith(prefix) for prefix in deny):
            return "deny", "this runner's settings forbid it"
        if any(segment.startswith(prefix) for prefix in allow):
            continue
        for pattern, why in _NEVER:
            if pattern.search(segment):
                return "deny", why
        for pattern, why in _ASK:
            if pattern.search(segment):
                if not (why == "deletes files recursively" and _only_build_output(segment, cwd)):
                    verdict, reason = "ask", why
    return verdict, reason


def _only_build_output(segment: str, cwd: str) -> bool:
    """``rm -rf node_modules dist .pytest_cache`` inside the worktree: no need to ask."""
    try:
        words = shlex.split(segment)
    except ValueError:
        return False
    targets = [word for word in words[1:] if not word.startswith("-")]
    disposable = {"node_modules", "dist", "build", ".next", ".turbo", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "coverage", ".coverage", "out", "target", ".venv", "tmp"}
    for target in targets:
        real = _expand(os.path.join(cwd, target))
        if not real.startswith(_expand(cwd).rstrip("/") + "/"):
            return False
        if not set(Path(real).relative_to(_expand(cwd)).parts) & disposable:
            return False
    return bool(targets)


def tool(name: str, arguments: dict[str, Any], *, cwd: str, allow: Iterable[str] = (), deny: Iterable[str] = ()) -> tuple[Verdict, str]:
    """The verdict on one tool call Claude wants to make, and why."""
    if name == "Bash":
        return bash(str(arguments.get("command") or ""), cwd=cwd, allow=allow, deny=deny)
    path = arguments.get("file_path") or arguments.get("notebook_path") or arguments.get("path")
    if isinstance(path, str) and path:
        absolute = path if os.path.isabs(os.path.expanduser(path)) else os.path.join(cwd, path)
        if private(absolute, cwd):
            return "deny", f"{path} holds credentials"
        if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            inside = _expand(absolute)
            if not inside.startswith(_expand(cwd).rstrip("/") + "/") and not _claude_plan(inside):
                return "ask", f"edits {path}, outside the task's worktree"
    return "allow", ""


def describe(name: str, arguments: dict[str, Any]) -> str:
    """The step in a sentence, for the person who decides."""
    if name == "Bash":
        return "Run: " + redact.text(str(arguments.get("command") or ""))[:400]
    path = arguments.get("file_path") or arguments.get("notebook_path") or arguments.get("path")
    if path:
        return f"{name} {path}"
    if arguments.get("url"):
        return f"{name} {arguments['url']}"
    return f"Use {name}"
