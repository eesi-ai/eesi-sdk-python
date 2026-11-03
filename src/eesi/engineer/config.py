"""The runner's settings: ``~/.eesi/engineer.json``, readable by you only (it holds the key).

``api_key`` and ``api_url`` (``EESI_API_KEY`` / ``EESI_API_URL`` win), the
runner's ``id`` (made once) and ``name``, the ``repos`` it works in (a name,
the folder, the ``checks`` to run after each task: tests, lint, and the
``links``: files and folders git does not track, like ``.venv``,
``node_modules`` or a test ``.env``, linked into each task's worktree so it
runs like the checkout),
``accepts`` (``org``: anyone's tasks in the organization; ``mine``: only the
key owner's), ``max_parallel`` tasks at once, ``home`` (where task worktrees
go), ``claude_path`` (the Claude Code command; the one on your PATH, else the
Agent SDK's own), ``allow``/``deny`` (command prefixes added to the
built-in policy, ``policy.py``) and ``unattended`` (nobody watches: a step
the policy would ask a person about goes ahead by itself and is logged on the
task; what the policy never allows stays refused).
"""

from __future__ import annotations

import json
import os
import platform
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HOME = Path.home() / ".eesi" / "engineer"


def path() -> Path:
    override = os.environ.get("EESI_ENGINEER_CONFIG")
    return Path(override) if override else Path.home() / ".eesi" / "engineer.json"


@dataclass
class Repo:
    name: str
    path: str
    checks: list[str] = field(default_factory=list)
    #: Untracked paths, relative to the repository, linked into each task's worktree.
    links: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "checks": list(self.checks), "links": list(self.links)}


@dataclass
class Config:
    id: str
    name: str
    api_key: str | None = None
    api_url: str | None = None
    repos: list[Repo] = field(default_factory=list)
    accepts: str = "org"
    max_parallel: int = 2
    home: str = str(DEFAULT_HOME)
    claude_path: str | None = None
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    #: Report the Claude Code sessions found in these repositories.
    sessions: bool = True
    #: Nobody watches this runner: "ask a person" steps go ahead by themselves.
    unattended: bool = False

    @classmethod
    def load(cls) -> Config:
        try:
            data = json.loads(path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        repos = [
            Repo(
                name=str(item.get("name") or Path(str(item.get("path") or "")).name),
                path=str(item.get("path") or ""),
                checks=[str(command) for command in item.get("checks") or []],
                links=[str(link) for link in item.get("links") or []],
            )
            for item in data.get("repos") or []
            if isinstance(item, dict) and item.get("path")
        ]
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            name=str(data.get("name") or platform.node() or "runner")[:80],
            api_key=os.environ.get("EESI_API_KEY") or data.get("api_key"),
            api_url=os.environ.get("EESI_API_URL") or data.get("api_url"),
            repos=repos,
            accepts=data.get("accepts") if data.get("accepts") in ("org", "mine") else "org",
            max_parallel=max(1, min(int(data.get("max_parallel") or 2), 6)),
            home=str(data.get("home") or DEFAULT_HOME),
            claude_path=data.get("claude_path") or None,
            allow=[str(item) for item in data.get("allow") or []],
            deny=[str(item) for item in data.get("deny") or []],
            sessions=data.get("sessions", True) is not False,
            unattended=data.get("unattended") is True,
        )

    def save(self) -> Path:
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id,
            "name": self.name,
            "api_url": self.api_url,
            "repos": [repo.as_dict() for repo in self.repos],
            "accepts": self.accepts,
            "max_parallel": self.max_parallel,
            "home": self.home,
            "claude_path": self.claude_path,
            "allow": self.allow,
            "deny": self.deny,
            "sessions": self.sessions,
            "unattended": self.unattended,
        }
        # The key is kept only when it did not come from the environment.
        if self.api_key and self.api_key != os.environ.get("EESI_API_KEY"):
            data["api_key"] = self.api_key
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        os.chmod(target, 0o600)
        return target

    def repo(self, name: str | None) -> Repo | None:
        if not self.repos:
            return None
        if not name:
            return self.repos[0] if len(self.repos) == 1 else None
        wanted = name.lower()
        return next((repo for repo in self.repos if repo.name.lower() == wanted), None)


def repo_name(folder: str) -> str:
    """A repository's name: its folder's, made safe to say and type."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(folder).resolve().name).strip("-") or "repo"
