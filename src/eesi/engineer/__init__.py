"""Nur as your team's engineering assistant: you talk with Nur, Claude Code does the work.

``eesi engineer`` runs on a computer where your code and Claude Code live::

    eesi engineer setup --api-key sk-eesi-... --repo ~/code/app --check "make test" --link .venv
    eesi engineer start          # take Nur's tasks; report sessions and commits
    eesi engineer sync           # report sessions and commits once, then stop
    eesi engineer status         # the settings (the key never shown) and the API's view

Hand Claude work by talking to Nur in the EESI console or the app ("ask Claude
to fix the login redirect, plan it first"), or from the console's Engineering
page. The runner claims the task, runs Claude Code through the Claude Agent
SDK in a git worktree of its own on a ``nur/`` branch, sends each step as it
happens, asks a person before anything outside a task's reach
(``policy.py``), and when it is done reads the diff from git, runs your
checks, and has Claude report what it did, what is left and why it decided
what it did. Nur tells you, and remembers.

It also lists the Claude Code sessions in your repositories (the Agent SDK's
session listing) and their commits, so Nur can tell you what happened in the
sessions you ran yourself. Only digests leave your computer, redacted.

Needs the Claude Agent SDK: ``pip install "eesi-sdk[engineer]"`` (with uv:
``uv tool install "eesi-sdk[engineer] @ git+https://github.com/eesi-ai/eesi-sdk-python"``),
and Claude Code signed in on this computer.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from typing import Any

from .config import Config, Repo, path, repo_name

__all__ = ["main"]


def _options(argv: list[str], name: str) -> list[str]:
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif argument.startswith(name + "="):
            values.append(argument[len(name) + 1 :])
    return values


def _option(argv: list[str], name: str) -> str | None:
    values = _options(argv, name)
    return values[-1] if values else None


def _sdk_missing() -> str | None:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return (
            "The Claude Agent SDK is not installed: pip install \"eesi-sdk[engineer]\" "
            "(or pip install claude-agent-sdk)."
        )
    return None


def _setup(argv: list[str]) -> int:
    from . import git

    config = Config.load()
    if key := _option(argv, "--api-key"):
        config.api_key = key
    if url := _option(argv, "--api-url"):
        config.api_url = url
    if name := _option(argv, "--name"):
        config.name = name[:80]
    if accepts := _option(argv, "--accepts"):
        if accepts not in ("org", "mine"):
            print("--accepts is org (anyone's tasks in your organization) or mine (only yours).")
            return 1
        config.accepts = accepts
    if parallel := _option(argv, "--max-parallel"):
        config.max_parallel = max(1, min(int(parallel), 6))
    if "--unattended" in argv:
        config.unattended = True
    if "--attended" in argv:
        config.unattended = False
    checks = _options(argv, "--check")
    links = _options(argv, "--link")
    for spec in _options(argv, "--repo"):
        folder, _, label = spec.partition("=")
        folder = os.path.abspath(os.path.expanduser(folder))
        top = git.toplevel(folder)
        if top is None:
            print(f"{folder} is not a git repository.")
            return 1
        repo = Repo(name=label or repo_name(top), path=top, checks=checks, links=links)
        config.repos = [existing for existing in config.repos if existing.path != top] + [repo]
    if checks and not _options(argv, "--repo") and config.repos:
        config.repos[-1].checks = checks
    if links and not _options(argv, "--repo") and config.repos:
        config.repos[-1].links = links
    saved = config.save()
    print(f"Saved {saved}.")
    if not config.api_key:
        print("Add your EESI API key (the API keys page of your EESI console): eesi engineer setup --api-key sk-eesi-...")
        return 1
    if not config.repos:
        print("Add a repository: eesi engineer setup --repo ~/code/app --check \"make test\"")
        return 1
    if missing := _sdk_missing():
        print(missing)
    if not (config.claude_path or shutil.which("claude")):
        print("Claude Code is not on your PATH; the Agent SDK's own copy will run (sign it in with `claude` first).")
    try:
        asyncio.run(_register(config))
    except Exception as error:  # noqa: BLE001
        print(f"The API did not take the runner: {error}")
        return 1
    print(
        f"{config.name} is registered for {', '.join(repo.name for repo in config.repos)}. "
        "Start it with: eesi engineer start"
    )
    return 0


async def _register(config: Config) -> dict[str, Any]:
    from ..client import NurClient
    from .runner import RunnerApi, describe

    async with NurClient(config.api_key, base_url=config.api_url) as nur:
        return await RunnerApi(nur, config.id).register(await asyncio.to_thread(describe, config))


def _start(argv: list[str]) -> int:
    from .runner import Runner

    del argv
    config = Config.load()
    if not config.api_key or not config.repos:
        print("Set it up first: eesi engineer setup --api-key sk-eesi-... --repo ~/code/app")
        return 1
    if missing := _sdk_missing():
        print(missing)
        return 1
    config.save()  # keeps the id it was given
    try:
        asyncio.run(Runner(config).run())
    except KeyboardInterrupt:
        print("Stopped. Tasks still running were left as they were; Nur sees them stop.", file=sys.stderr)
    return 0


def _sync(argv: list[str]) -> int:
    from ..client import NurClient
    from .runner import Runner, RunnerApi, describe

    del argv
    config = Config.load()
    if not config.api_key or not config.repos:
        print("Set it up first: eesi engineer setup --api-key sk-eesi-... --repo ~/code/app")
        return 1
    config.save()

    async def once() -> dict[str, Any]:
        async with NurClient(config.api_key, base_url=config.api_url, http_timeout=60) as nur:
            api = RunnerApi(nur, config.id)
            await api.register(await asyncio.to_thread(describe, config))
            return await Runner(config).sync_once(api)

    print(json.dumps(asyncio.run(once()), indent=2))
    return 0


def _status(argv: list[str]) -> int:
    del argv
    config = Config.load()
    report: dict[str, Any] = {
        "settings": str(path()),
        "id": config.id,
        "name": config.name,
        "api_key": "set" if config.api_key else None,
        "api_url": config.api_url or "https://api.eesi.ai",
        "repos": [repo.as_dict() for repo in config.repos],
        "accepts": config.accepts,
        "max_parallel": config.max_parallel,
        "home": config.home,
        "claude": config.claude_path or shutil.which("claude") or "the Agent SDK's own",
        "agent_sdk": _sdk_missing() or "installed",
    }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "help"
    rest = argv[1:]
    if command == "setup":
        return _setup(rest)
    if command == "start":
        return _start(rest)
    if command == "sync":
        return _sync(rest)
    if command == "status":
        return _status(rest)
    print(__doc__)
    return 0
