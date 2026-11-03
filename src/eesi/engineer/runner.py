"""The runner: the API's tasks in, Claude Code at work, what happened back out.

Three loops share one connection to the API:

- serving: a long poll for what to do (a task to start, a task someone
  cancelled, a finished branch someone asked to publish), each task run in
  the background through ``claude.TaskRun``, up to ``max_parallel`` at once;
- syncing: every minute and a half, the Claude Code sessions that changed in
  the runner's repositories (``sessions.Discovery``), and every five minutes
  their recent commits;
- registering: every minute, what the runner is and where it works.

A connection that drops is retried with a pause; a task's run never raises,
so one failing task never takes the runner down.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from .._version import __version__
from . import git
from .claude import Outcome, TaskRun
from .config import Config
from .sessions import Discovery

SYNC_SECONDS = 90
COMMITS_SECONDS = 300
REGISTER_SECONDS = 60
RETRY_SECONDS = 5


class RunnerApi:
    """The runner's side of ``/v1/engineering/runners/{id}/...``."""

    def __init__(self, nur: Any, runner_id: str) -> None:
        self.nur = nur
        self.base = f"/engineering/runners/{runner_id}"

    async def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self.nur._request(method, f"{self.base}{path}", **kwargs)
        return dict(response.json()) if response.content else {}

    async def register(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._json("PUT", "", json=body)

    async def next(self, *, holding: list[str], wait: float) -> dict[str, Any]:
        return await self._json("POST", "/next", json={"holding": holding, "wait": wait})

    async def sync(self, *, sessions: list[dict[str, Any]], commits: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._json("POST", "/sync", json={"sessions": sessions, "commits": commits})

    async def report(self, task_id: str, **fields: Any) -> dict[str, Any]:
        return await self._json("POST", f"/tasks/{task_id}/report", json={k: v for k, v in fields.items() if v is not None})

    async def finish(self, task_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._json("POST", f"/tasks/{task_id}/finish", json=body)

    async def ask(self, task_id: str, **fields: Any) -> dict[str, Any]:
        return await self._json("POST", f"/tasks/{task_id}/approvals", json=fields)

    async def decision(self, approval_id: str, *, wait: float) -> dict[str, Any]:
        return await self._json("GET", f"/approvals/{approval_id}", params={"wait": wait})

    async def published(self, task_id: str, *, approval_id: str, pr_url: str | None, error: str | None) -> dict[str, Any]:
        return await self._json(
            "POST", f"/tasks/{task_id}/published", json={"approval_id": approval_id, "pr_url": pr_url, "error": error}
        )


def claude_version(config: Config) -> str | None:
    command = config.claude_path or shutil.which("claude")
    if not command:
        return None
    try:
        done = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip().split(" ")[0][:40] or None


def describe(config: Config) -> dict[str, Any]:
    """What the runner tells the API it is."""
    return {
        "name": config.name,
        "platform": f"{platform.system()} {platform.machine()}"[:80],
        "claude_version": claude_version(config),
        "version": __version__,
        "accepts": config.accepts,
        "max_parallel": config.max_parallel,
        "repos": [
            {
                "name": repo.name,
                "path": repo.path,
                "remote": git.remote_url(repo.path),
                "default_branch": git.default_branch(repo.path),
                "branch": git.current_branch(repo.path),
                "checks": repo.checks,
            }
            for repo in config.repos
        ],
    }


class Runner:
    def __init__(
        self,
        config: Config,
        *,
        echo: Callable[[str], None] = print,
        task_run: Callable[..., TaskRun] = TaskRun,
    ) -> None:
        self.config = config
        self.echo = echo
        self.task_run = task_run
        self.running: dict[str, TaskRun] = {}
        self._background: set[asyncio.Task[Any]] = set()
        self.discovery = Discovery(config.home)

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def run(self) -> None:
        from ..client import NurClient

        async with NurClient(self.config.api_key, base_url=self.config.api_url, http_timeout=40) as nur:
            api = RunnerApi(nur, self.config.id)
            state = await api.register(describe(self.config))
            repos = ", ".join(repo.name for repo in self.config.repos) or "no repositories"
            self.echo(f"{self.config.name} is ready for Nur's tasks in {repos} (budget ${state['settings']['max_budget_usd']:.2f} a task).")
            await asyncio.gather(self._serve(api), self._syncing(api), self._registering(api))

    async def _serve(self, api: RunnerApi) -> None:
        while True:
            try:
                work = await api.next(holding=list(self.running), wait=20)
            except Exception as error:  # noqa: BLE001 - the next poll retries
                self.echo(f"Could not reach the API ({error}); trying again.")
                await asyncio.sleep(RETRY_SECONDS)
                continue
            for task_id in work.get("stop") or []:
                run = self.running.get(task_id)
                if run is not None:
                    self.echo(f"Stopping {run.task['ref']}: it was cancelled.")
                    run.stop()
            for task in work.get("tasks") or []:
                self._spawn(self.execute(api, task))
            for request in work.get("publish") or []:
                self._spawn(self.publish(api, request))

    async def execute(self, api: RunnerApi, task: dict[str, Any]) -> Outcome:
        repo = self.config.repo(task.get("repo"))
        run = self.task_run(task, repo, self.config, api, echo=self.echo)
        self.running[task["id"]] = run
        self.echo(f"{task['ref']} started: {task['title']}")
        try:
            outcome = await run.run()
            body = outcome.as_finish()
            body["events"] = run._events[:200]
            for attempt in range(5):
                try:
                    await api.finish(task["id"], body)
                    break
                except Exception as error:  # noqa: BLE001 - the result is worth retrying for
                    if attempt == 4:
                        self.echo(f"Could not report {task['ref']}: {error}")
                    await asyncio.sleep(RETRY_SECONDS * (attempt + 1))
            verdict = "done" if outcome.ok else f"could not finish ({outcome.error})"
            self.echo(f"{task['ref']} {verdict}: {len(outcome.diff_stat)} files changed, {len(outcome.checks)} checks.")
            return outcome
        finally:
            self.running.pop(task["id"], None)

    async def publish(self, api: RunnerApi, request: dict[str, Any]) -> None:
        folder, branch = request.get("cwd"), request.get("branch")
        if not folder or not branch:
            await api.published(request["task_id"], approval_id=request["approval_id"], pr_url=None, error="the task has no branch here")
            return
        body = (
            f"{request.get('objective') or ''}\n\n## What Claude reports\n\n{request.get('report') or '(no report)'}\n\n"
            f"Handed to Claude through Nur as {request['ref']}; published on a person's request."
        )
        link, error = await asyncio.to_thread(git.push_and_open_pr, folder, branch, request.get("title") or branch, body)
        await api.published(request["task_id"], approval_id=request["approval_id"], pr_url=link, error=error)
        self.echo(f"{request['ref']}: {'pull request ' + link if link else 'not published: ' + str(error)}")

    async def sync_once(self, api: RunnerApi, *, commits: bool = True) -> dict[str, Any]:
        sessions: list[dict[str, Any]] = []
        if self.config.sessions:
            try:
                sessions = await asyncio.to_thread(self.discovery.changed, self.config.repos)
            except ImportError:
                sessions = []
        recent: list[dict[str, Any]] = []
        if commits:
            for repo in self.config.repos:
                recent.extend(await asyncio.to_thread(git.recent_commits, repo.path, repo.name))
        counts: dict[str, Any] = {"sessions": 0, "commits": 0}
        for start in range(0, max(len(sessions), 1), 50):
            batch = sessions[start : start + 50]
            if not batch and start:
                break
            counts = await api.sync(sessions=batch, commits=recent if start == 0 else [])
        self.discovery.commit()
        return {"sessions": len(sessions), "commits": len(recent), **{f"api_{k}": v for k, v in counts.items()}}

    async def _syncing(self, api: RunnerApi) -> None:
        last_commits = 0.0
        while True:
            try:
                due = time.monotonic() - last_commits >= COMMITS_SECONDS
                await self.sync_once(api, commits=due)
                if due:
                    last_commits = time.monotonic()
            except Exception as error:  # noqa: BLE001 - the next round retries
                self.echo(f"Could not sync sessions ({error}).")
            await asyncio.sleep(SYNC_SECONDS)

    async def _registering(self, api: RunnerApi) -> None:
        while True:
            await asyncio.sleep(REGISTER_SECONDS)
            try:
                await api.register(await asyncio.to_thread(describe, self.config))
            except Exception as error:  # noqa: BLE001
                self.echo(f"Could not refresh the runner ({error}).")
