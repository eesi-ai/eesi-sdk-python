"""Git for the runner: a worktree per task, the diff as git computes it, recent commits.

Every task gets its own worktree and branch (``nur/t12-fix-the-login``) under
the runner's home, based on the repository's default branch as the remote has
it, so a task never touches the checkout you are working in and two tasks
never touch each other. What a task changed is read from git afterwards,
never from what Claude says it changed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import redact

BRANCH_PREFIX = "nur/"
#: The longest a patch sent to the API may be.
PATCH_LIMIT = 400_000


class GitError(RuntimeError):
    pass


def run(cwd: str | Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def out(cwd: str | Path, *args: str, timeout: float = 60) -> str:
    try:
        done = run(cwd, *args, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def toplevel(folder: str | Path) -> str | None:
    return out(folder, "rev-parse", "--show-toplevel") or None


def remote_url(repo: str | Path) -> str:
    return out(repo, "remote", "get-url", "origin")


def current_branch(folder: str | Path) -> str:
    return out(folder, "branch", "--show-current")


def head(folder: str | Path) -> str:
    return out(folder, "rev-parse", "HEAD")


def default_branch(repo: str | Path) -> str:
    """The branch work lands on: origin's HEAD, else main or master, else the current one."""
    symbolic = out(repo, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD")
    if symbolic.startswith("refs/remotes/origin/"):
        return symbolic[len("refs/remotes/origin/") :]
    for name in ("main", "master"):
        if out(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}"):
            return name
    return current_branch(repo) or "main"


def slug(text: str, limit: int = 40) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return words[:limit].rstrip("-") or "task"


def inside(links: Sequence[str]) -> list[str]:
    """The links that name a path inside the repository (never ``..`` or absolute)."""
    return [link for link in links if link and not Path(link).is_absolute() and ".." not in Path(link).parts]


def link_environment(repo: str | Path, folder: str | Path, links: Sequence[str]) -> list[str]:
    """Link the repository's untracked environment (``.venv``, ``node_modules``,
    a test ``.env``) into a worktree, so its checks and Claude's commands run as
    they do in the checkout. Only paths inside the repository that exist there
    and are missing here; returns the ones linked."""
    repo, folder = Path(repo).resolve(), Path(folder)
    linked: list[str] = []
    for link in inside(links):
        relative = Path(link)
        source, target = repo / relative, folder / relative
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=source.is_dir())
        linked.append(str(relative))
    return linked


def create_worktree(
    repo: str | Path,
    home: str | Path,
    ref: str,
    title: str,
    *,
    start: str | None = None,
    prefix: str = BRANCH_PREFIX,
    links: Sequence[str] = (),
) -> tuple[str, str, str]:
    """A fresh worktree and branch for a task: from *start* (a branch) when it
    exists, else from the default branch as origin has it, with the
    repository's *links* in it.

    Returns (folder, branch, base commit).
    """
    repo = Path(repo)
    default = default_branch(repo)
    # The newest of the default branch, without touching the checkout itself.
    try:
        run(repo, "fetch", "--quiet", "origin", default, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        pass
    candidates = ([start, f"origin/{start}"] if start else []) + [f"origin/{default}", default, "HEAD"]
    base = next(
        (
            candidate
            for candidate in candidates
            if out(repo, "rev-parse", "--verify", "--quiet", candidate)
        ),
        "HEAD",
    )
    base_commit = out(repo, "rev-parse", base)
    branch = f"{prefix}{ref}-{slug(title)}"
    folder = Path(home) / "worktrees" / repo.name / ref
    if folder.exists():
        # A task that ran here before (a retry): keep its branch and work.
        if (folder / ".git").exists():
            link_environment(repo, folder, links)
            return str(folder), current_branch(folder) or branch, base_commit
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    existing = out(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    args = ["worktree", "add", str(folder), branch] if existing else ["worktree", "add", "-b", branch, str(folder), base]
    done = run(repo, *args, timeout=120)
    if done.returncode != 0:
        raise GitError(f"could not make a worktree: {(done.stderr or done.stdout).strip()[:300]}")
    link_environment(repo, folder, links)
    return str(folder), branch, base_commit


def at_commit(folder: str | Path, commit: str, place: str | Path, links: Sequence[str] = ()) -> str | None:
    """A throwaway worktree of *folder*'s repository at *commit*, with the same
    *links* as the task's: where a check runs as it was before."""
    place = Path(place)
    if place.exists():
        shutil.rmtree(place, ignore_errors=True)
    place.parent.mkdir(parents=True, exist_ok=True)
    done = run(folder, "worktree", "add", "--detach", str(place), commit, timeout=120)
    if done.returncode != 0:
        return None
    top = toplevel(folder)
    if top and links:
        # The task's worktree links to the repository's own environment.
        common = out(folder, "rev-parse", "--path-format=absolute", "--git-common-dir")
        link_environment(Path(common).parent if common else top, place, links)
    return str(place)


def drop_worktree(folder: str | Path, place: str | Path) -> None:
    run(folder, "worktree", "remove", "--force", str(place), timeout=60)
    shutil.rmtree(place, ignore_errors=True)


#: What running code leaves behind, never part of a task's change: committed
#: without these even in a repository whose .gitignore does not name them.
#: What every pull request the runner opens begins with, so the team can tell
#: the self-improvement loop's work from a person's (the API's
#: ``engineering.tasks.PR_PREFIX``, kept the same by its tests).
PR_PREFIX = "[eesi-engineer]"

#: Where a task's evidence goes in its worktree: files the API hands over for
#: Claude to read (for a learning task, what the bench recorded around its
#: problem's moments), never committed.
EVIDENCE_DIR = ".nur-evidence"
#: How much evidence a task may bring, in bytes.
EVIDENCE_LIMIT = 2_000_000

NEVER_COMMITTED = (
    f"{EVIDENCE_DIR}/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/node_modules/**",
    "**/.next/**",
    "**/.turbo/**",
    "**/.DS_Store",
)


def hand_over(folder: str | Path, files: dict[str, Any]) -> list[str]:
    """Put a task's evidence in its worktree (``EVIDENCE_DIR``), in place of
    what an earlier run of it left: only plain names inside that folder, up to
    ``EVIDENCE_LIMIT``. The names written."""
    root = Path(folder) / EVIDENCE_DIR
    if root.is_symlink():
        return []
    shutil.rmtree(root, ignore_errors=True)
    written: list[str] = []
    total = 0
    for name, text in sorted((files or {}).items()):
        relative = Path(str(name))
        if not isinstance(text, str) or relative.is_absolute() or not relative.parts or ".." in relative.parts:
            continue
        total += len(text.encode())
        if total > EVIDENCE_LIMIT:
            break
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append(relative.as_posix())
    return written


def commit_all(folder: str | Path, message: str, *, links: Sequence[str] = ()) -> str | None:
    """Commit everything the task changed on its branch; None when it changed
    nothing. Never the environment linked in (a ``.venv/`` rule in .gitignore
    does not match a symlink called .venv)."""
    run(
        folder,
        "add",
        "-A",
        "--",
        ".",
        *(f":(exclude,glob){pattern}" for pattern in NEVER_COMMITTED),
        *(f":(exclude,literal){link}" for link in inside(links)),
    )
    if not out(folder, "diff", "--cached", "--name-only"):
        return None
    done = run(
        folder,
        "-c",
        "user.useConfigOnly=false",
        "commit",
        "--no-verify",
        "-q",
        "-m",
        message,
        timeout=120,
    )
    if done.returncode != 0:
        raise GitError(f"could not commit: {(done.stderr or done.stdout).strip()[:300]}")
    return head(folder)


def diff_stat(folder: str | Path, base: str) -> list[dict[str, Any]]:
    """Each file the branch changed since *base*: lines added and removed, and how."""
    statuses: dict[str, str] = {}
    for line in out(folder, "diff", "--name-status", "-M", base, "HEAD").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            code = parts[0][:1]
            statuses[parts[-1]] = {
                "A": "added",
                "D": "deleted",
                "R": "renamed",
                "M": "modified",
            }.get(code, "modified")
    stat: list[dict[str, Any]] = []
    for line in out(folder, "diff", "--numstat", "-M", base, "HEAD").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name = parts[2]
        if " => " in name:
            name = re.sub(r"\{(.*?) => (.*?)\}", r"\2", name).split(" => ")[-1]
        stat.append(
            {
                "path": name,
                "added": int(parts[0]) if parts[0].isdigit() else 0,
                "removed": int(parts[1]) if parts[1].isdigit() else 0,
                "status": statuses.get(name, "modified"),
            }
        )
    return stat


def diff_patch(folder: str | Path, base: str, *, limit: int = PATCH_LIMIT) -> str:
    """The branch's unified diff since *base*, secrets and secret files left out."""
    patch = out(folder, "diff", "-M", "--no-color", base, "HEAD", timeout=120)
    patch = redact.diff(patch)
    if len(patch) > limit:
        patch = patch[:limit] + "\n[the diff is longer; cut here]"
    return patch


def recent_commits(repo: str | Path, name: str, *, hours: int = 72, limit: int = 300) -> list[dict[str, Any]]:
    """Commits on any branch in the last *hours*, with their size."""
    raw = out(
        repo,
        "log",
        "--all",
        "--source",
        "--no-merges",
        f"--since={hours}.hours",
        f"--max-count={limit}",
        "--format=%x1e%H%x1f%an%x1f%aI%x1f%S%x1f%s",
        "--shortstat",
        timeout=60,
    )
    commits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in raw.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        header, _, stats = record.partition("\n")
        fields = header.split("\x1f")
        if len(fields) < 5 or fields[0] in seen:
            continue
        seen.add(fields[0])
        numbers = {
            key: int(value)
            for value, key in re.findall(r"(\d+) (file|insertion|deletion)", stats)
        }
        source = fields[3]
        branch = re.sub(r"^refs/(heads|remotes)/", "", source) if source else None
        commits.append(
            {
                "repo": name,
                "sha": fields[0],
                "author": fields[1][:120],
                "committed_at": fields[2],
                "branch": branch,
                "subject": redact.text(fields[4])[:300],
                "files": numbers.get("file", 0),
                "insertions": numbers.get("insertion", 0),
                "deletions": numbers.get("deletion", 0),
            }
        )
    return commits


def pr_title(title: str) -> str:
    """A pull request's title as the runner opens it: the team's prefix first, once."""
    title = title.strip()
    return title if title.startswith(PR_PREFIX) else f"{PR_PREFIX} {title}"


def push_and_open_pr(folder: str | Path, branch: str, title: str, body: str) -> tuple[str | None, str | None]:
    """Push the branch and open a draft pull request, its title beginning with
    ``PR_PREFIX``. Returns (link, error)."""
    pushed = run(folder, "push", "--no-verify", "-u", "origin", branch, timeout=180)
    if pushed.returncode != 0:
        return None, f"git push failed: {(pushed.stderr or pushed.stdout).strip()[-300:]}"
    gh = shutil.which("gh")
    if gh is None:
        return None, "pushed the branch, but the gh command is not installed to open the pull request"
    try:
        done = subprocess.run(
            [gh, "pr", "create", "--draft", "--head", branch, "--title", pr_title(title)[:200], "--body", body[:60_000]],
            cwd=str(folder),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"pushed the branch; gh pr create failed: {exc}"
    link = next((line.strip() for line in reversed(done.stdout.splitlines()) if line.strip().startswith("https://")), None)
    if done.returncode != 0 or not link:
        return None, f"pushed the branch; gh pr create failed: {(done.stderr or done.stdout).strip()[-300:]}"
    return link, None
