"""The code line's listener: what a call hears, and how asked-for work gets done.

The agents are stand-in scripts that print what the real ones stream
(``claude -p --output-format stream-json``, ``codex exec --json``), so the
parsing of steps, session ids and results runs for real.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import run_async
from eesi import companion
from eesi.client import PhoneCall
from eesi.companion import Settings
from eesi.listener import Listener, Task

CLAUDE = """#!{python}
import json, sys
argv = sys.argv[1:]
with open({log!r}, "a") as out:
    out.write(json.dumps(argv) + "\\n")
session = argv[argv.index("--resume") + 1] if "--resume" in argv else "s-new"
for event in [
    {{"type": "system", "subtype": "init", "session_id": session}},
    {{"type": "assistant", "session_id": session, "message": {{"content": [
        {{"type": "text", "text": "I'll add the page."}},
        {{"type": "tool_use", "name": "Write", "input": {{"file_path": "/repo/web/login.html"}}}},
        {{"type": "tool_use", "name": "Bash", "input": {{"command": "python -m pytest -q"}}}},
    ]}}}},
    {{"type": "result", "subtype": "success", "is_error": False,
      "result": "Added the login page; 4 tests pass.", "session_id": session}},
]:
    print(json.dumps(event), flush=True)
"""

CODEX = """#!{python}
import json, sys
argv = sys.argv[1:]
report = argv[argv.index("--output-last-message") + 1]
for event in [
    {{"type": "thread.started", "thread_id": "th-1"}},
    {{"type": "item.completed", "item": {{"type": "command_execution", "command": "npm test"}}}},
    {{"type": "item.completed", "item": {{"type": "file_change", "changes": [{{"path": "/repo/src/app.ts"}}]}}}},
]:
    print(json.dumps(event), flush=True)
open(report, "w").write("Wired the app; npm test passes.")
"""

BROKEN = """#!{python}
import sys
print("Failed to authenticate: OAuth session expired", file=sys.stderr)
sys.exit(1)
"""


def _agent(tmp_path: Path, monkeypatch, name: str, source: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / f"{name}.log"
    script = bin_dir / name
    script.write_text(source.format(python=sys.executable, log=str(log)))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    return log


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "demo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "app.py").write_text("print('hi')\n")
    return repo


def _listener(tmp_path, monkeypatch, *, agent="claude", **settings) -> Listener:
    monkeypatch.setattr(companion, "HOME", tmp_path / "home")
    return Listener(
        Settings(phone="+14155550123", name="Sam", **settings),
        str(_repo(tmp_path)),
        agent=agent,
        echo=lambda line: None,
    )


def test_an_idle_listener_asks_what_to_build(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch)

    instructions, greeting = listener.brief()

    assert (
        greeting
        == "Hi Sam, it's Nur. Nothing is running in demo. What should Claude Code build?"
    )
    assert instructions.startswith("You are Nur, Sam's coding companion, on the phone")
    assert "Nothing is running right now." in instructions
    assert "repeat back exactly what you will hand to the agent" in instructions


def test_a_call_mid_task_hears_its_latest_steps_and_the_queue(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch)
    running = Task(text="Add a login page", started_at=time.time() - 240)
    running.step("edited login.html")
    running.step("ran python -m pytest -q")
    done = Task(
        text="Set up the repo",
        started_at=time.time() - 900,
        finished_at=time.time() - 600,
        ok=True,
        report="Created the app skeleton.",
    )
    listener.tasks = [done, running, Task(text="Add a logout button")]

    instructions, greeting = listener.brief()

    assert greeting == (
        "Hi Sam, it's Nur. Claude Code is working on Add a login page, 4 minutes in. "
        "What would you like?"
    )
    assert (
        'Running now, for 4 minutes: "Add a login page". Its latest steps: '
        "edited login.html; ran python -m pytest -q." in instructions
    )
    assert 'Queued after it: "Add a logout button".' in instructions
    assert (
        'Done 10 minutes ago (finished): "Set up the repo". What the agent said: '
        "Created the app skeleton." in instructions
    )


@run_async
async def test_tasks_run_in_one_claude_session_with_their_steps(tmp_path, monkeypatch):
    log = _agent(tmp_path, monkeypatch, "claude", CLAUDE)
    listener = _listener(tmp_path, monkeypatch)
    first, second = Task(text="Add a login page"), Task(text="Add a logout button")

    await listener.work(first)
    await listener.work(second)

    assert (first.ok, first.report) == (True, "Added the login page; 4 tests pass.")
    assert first.activity == [
        "said: I'll add the page.",
        "edited login.html",
        "ran python -m pytest -q",
    ]
    runs = [json.loads(line) for line in log.read_text().splitlines()]
    assert runs[0][:3] == ["-p", "--output-format", "stream-json"]
    assert runs[0][-1] == "From a phone call with the developer: Add a login page"
    # The second task resumes the session the first one started.
    assert runs[1][:3] == ["-p", "--resume", "s-new"]
    assert listener.session_id == "s-new"


@run_async
async def test_codex_tasks_report_through_their_last_message(tmp_path, monkeypatch):
    _agent(tmp_path, monkeypatch, "codex", CODEX)
    listener = _listener(tmp_path, monkeypatch, agent="codex")
    task = Task(text="Wire the app")

    await listener.work(task)

    assert (task.ok, task.report) == (True, "Wired the app; npm test passes.")
    assert task.activity == ["ran npm test", "changed app.ts"]
    assert listener.session_id == "th-1"


@run_async
async def test_an_agent_that_cannot_run_says_why(tmp_path, monkeypatch):
    _agent(tmp_path, monkeypatch, "claude", BROKEN)
    listener = _listener(tmp_path, monkeypatch)
    task = Task(text="Add a login page")

    await listener.work(task)

    assert (task.ok, task.report) == (
        False,
        "Failed to authenticate: OAuth session expired",
    )
    _, greeting = listener.brief(done=task)
    assert greeting == (
        "Hi Sam, it's Nur. Claude Code couldn't finish Add a login page: "
        "Failed to authenticate: OAuth session expired."
    )


class _Nur:
    def __init__(self, transcript):
        self.transcript = transcript
        self.calls: list[dict] = []

    async def wait_for_call(self, call_id, **kwargs):
        return PhoneCall(
            id=call_id,
            session_id="sess",
            status="completed",
            to="+14155550123",
            created_at=time.time(),
            transcript=self.transcript,
        )

    async def chat(self, messages, **kwargs):
        assert "Developer: Build me a login page" in messages[-1]["content"]
        return '{"follow_ups": ["Add a login page with email and password."]}'

    async def call(self, to, **options):
        self.calls.append({"to": to, **options})
        return PhoneCall(
            id="call-out", session_id="s", status="dialing", to=to, created_at=0
        )


@run_async
async def test_what_a_call_asks_for_is_queued(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch)
    nur = _Nur(
        (
            ("nur", "Hi Sam, it's Nur. Nothing is running in demo."),
            ("person", "Build me a login page, please."),
            (
                "nur",
                "I'll have Claude add a login page with email and password. Right?",
            ),
            ("person", "Yes."),
        )
    )

    tasks = await listener.take(nur, "call-1")

    assert [task.text for task in tasks] == [
        "Add a login page with email and password."
    ]
    assert listener.queue.qsize() == 1 and listener.changed.is_set()


@run_async
async def test_a_finished_task_calls_with_the_result(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch)
    nur = _Nur(())
    task = Task(
        text="Add a login page",
        started_at=time.time() - 60,
        finished_at=time.time(),
        ok=True,
        report="Added the login page; 4 tests pass.",
    )
    listener.tasks = [task]

    await listener.call_about(nur, task)

    (call,) = nur.calls
    assert call["to"] == "+14155550123"
    assert call["greeting"] == (
        "Hi Sam, it's Nur. Claude Code finished Add a login page: Added the login "
        "page; 4 tests pass. Want the details, or what's next?"
    )


@pytest.mark.parametrize("agent", ["cursor"])
def test_listen_knows_its_agents(tmp_path, monkeypatch, agent, capsys):
    monkeypatch.setattr(companion, "HOME", tmp_path / "home")

    assert companion.main(["listen", "--agent", agent]) == 1
    assert "claude or codex" in capsys.readouterr().out


def test_a_text_to_the_code_line_gets_the_same_facts(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch)
    listener.tasks = [Task(text="Add a login page", started_at=time.time() - 120)]

    texting = listener.texting()

    assert texting.startswith("You are Nur, Sam's coding companion, texting with Sam")
    assert 'Running now, for 2 minutes: "Add a login page".' in texting


@run_async
async def test_what_a_text_asks_for_is_queued(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch)

    class Nur(_Nur):
        async def chat(self, messages, **kwargs):
            assert "Developer: Build me a login page" in messages[-1]["content"]
            return '{"follow_ups": ["Add a login page with email and password."]}'

    tasks = await listener.take_text(
        Nur(()),
        "Build me a login page please",
        "On it: I'll have Claude Code add a login page with email and password.",
    )

    assert [task.text for task in tasks] == [
        "Add a login page with email and password."
    ]
    assert listener.queue.qsize() == 1


@run_async
async def test_a_finished_task_can_be_texted(tmp_path, monkeypatch):
    listener = _listener(tmp_path, monkeypatch, notify="text")
    sent: list = []

    class Nur:
        async def send_text(self, to, text):
            sent.append((to, text))
            return {"id": "msg-1"}

    task = Task(
        text="Add a login page",
        started_at=time.time() - 60,
        finished_at=time.time(),
        ok=True,
        report="Added the login page; 4 tests pass.",
    )

    await listener.text_about(Nur(), task)

    assert sent == [
        (
            "+14155550123",
            "Claude Code finished Add a login page: Added the login page; 4 tests pass.",
        )
    ]
