"""The coding companion: what it reads off a session, what it tells Nur, what it hands back."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import run_async
from eesi import companion
from eesi.client import PhoneCall, PhoneNumber
from eesi.companion import Digest, Settings


def _transcript(path: Path, cwd: Path, *, started: float) -> Path:
    stamp = datetime.fromtimestamp(started, tz=UTC).isoformat().replace("+00:00", "Z")
    events = [
        {
            "type": "user",
            "timestamp": stamp,
            "message": {
                "role": "user",
                "content": "Create hello.py that greets Nur, with a test.",
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll create the script."},
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": str(cwd / "hello.py"),
                            "content": "print('hi')",
                        },
                    },
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": str(cwd / "test_hello.py"),
                            "content": "...",
                        },
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "python -m pytest -q\necho done"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "is_error": True,
                        "content": "1 failed",
                    }
                ],
            },
        },
        {
            "type": "user",
            "isMeta": True,
            "message": {
                "role": "user",
                "content": "<command-name>/clear</command-name>",
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Done. hello.py prints a greeting and test_hello.py checks it; both tests pass.",
                    }
                ],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    return path


def test_a_claude_session_is_read_from_its_transcript(tmp_path):
    started = time.time() - 300
    digest = companion.read_claude_session(
        str(_transcript(tmp_path / "t.jsonl", tmp_path, started=started)),
        session_id="s-1",
        cwd=str(tmp_path),
    )

    assert digest.asked == ["Create hello.py that greets Nur, with a test."]
    assert digest.answer.startswith("Done. hello.py prints a greeting")
    assert digest.files == ["hello.py", "test_hello.py"]
    assert digest.commands == ["python -m pytest -q"]
    assert digest.errors == 1
    assert 4.5 < (digest.turn_minutes or 0) < 6


def test_a_codex_turn_is_read_from_its_notify_payload(tmp_path):
    digest = companion.read_codex_turn(
        {
            "type": "agent-turn-complete",
            "thread-id": "th-9",
            "input-messages": ["Rename the module."],
            "last-assistant-message": "Renamed util.py to helpers.py.",
        },
        cwd=str(tmp_path),
    )

    assert (digest.agent, digest.session_id, digest.asked, digest.answer) == (
        "Codex",
        "th-9",
        ["Rename the module."],
        "Renamed util.py to helpers.py.",
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_code_stats_count_what_moved_since_the_last_commit(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "app.py").write_text("a\nb\nc\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "first")
    (tmp_path / "app.py").write_text("a\nB\nc\nd\n")
    (tmp_path / "new.py").write_text("x\ny\n")

    stats = companion.code_stats(str(tmp_path))

    assert stats.repo == tmp_path.name
    assert (stats.changed_files, stats.new_files) == (1, 1)
    assert stats.added == 2 + 2 and stats.removed == 1
    assert stats.commits_today == 1 and stats.tracked_files == 1
    assert stats.languages == [("Python", 6)], "new files count too"


def test_the_briefing_tells_nur_what_happened_and_how_to_talk(tmp_path):
    digest = Digest(
        agent="Claude Code",
        session_id="s-1",
        cwd=str(tmp_path),
        asked=["Create hello.py with a test."],
        answer="Done. hello.py prints a greeting and its test passes.",
        files=["hello.py", "test_hello.py"],
        commands=["python -m pytest -q"],
    )
    stats = companion.CodeStats(
        repo="demo",
        branch="main",
        added=12,
        removed=1,
        changed_files=1,
        new_files=2,
        commits_today=3,
        tracked_files=10,
        languages=[("Python", 120)],
    )

    instructions, greeting = companion.briefing(digest, stats, name="Amr")

    assert instructions.startswith("You are Nur, Amr's coding companion.")
    assert "What Amr asked for last: Create hello.py with a test." in instructions
    assert "Files it wrote or edited: hello.py, test_hello.py" in instructions
    assert "+12 / -1 lines" in instructions and "Python 120 lines" in instructions
    assert "repeat back exactly what you will tell the agent" in instructions
    assert (
        greeting
        == "Hi Amr, it's Nur. Claude Code just finished in demo: Done. Want me to walk you through it?"
    )


class _Nur:
    """A NurClient stand-in: records each call and plays back one transcript per call."""

    calls: list[dict] = []
    transcripts: list[tuple[tuple[str, str], ...]] = []

    def __init__(self, *args, **kwargs):
        self.options = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def call(self, to, **options):
        _Nur.calls.append({"to": to, **options})
        return PhoneCall(
            id=f"call-{len(_Nur.calls)}",
            session_id="sess-1",
            status="dialing",
            to=to,
            created_at=time.time(),
        )

    async def wait_for_call(self, call_id, **kwargs):
        return PhoneCall(
            id=call_id,
            session_id="sess-1",
            status="completed",
            to="+14155550123",
            created_at=time.time(),
            duration_seconds=58.0,
            transcript=_Nur.transcripts.pop(0),
        )

    async def chat(self, messages, **kwargs):
        if "Have it add a docstring" in messages[-1]["content"]:
            return '{"follow_ups": ["Add a docstring to the greet function."]}'
        return '{"follow_ups": []}'


_ASKED = (
    ("nur", "Hi Amr, it's Nur. Claude Code just finished in demo."),
    ("person", "Nice. Have it add a docstring to greet, please."),
    ("nur", "I'll tell Claude: add a docstring to the greet function. Right?"),
    ("person", "Yes, that's it. Thanks!"),
)
_THANKS = (
    ("nur", "Hi Amr, it's Nur again. Claude Code did what you asked."),
    ("person", "Great, thanks. Bye."),
)


def _calls(monkeypatch, tmp_path, *transcripts):
    monkeypatch.setattr(companion, "HOME", tmp_path / "home")
    monkeypatch.setattr("eesi.client.NurClient", _Nur)
    _Nur.calls = []
    _Nur.transcripts = list(transcripts)


@run_async
async def test_what_was_asked_for_runs_in_the_same_session_and_nur_calls_back(
    tmp_path, monkeypatch
):
    _calls(monkeypatch, tmp_path, _ASKED, _THANKS)
    ran: list[dict] = []

    def run(command, **kwargs):
        ran.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "is_error": False,
                    "result": "Added a docstring to greet in hello.py.",
                    "session_id": "s-1",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(companion, "_run", run)
    digest = Digest(
        agent="Claude Code",
        session_id="s-1",
        cwd=str(tmp_path),
        asked=["Create hello.py."],
        answer="Done.",
    )

    outcome = await companion.call_about(
        digest, Settings(phone="+14155550123", name="Amr")
    )

    first, second = _Nur.calls
    assert first["to"] == "+14155550123" and first["greeting"].startswith(
        "Hi Amr, it's Nur."
    )
    assert "You are Nur, Amr's coding companion." in first["instructions"]
    (follow_up,) = ran
    assert follow_up["command"][:6] == [
        "claude",
        "-p",
        "--resume",
        "s-1",
        "--output-format",
        "json",
    ]
    assert follow_up["command"][-1] == (
        "Follow-up from a phone call with the developer:\n"
        "- Add a docstring to the greet function."
    )
    # The run calls no one itself: the companion calls back when it ends.
    assert follow_up["env"]["NUR_COMPANION_CALL_ON"] == "never"
    assert follow_up["cwd"] == str(tmp_path)
    assert second["greeting"] == (
        f"Hi Amr, it's Nur again. Claude Code did what you asked in {tmp_path.name}: "
        "Added a docstring to greet in hello.py. Anything else?"
    )
    assert (
        "What Amr asked for on the last call: Add a docstring" in second["instructions"]
    )
    assert [call["follow_ups"] for call in outcome["calls"]] == [
        ["Add a docstring to the greet function."],
        [],
    ]
    records = [
        json.loads(line)
        for line in (tmp_path / "home" / "calls.jsonl").read_text().splitlines()
    ]
    assert records[1]["follow_up"] == {
        "ok": True,
        "report": "Added a docstring to greet in hello.py.",
    }


@run_async
async def test_a_follow_up_that_cannot_run_is_called_back_too(tmp_path, monkeypatch):
    _calls(monkeypatch, tmp_path, _ASKED, _THANKS)
    monkeypatch.setattr(
        companion,
        "_run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Failed to authenticate: OAuth session expired and could not be refreshed\n",
        ),
    )

    await companion.call_about(
        Digest(agent="Claude Code", session_id="s-1", cwd=str(tmp_path)),
        Settings(phone="+14155550123", name="Amr"),
    )

    second = _Nur.calls[1]
    assert second["greeting"] == (
        "Hi Amr, it's Nur again. I couldn't get Claude Code to do what you asked: "
        "Failed to authenticate: OAuth session expired and could not be refreshed."
    )
    assert "It could not be done: Failed to authenticate" in second["instructions"]


def test_a_missing_agent_is_a_reason_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(companion, "HOME", tmp_path)

    def run(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(companion, "_run", run)

    done = companion.run_follow_up(
        Digest(agent="Claude Code", session_id="s-1", cwd=str(tmp_path)),
        ["Add tests."],
        Settings(),
    )

    assert done == companion.FollowUp(
        False, "the claude command is not installed on that computer"
    )


def test_codex_follow_ups_resume_its_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(companion, "HOME", tmp_path)
    ran: list[list[str]] = []

    def run(command, **kwargs):
        ran.append(command)
        report = command[command.index("--output-last-message") + 1]
        Path(report).write_text("Added the tests; all pass.")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(companion, "_run", run)

    done = companion.run_follow_up(
        Digest(agent="Codex", session_id="th-9", cwd=str(tmp_path)),
        ["Add tests."],
        Settings(),
    )

    assert ran[0][:6] == [
        "codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "-c",
        'approval_policy="never"',
    ]
    assert ran[0][8:10] == ["resume", "th-9"]
    assert done == companion.FollowUp(True, "Added the tests; all pass.", "th-9")


def test_settings_keep_the_key_to_this_user(tmp_path, monkeypatch):
    monkeypatch.setattr(companion, "HOME", tmp_path / "companion")
    monkeypatch.delenv("EESI_API_KEY", raising=False)
    monkeypatch.delenv("NUR_COMPANION_PHONE", raising=False)

    path = Settings.save(
        {"phone": "+14155550123", "api_key": "sk-eesi-test", "name": None}
    )
    Settings.save({"name": "Amr"})

    assert path == tmp_path / "companion.json"
    assert path.stat().st_mode & 0o777 == 0o600
    settings = Settings.load()
    assert (settings.phone, settings.name, settings.api_key) == (
        "+14155550123",
        "Amr",
        "sk-eesi-test",
    )
    monkeypatch.setenv("EESI_API_KEY", "sk-eesi-env")
    assert Settings.load().api_key == "sk-eesi-env"


class _Numbers:
    sent: list[str] = []
    confirmed: list[tuple[str, str]] = []

    def __init__(self, api_key=None, **kwargs):
        assert api_key == "sk-eesi-test"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def verify_phone_number(self, number):
        _Numbers.sent.append(number)
        return PhoneNumber(number=number, status="code_sent")

    async def confirm_phone_number(self, number, code, *, name=None):
        _Numbers.confirmed.append((number, code, name))
        return PhoneNumber(number=number, status="confirmed", confirmed_at=1.0)


def test_setup_saves_the_settings_and_nur_phones_a_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(companion, "HOME", tmp_path / "companion")
    monkeypatch.setattr("eesi.client.NurClient", _Numbers)
    monkeypatch.delenv("EESI_API_KEY", raising=False)
    monkeypatch.delenv("NUR_COMPANION_PHONE", raising=False)
    _Numbers.sent, _Numbers.confirmed = [], []

    assert (
        companion.main(
            [
                "setup",
                "--phone",
                "+1 (415) 555-0123",
                "--name",
                "Amr",
                "--api-key",
                "sk-eesi-test",
            ]
        )
        == 0
    )
    assert _Numbers.sent == ["+14155550123"]
    assert "eesi companion confirm" in capsys.readouterr().out
    assert companion.main(["confirm", "482 917"]) == 0
    assert _Numbers.confirmed == [("+14155550123", "482917", "Amr")]
    assert companion.main(["setup", "--phone", "4155550123"]) == 1


def test_call_me_calls_about_the_newest_session_here(tmp_path, monkeypatch):
    monkeypatch.setattr(companion, "HOME", tmp_path / "companion")
    monkeypatch.setenv("NUR_COMPANION_PHONE", "+14155550123")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    repo = tmp_path / "my.repo"
    repo.mkdir()
    # Claude Code names a project's folder after its path, every other character a dash.
    folder = tmp_path / "claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(repo))
    folder.mkdir(parents=True)
    (folder / "old.jsonl").write_text("{}\n")
    newest = folder / "s-42.jsonl"
    newest.write_text("{}\n")
    os.utime(folder / "old.jsonl", (1, 1))
    detached: list = []
    monkeypatch.setattr(
        companion, "_detach", lambda argv, payload: detached.append(payload)
    )

    assert companion.main(["call", "--cwd", str(repo)]) == 0

    (payload,) = detached
    assert payload["transcript_path"] == str(newest)
    assert payload["session_id"] == "s-42"


def test_what_is_said_drops_folders_and_marks():
    assert (
        companion._spoken(
            "Added `greet` in /home/amr/.cache/demo/hello.py and src/app/main.ts; and/or 1/2."
        )
        == "Added greet in hello.py and main.ts; and/or 1/2."
    )


def test_a_session_started_by_another_reports_through_its_hand_back(tmp_path):
    path = tmp_path / "agent.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "SubagentHandback",
                            "input": {"message": "Added shout(); 2 tests pass."},
                        }
                    ],
                },
            }
        )
        + "\n"
    )

    digest = companion.read_claude_session(
        str(path), session_id="a-1", cwd=str(tmp_path)
    )

    assert digest.answer == "Added shout(); 2 tests pass."


@run_async
async def test_nothing_is_handed_over_when_nothing_was_asked():
    class Quiet(_Nur):
        async def chat(self, messages, **kwargs):
            raise AssertionError("no person turns: no extraction")

    assert await companion.follow_ups(Quiet(), (("nur", "Hello?"),)) == []


@pytest.mark.parametrize(
    "call_on, minutes, event, calls",
    [
        ("stop", 0.1, "Stop", True),
        ("long", 0.1, "Stop", False),
        ("long", 5.0, "Stop", True),
        ("long", 0.1, "SessionEnd", True),
        ("never", 50.0, "Stop", False),
    ],
)
def test_when_nur_calls(tmp_path, monkeypatch, call_on, minutes, event, calls):
    monkeypatch.setenv("NUR_COMPANION_CALL_ON", call_on)
    monkeypatch.setenv("NUR_COMPANION_HOME", str(tmp_path))
    detached: list = []
    monkeypatch.setattr(
        companion, "_detach", lambda argv, payload: detached.append(payload)
    )
    transcript = _transcript(
        tmp_path / "t.jsonl", tmp_path, started=time.time() - minutes * 60
    )

    companion.claude_hook(
        json.dumps(
            {
                "hook_event_name": event,
                "session_id": "s-1",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            }
        )
    )

    assert bool(detached) is calls
    if calls:
        assert detached[0]["session_id"] == "s-1" and detached[0]["agent"] == "claude"


def test_a_stop_hook_already_continuing_never_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("NUR_COMPANION_CALL_ON", "stop")
    detached: list = []
    monkeypatch.setattr(
        companion, "_detach", lambda argv, payload: detached.append(payload)
    )

    companion.claude_hook(
        json.dumps({"hook_event_name": "Stop", "stop_hook_active": True})
    )

    assert detached == []
