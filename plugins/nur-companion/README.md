# Nur coding companion

When your coding agent finishes, Nur phones you: what you asked for, what the
agent did, the files it touched, how the code moved (lines added and removed,
commits today, languages). Ask it anything about the run. Ask for more work,
and when you hang up Nur runs it in the same agent session, then calls you back
with what it did, or with why it could not run. Say "that's all" and Nur hangs
up.

Works with Claude Code (this plugin) and Codex (its `notify` program).

## Set up (two minutes)

1. Install the EESI SDK. It puts an `eesi` command on your path:

   ```bash
   uv tool install git+https://github.com/eesi-ai/eesi-sdk-python
   # or: pipx install git+https://github.com/eesi-ai/eesi-sdk-python
   ```

2. Tell it who to call, with your EESI API key (the API keys page of your EESI
   console). Nur phones that number with a six-digit code; enter it:

   ```bash
   eesi companion setup --phone +14155550123 --name Sam --api-key sk-eesi-...
   eesi companion confirm 482917
   ```

   The settings live in `~/.eesi/companion.json`, readable by you only.

3. Connect your agent.

   **Claude Code**

   ```text
   /plugin marketplace add eesi-ai/eesi-sdk-python
   /plugin install nur-companion@eesi
   ```

   `/nur-companion:setup` walks through step 2 from inside Claude Code, and
   `/nur-companion:call-me` asks for an update now. Without the plugin,
   `eesi companion claude-hooks` prints the hooks for `~/.claude/settings.json`.

   **Codex**, in `~/.codex/config.toml`:

   ```toml
   notify = ["eesi", "companion", "codex-notify"]
   ```

`eesi companion status` shows the settings and the confirmed numbers.

## Call Nur

```bash
eesi companion listen            # beside your repository; --agent codex for Codex
```

It prints the code line's number. Call it from your confirmed phone and say
what to build: when you hang up, the work runs in one coding session, and Nur
calls you when it is done. Call again any time for progress (what is running,
its latest steps, what is queued and done) or to add work.

Or text it: "how's it going?", "also add a logout button". Nur answers from
the same facts and queues what you ask for. `--notify text` texts you when a
task is done instead of calling.

## When Nur calls

| Setting | Default | |
| --- | --- | --- |
| `call_on` | `long` | `long`: after a turn of at least `min_minutes` (3), and when a session ends; `stop`: after every turn; `never` |
| `follow_up` | `true` | Run what you ask for on the call in the same session, then call back |
| `follow_up_minutes` | `60` | How long a follow-up may run |
| `claude_args` | `--permission-mode acceptEdits` | Flags for follow-up runs of `claude -p --resume` |
| `codex_args` | `--sandbox workspace-write -c approval_policy='"never"'` | Flags for follow-up runs of `codex exec resume` |

Set them with `eesi companion setup --call-on stop`, in `~/.eesi/companion.json`,
or as `NUR_COMPANION_CALL_ON`-style environment variables (`EESI_API_KEY` and
`EESI_API_URL` win over the file).

The hook returns at once: the call, the follow-up and the call back happen in
the background, so the agent never waits. Logs and a record of every call are
in `~/.eesi/companion/`. Follow-ups need the agent's own command line on that
computer (`claude` or `codex`), signed in.

## What is shared

What Nur knows on a call is a briefing written on your machine from the
session's transcript and `git`: it is sent to EESI as the call's instructions
and kept with the call's transcript like any session of your organization.
On the shared EESI line Nur calls only numbers confirmed with a code, and the
code is read by the phone carrier, never by a session.
