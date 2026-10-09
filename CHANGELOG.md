# Changelog

All notable changes to `eesi-sdk` for Python. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package
follows [Semantic Versioning](https://semver.org/). Before 1.0, a minor version
may change public APIs; such changes are listed under **Changed**.

## 0.3.0

### Added

- `character.learn_from_session(event, player_id=...)`: learn a player's
  memories from a client-secret session's `session.ended` webhook, which now
  carries the whole transcript; `input.completed` posts what the player said.
- `MemoryPolicy(rules=...)`: what may be remembered, in your words.
- `eesi`, the Nur SDK, in the same wheel as the REST client:
  `from eesi import NurClient`.
- Characters: persona, traits, speaking style, goals, boundaries, knowledge,
  examples, language and voice; presets `npc`, `multiplayer_npc`, `companion`,
  `storyteller`, `support_agent` and `assistant`; save, load and clone as JSON.
- Voices by built-in name, by id, or described in words (`nur.voice_design`),
  created once in your library and reused by name.
- Realtime sessions on Nur Live: turn-taking modes (`natural`, `app`,
  `director`), barge-in, resume after a dropped connection, typed events,
  `ask()`, `cue()`, `moment()`, `say()`, `interrupt()` and hangup handling.
- Tools from plain functions, with schemas from type hints and docstrings,
  validation, `allow=` permissions, confirmation, timeouts and long-running
  work (`Started`).
- Live context in named, typed blocks with expiry, throttling and the
  gateway's limits checked before sending.
- Local memory in SQLite: per player, character and namespace; recall by
  relevance, importance and recency; learning after each session; preferences;
  inspection, correction, export, import and erasure.
- Multiplayer: a shared party log across players' sessions, and players bound
  to voices on a shared microphone.
- Client secrets (`character.client_secret`) so browsers and game engines
  connect without your API key, with `lock=` for what the client may not
  change.
- Traces that hold no words by default, per-reply latency, the `python -m eesi`
  command line, and `eesi.testing` scripted players for regression tests.
- `EESIClient.create_realtime_client_secret` in the REST client.
- A server control channel for client secrets (`control=True`,
  `character.control(secret)`): the server steers a session a client opened,
  without relaying audio. `secret.join` is what to send the client, in the
  form the TypeScript SDK's `joinSession` takes.
- `Room`: several microphones mixed into one conversation, each turn
  attributed to the player who was loudest.
- Languages: pin a character or a session to one language (greeting
  included), or answer each player in a group in their own.
- Speaking first: `when()` triggers on live state and `Proactive` idle
  check-ins, decided out of band and spoken only when worth it.
