# EESI for Python

Characters you can talk to, in real time, on [Nur Live](https://docs.eesi.ai/realtime/live):
a persona, a voice, tools, live state from your game or app, and long memory of
every player. Plus a typed client for the rest of the EESI API.

```bash
pip install "eesi-sdk[audio] @ git+https://github.com/eesi-ai/eesi-sdk-python"
export EESI_API_KEY=sk-eesi-...
```

See [Install](#install) for uv, requirements files, pinning and servers.

```python
import asyncio
from eesi import NurClient

nur = NurClient()

kael = nur.npc(
    name="Kael",
    persona="A mysterious elven merchant who distrusts kings.",
    voice="Atlas",
    memory=True,
)

asyncio.run(kael.start())   # talk through your microphone until Ctrl+C
```

Or without code: `python -m eesi talk --name Luna --preset companion --persona "A curious companion."`

## Install

Python 3.10 or later.

```bash
pip install "eesi-sdk[audio] @ git+https://github.com/eesi-ai/eesi-sdk-python"
```

| With | Command |
| --- | --- |
| uv | `uv add "eesi-sdk[audio] @ git+https://github.com/eesi-ai/eesi-sdk-python"` |
| Poetry | `poetry add "git+https://github.com/eesi-ai/eesi-sdk-python" --extras audio` |
| requirements.txt | `eesi-sdk[audio] @ git+https://github.com/eesi-ai/eesi-sdk-python` |
| A fixed version | `pip install "eesi-sdk[audio] @ git+https://github.com/eesi-ai/eesi-sdk-python@<commit>"` |

- The `audio` extra adds this computer's microphone and speaker (`numpy`,
  `sounddevice`). Leave it out on a server, which streams audio another way.
  On Linux, `sounddevice` needs PortAudio: `sudo apt install libportaudio2`.
- Your API key: create one on the API keys page of the
  [EESI console](https://platform.eesi.ai), then `export EESI_API_KEY=sk-eesi-...`
  (or pass `NurClient(api_key=...)`). Keep it on machines you control; give
  browsers and game clients a client secret instead.
- Check it works: `python -m eesi chat --name Kael --preset npc --persona "An elven merchant."`

Both imports come from this one package: `from eesi import NurClient` (the
characters) and `from eesi import EESIClient` (the REST API). The PyPI release
(`pip install "eesi-sdk[audio]"`) is on the way; the code is the same.

## What you get

- **Real speech to speech.** Nur Live listens while it speaks, decides when a
  turn has ended, backchannels, and stops when someone cuts in. The SDK drops
  the character's queued audio the moment it is cut off.
- **Characters, not prompts.** Persona, traits, speaking style, goals,
  boundaries, knowledge, language and voice, with presets for NPCs,
  multiplayer NPCs, companions, storytellers, support agents and assistants.
  Save a character as JSON, load it, clone it.
- **Tools.** `@npc.tool` on any function. Schemas come from type hints,
  arguments are validated, permissions are checked in your code, slow work
  reports back when it finishes, and nothing blocks the audio.
- **Live context.** `await session.context.update({...})` as often as your game
  ticks. The character reads the latest state on its next reply; your game
  stays the authority.
- **Long memory, local.** A SQLite file on your machine. Each session starts
  with what matters about this player, recalls more as they talk, and learns
  facts, preferences, events and promises when it ends. Inspect, correct,
  export and erase any of it. Players never share memories.
- **Games and multiplayer.** One character for many players, several players
  on one microphone, cues and paced remarks, and client secrets so browsers
  and game engines connect without your API key and cannot change what you
  lock (instructions, tools, voice, turn-taking, context, conversation).
- **Observability.** Typed events, a privacy-preserving JSONL trace, and
  latency per reply.

## A fuller example

```python
from typing import Literal
from eesi import NurClient, ToolContext

nur = NurClient()

kael = nur.npc(
    name="Kael",
    persona="An elven merchant who keeps the northern gate of Eldermoor.",
    voice="Atlas",
    boundaries=["Never reveal the location of the hidden vault."],
    memory=nur.memory.local("./kael-memory"),
    context={"realm": "Eldermoor", "time_of_day": "dusk"},
)

@kael.tool(allow=lambda ctx: game.reputation(ctx.player_id) != "untrusted")
async def open_gate(gate: Literal["north", "south"], ctx: ToolContext) -> str:
    """Open one of the castle gates for the player."""
    return await game.open_gate(gate, for_player=ctx.player_id)

async def talk(player_id: str) -> None:
    async with kael.connect(player_id, audio=nur.audio.local()) as session:
        await session.context.update({"location": "the northern gate", "gate": "closed"})

        @session.on("transcript.final")
        def heard(event):
            print("player:", event.text)

        await session.run()
```

## Browsers and game clients

Your API key stays on your server. A browser or a game build gets a client
secret that opens a session already configured as the character, and cannot
change what you lock; the client streams audio straight to EESI. With
`control=True` your server steers that session (state, tools, cues, memory)
without relaying audio:

```python
secret = await kael.client_secret("p42", player_name="Mira", lock="all", control=True)
send_to_client(secret.join)                  # the TypeScript SDK's joinSession(join) takes it as it is
async with kael.control(secret) as session:  # ready when the player connects
    await session.context.update({"location": "the northern gate"})
    await session.run()
```

## Several players, any language, speaking first

```python
from eesi import Player, Proactive

room = kael.room([Player("Ana", "p1", language="ar"), Player("Ben", "p2", language="es")])
kael = nur.npc(name="Kael", language="ar-EG")                      # or follow the player (the default)
kael.when(lambda state: state.get("health", 100) < 20, "The player is badly hurt.")
luna = nur.companion(name="Luna", proactive=Proactive(idle_after=30))
```

A room mixes your voice chat's microphones into one conversation and knows who
spoke from the audio; each player is answered in their own language.

## Guides and examples

- The guide: [docs.eesi.ai/realtime/characters](https://docs.eesi.ai/realtime/characters)
- Examples in [`examples/`](examples): an RPG NPC wired to a game loop, a
  companion with long memory, a multiplayer character, a tool-using support
  agent, a game server that hands clients secrets, and a scripted run you can
  put in CI.

## The REST client

Every other endpoint (speech, transcription, voices, audio intelligence) is on
`EESIClient`:

```python
from eesi import EESIClient   # also importable as eesi_sdk.EESIClient

with EESIClient() as client:
    voices = client.list_voices()
```

## Tests

```bash
python -m pytest tests -q                         # unit tests and a local fake gateway
EESI_API_KEY=... EESI_LIVE_TESTS=1 python -m pytest tests/test_nur_live.py   # real Nur Live sessions
```

The first set proves what the SDK sends and how it reacts; it cannot prove
anything about speech or the model. The live tests run real sessions against
the API and cost usage.
