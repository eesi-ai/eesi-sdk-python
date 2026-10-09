"""A game server that lets browsers and game engines talk to a character without the API key.

    pip install fastapi uvicorn
    uvicorn examples.game_server:app --port 8787

Your server keeps EESI_API_KEY. A game client (a browser, Unity, Unreal,
Godot) asks your server for a session and gets a one-use URL. It opens that
URL as a WebSocket and is talking to Kael at once: the session already has
his instructions, voice, tools, the player's memories and the live state you
passed, and the client cannot replace his instructions or tools.

What the client does on the socket (the Nur Live protocol, docs.eesi.ai/realtime/live):

- send microphone audio as ``input_audio_buffer.append`` (16-bit mono PCM,
  16 kHz, base64, about 20 ms per message);
- play ``response.output_audio.delta`` (24 kHz PCM) and drop what is queued
  on ``input_audio_buffer.speech_started`` (the player cut in);
- send ``eesi.context.update`` when game state changes;
- on ``response.function_call_arguments.done``, POST the call to
  ``/npc/kael/tools/<name>`` here, send the returned output back as a
  ``conversation.item.create`` of type ``function_call_output``, then a
  ``response.create``;
- when the conversation ends, POST what was said to ``/npc/kael/transcript``
  so Kael remembers it next time.

Authenticate your players your own way; ``player_id`` below stands in for it.
"""

from typing import Any

from eesi import NurClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

nur = NurClient()
kael = nur.npc(
    name="Kael",
    persona="An elven merchant who keeps the northern gate of Eldermoor.",
    voice="Atlas",
    memory=nur.memory.local("./kael-memory"),
    context={"realm": "Eldermoor"},
)
gates = {"north": "closed"}


@kael.tool
def open_gate(gate: str = "north") -> str:
    """Open one of the castle gates."""
    gates[gate] = "open"
    return f"The {gate} gate is open."


app = FastAPI(title="Kael's game server")


class SessionRequest(BaseModel):
    player_id: str
    player_name: str | None = None
    location: str = "the northern gate"


class ToolCall(BaseModel):
    player_id: str
    call_id: str
    arguments: str = "{}"


class Transcript(BaseModel):
    player_id: str
    player_name: str | None = None
    #: [speaker, text] pairs: "player" for the player, "event" for game events, anything else for Kael.
    lines: list[tuple[str, str]]


@app.post("/npc/kael/session")
async def open_session(request: SessionRequest) -> dict[str, Any]:
    secret = await kael.client_secret(
        request.player_id,
        player_name=request.player_name,
        context={"location": request.location, "north_gate": gates["north"]},
        expires_in=60,
    )
    return {"url": secret.url, "expires_at": secret.expires_at}


@app.post("/npc/kael/tools/{name}")
async def run_tool(name: str, call: ToolCall) -> dict[str, str]:
    if name not in {tool.name for tool in kael.tools}:
        raise HTTPException(404, f"Kael has no tool {name}")
    output = await kael.run_tool(
        name, call.arguments, player_id=call.player_id, call_id=call.call_id
    )
    return {"output": output}


@app.post("/npc/kael/transcript")
async def learn(transcript: Transcript) -> dict[str, int]:
    added, _, _ = await kael.memory.learn(
        transcript.lines,
        player_id=transcript.player_id,
        player_name=transcript.player_name,
    )
    return {"remembered": len(added)}
