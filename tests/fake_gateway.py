"""A Nur Live gateway stand-in for SDK tests: a real WebSocket server, scripted by the test.

It greets like the gateway (recording disclosure, ``eesi.session`` with a
resume token, ``session.created``), answers every ``session.update`` with
``session.updated``, records every client frame, and lets a test push any
server event, drop the connection, or refuse the handshake. It knows nothing
about speech: tests decide what the "model" says.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import Callable
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response


class FakeGateway:
    def __init__(
        self, *, resumable: bool = True, refuse_status: int | None = None
    ) -> None:
        self.resumable = resumable
        self.refuse_status = refuse_status
        self.frames: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[Headers] = []
        self.connections = 0
        self.session_id = f"conv_{uuid.uuid4().hex[:12]}"
        self.resume_token = "resume-secret"
        self._ws: ServerConnection | None = None
        self._frame_event = asyncio.Event()
        self._server: Any = None
        self.port = 0
        self.on_frame: Callable[[dict[str, Any]], Any] | None = None
        self.control_tools = "server"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> FakeGateway:
        self._server = await serve(
            self._handler, "127.0.0.1", 0, process_request=self._process_request
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    def _process_request(
        self, connection: ServerConnection, request: Any
    ) -> Response | None:
        if self.refuse_status is not None:
            return connection.respond(self.refuse_status, "refused\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections += 1
        self._ws = ws
        self.paths.append(ws.request.path)
        self.headers.append(ws.request.headers)
        if ws.request.path.startswith("/v1/realtime/control/"):
            await self._control(ws)
            return
        resumed = "resume=" in ws.request.path
        await ws.send(json.dumps({"type": "eesi.synthetic_audio", "declared": True}))
        await ws.send(
            json.dumps(
                {"type": "eesi.recording_disclosure", "disclosure": {"recorded": False}}
            )
        )
        greeting: dict[str, Any] = {
            "type": "eesi.session",
            "session_id": self.session_id,
            "resume": {"supported": self.resumable},
            "recording_session_id": f"rec_{self.connections}",
        }
        if self.resumable:
            greeting["resume_token"] = self.resume_token
        if resumed:
            greeting["resumed"] = {"replayed_items": 4}
        await ws.send(json.dumps(greeting))
        await ws.send(
            json.dumps(
                {
                    "type": "session.created",
                    "session": {"id": "sess_1", "model": "nur-live-v1"},
                }
            )
        )
        try:
            async for raw in ws:
                frame = json.loads(raw)
                self.frames.append(frame)
                self._frame_event.set()
                if frame.get("type") == "session.update":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "session": frame.get("session", {}),
                            }
                        )
                    )
                if (
                    frame.get("type") == "conversation.item.create"
                    and frame.get("item", {}).get("type") == "message"
                ):
                    await ws.send(
                        json.dumps(
                            {"type": "conversation.item.added", "item": frame["item"]}
                        )
                    )
                if self.on_frame is not None:
                    result = self.on_frame(frame)
                    if asyncio.iscoroutine(result):
                        await result
        except Exception:
            pass

    async def _control(self, ws: ServerConnection) -> None:
        """A control channel: the session is announced, frames are recorded."""
        control_id = ws.request.path.rsplit("/", 1)[-1]
        await ws.send(
            json.dumps(
                {
                    "type": "eesi.control.attached",
                    "control_id": control_id,
                    "session_id": self.session_id,
                    "recording_session_id": "rec_control",
                    "tools": self.control_tools,
                }
            )
        )
        try:
            async for raw in ws:
                frame = json.loads(raw)
                self.frames.append(frame)
                self._frame_event.set()
        except Exception:
            pass

    async def end_control(self, reason: str = "client_closed") -> None:
        await self.send({"type": "eesi.control.ended", "reason": reason})

    # ── scripting ────────────────────────────────────────────────────────

    async def send(self, event: dict[str, Any]) -> None:
        assert self._ws is not None, "no client connected"
        await self._ws.send(json.dumps(event))

    async def drop(self, code: int = 1011) -> None:
        """Close the client's socket the way a lost connection or a failed backend would."""
        assert self._ws is not None
        await self._ws.close(code, "dropped")

    async def wait_for(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 5.0
    ) -> dict[str, Any]:
        """The first recorded frame (from now on, or already seen) matching *predicate*."""
        deadline = asyncio.get_running_loop().time() + timeout
        seen = 0
        while True:
            for frame in self.frames[seen:]:
                if predicate(frame):
                    return frame
            seen = len(self.frames)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError(
                    f"no matching frame; got {[f.get('type') for f in self.frames]}"
                )
            self._frame_event.clear()
            try:
                await asyncio.wait_for(self._frame_event.wait(), remaining)
            except asyncio.TimeoutError:
                pass

    async def wait_type(
        self, kind: str, timeout: float = 5.0, after: int = 0
    ) -> dict[str, Any]:
        def match(frame: dict[str, Any]) -> bool:
            return frame.get("type") == kind and self.frames.index(frame) >= after

        return await self.wait_for(match, timeout)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [frame for frame in self.frames if frame.get("type") == kind]

    # ── what a model would do ────────────────────────────────────────────

    async def reply(
        self, response_id: str, text: str, *, audio_ms: int = 200, done: bool = True
    ) -> None:
        """A spoken reply: created, a transcript, some audio, done."""
        await self.send({"type": "response.created", "response": {"id": response_id}})
        await self.send(
            {
                "type": "response.output_audio_transcript.delta",
                "response_id": response_id,
                "delta": text,
            }
        )
        samples = 24 * audio_ms
        await self.send(
            {
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "delta": base64.b64encode(b"\x10\x00" * samples).decode(),
            }
        )
        await self.send(
            {
                "type": "response.output_audio_transcript.done",
                "response_id": response_id,
                "transcript": text,
                "event_id": f"e_{response_id}",
            }
        )
        if done:
            await self.send(
                {
                    "type": "response.done",
                    "response": {"id": response_id, "status": "completed"},
                }
            )

    async def player_says(self, text: str, item_id: str = "item_1") -> None:
        await self.send(
            {"type": "input_audio_buffer.speech_started", "item_id": item_id}
        )
        await self.send(
            {"type": "input_audio_buffer.speech_stopped", "item_id": item_id}
        )
        await self.send(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": text,
            }
        )
