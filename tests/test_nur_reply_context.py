"""Explicit replies follow pending application state on the same ordered socket.

Real SDK sessions/publishers use an in-memory socket here. These tests establish
wire ordering and failure boundaries, not model compliance or deployed behavior.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from conftest import run_async
from eesi import NurClient, SessionClosedError
from eesi.session import Session
from fake_gateway import FakeGateway

OLD_GOAL = "B2: Bargain for the unpaid forty pounds. Do not give the toast yet."
FINAL_GOAL = "B4: Give the warm toast now, including the unpaid forty pounds."


class RecordingSocket:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.before_send: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    async def send(self, text: str) -> None:
        frame = json.loads(text)
        if self.before_send is not None:
            await self.before_send(frame)
        self.frames.append(frame)


@asynccontextmanager
async def context_session(api):
    async with NurClient(api_key="sk-eesi-test", transport=api.transport) as nur:
        session = nur.character("Juniper", greeting=False).connect()
        socket = RecordingSocket()
        session._ws = socket
        session.context.seed({"director": OLD_GOAL})
        # This is the production publisher, including its real pending timer,
        # lock, deduplication and ordered Session._send transport.
        await session._publisher.start()
        socket.frames.clear()
        try:
            yield session, socket
        finally:
            await session._publisher.close()


@pytest.mark.parametrize("method", ["respond", "say", "send_text"])
@run_async
async def test_pending_context_is_sent_before_an_explicit_reply(api, method):
    async with context_session(api) as (session, socket):
        await session.context.update({"director": FINAL_GOAL})
        await getattr(session, method)("Make the final toast.")
        frames = socket.frames
        context_index = next(
            i
            for i, frame in enumerate(frames)
            if frame["type"] == "eesi.context.update"
        )
        response_index = next(
            i for i, frame in enumerate(frames) if frame["type"] == "response.create"
        )
        assert context_index < response_index
        assert frames[context_index]["key"] == "scene"
        assert frames[context_index]["text"] == f"director: {FINAL_GOAL}"
        assert OLD_GOAL not in frames[context_index]["text"]
        await session._publisher.flush()
        assert sum(f["type"] == "eesi.context.update" for f in frames) == 1


@run_async
async def test_failed_context_flush_does_not_request_a_reply(api):
    async with context_session(api) as (session, socket):

        async def fail_context(frame):
            if frame["type"] == "eesi.context.update":
                raise RuntimeError("context transport failed")

        socket.before_send = fail_context
        await session.context.update({"director": FINAL_GOAL})
        with pytest.raises(RuntimeError, match="context transport failed"):
            await session.respond("Make the final toast.")
        assert not socket.frames
        assert session._publisher.current()["scene"] == f"director: {OLD_GOAL}"


@pytest.mark.parametrize("ended", [False, True])
@run_async
async def test_session_closing_during_context_send_does_not_request_a_reply(api, ended):
    async with context_session(api) as (session, socket):

        async def end_during_context(frame):
            if frame["type"] == "eesi.context.update":
                if ended:
                    session._ended.set()
                else:
                    session._closing = True

        socket.before_send = end_during_context
        await session.context.update({"director": FINAL_GOAL})
        with pytest.raises(SessionClosedError):
            await session.say("A final toast.")
        assert [f["type"] for f in socket.frames] == ["eesi.context.update"]


@run_async
async def test_explicit_reply_waits_for_background_flush_without_duplicate_context(api):
    async with context_session(api) as (session, socket):
        entered, release = asyncio.Event(), asyncio.Event()

        async def block_context(frame):
            if frame["type"] == "eesi.context.update":
                entered.set()
                await release.wait()

        socket.before_send = block_context
        session._publisher._min_interval = 0
        await session.context.update({"director": FINAL_GOAL})
        await asyncio.wait_for(entered.wait(), 1)
        reply = asyncio.create_task(session.respond("Make the final toast."))
        try:
            await asyncio.sleep(0)  # Let the reply reach the publisher's held lock.
            assert not reply.done()
            assert not socket.frames
            release.set()
            await asyncio.wait_for(reply, 1)
            await session._publisher.flush()
            assert [f["type"] for f in socket.frames] == [
                "eesi.context.update",
                "response.create",
            ]
        finally:
            release.set()
            if not reply.done():
                reply.cancel()
                await asyncio.gather(reply, return_exceptions=True)


@run_async
async def test_closed_session_does_not_flush_pending_context(api):
    async with context_session(api) as (session, socket):
        await session.context.update({"director": FINAL_GOAL})
        session._closing = True
        with pytest.raises(SessionClosedError):
            await Session._request_reply(session, {"type": "response.create"})
        assert not socket.frames


@pytest.mark.parametrize("method", ["respond", "say"])
@run_async
async def test_ordered_websocket_replaces_the_old_director_before_the_reply(
    api, method
):
    async with FakeGateway() as gateway:
        async with NurClient(
            api_key="sk-eesi-test", base_url=gateway.base_url, transport=api.transport
        ) as nur:
            async with nur.character("Juniper", greeting=False).connect() as session:
                await session.context.update({"director": OLD_GOAL})
                await session._publisher.flush()
                await gateway.wait_for(
                    lambda f: (
                        f.get("type") == "eesi.context.update"
                        and f.get("text") == f"director: {OLD_GOAL}"
                    )
                )
                await session.context.update({"director": FINAL_GOAL})
                await getattr(session, method)("Make the final toast.")
                response = await gateway.wait_type("response.create")
                before_response = gateway.frames[: gateway.frames.index(response)]
                contexts = [
                    f for f in before_response if f["type"] == "eesi.context.update"
                ]
                assert contexts[-1]["key"] == "scene"
                assert contexts[-1]["text"] == f"director: {FINAL_GOAL}"
