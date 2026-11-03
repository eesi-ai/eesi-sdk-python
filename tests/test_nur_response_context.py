"""Provider receipts through the real SDK and a local scripted WebSocket.

These tests establish bounded metadata and transport behavior, not model
comprehension, generated acting, deployed availability or heard audio.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from conftest import FakeAPI, run_async
from eesi import NurClient
from eesi.events import AudioOutputChunk, Event, ResponseContext
from eesi.session import Session
from eesi.trace import Trace
from fake_gateway import FakeGateway


def receipt(response_id: str) -> dict[str, Any]:
    return {
        "type": "eesi.response_context",
        "schema_version": 1,
        "event_id": "receipt_event",
        "response_id": response_id,
        "turn_id": None,
        "turn_revision": None,
        "status": "provider_started",
        "receipt": {
            "receipt_id": "receipt_1",
            "coverage": "serialized_provider_input",
            "provider_format": "chat_completions",
            "model": "nur-llm-v1",
            "input_sha256": "a" * 64,
            "item_count": 1,
        },
    }


MALFORMED = [
    (("type",), "response.context"),
    (("type",), None),
    (("schema_version",), True),
    (("schema_version",), 2),
    (("schema_version",), None),
    (("status",), "completed"),
    (("receipt",), []),
    (("receipt",), None),
    (("receipt", "coverage"), "heard_audio"),
    (("receipt", "receipt_id"), ""),
    (("receipt", "receipt_id"), " "),
    (("receipt", "receipt_id"), "x" * 257),
    (("receipt", "receipt_id"), "receipt\n1"),
    (("receipt", "receipt_id"), " receipt_1"),
    (("receipt", "receipt_id"), "receipt_1 "),
    (("receipt", "model"), False),
    (("receipt", "model"), "nur\u200b"),
    (("receipt", "model"), "nur\u00a0model"),
    (("receipt", "model"), "x" * 257),
    (("receipt", "model"), " nur-llm-v1"),
    (("receipt", "model"), "nur-llm-v1 "),
    (("receipt", "input_sha256"), "A" * 64),
    (("receipt", "input_sha256"), "g" * 64),
    (("receipt", "input_sha256"), "a" * 63),
    (("receipt", "provider_format"), "unknown"),
    (("receipt", "item_count"), True),
    (("receipt", "item_count"), -1),
    (("receipt", "item_count"), 1.5),
    (("receipt", "item_count"), "1"),
    (("receipt", "item_count"), 2**53),
    (("turn_id",), ""),
    (("turn_id",), "x" * 257),
    (("turn_id",), "turn\n1"),
    (("turn_id",), 1),
    (("turn_id",), " turn_1 "),
    (("turn_revision",), True),
    (("turn_revision",), -1),
    (("turn_revision",), 1.5),
    (("turn_revision",), 2**53),
    (("response_id",), ""),
    (("response_id",), "r_receipt\n"),
    (("response_id",), "x" * 257),
    (("response_id",), " r_receipt "),
]


def malformed(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    frame = receipt("r_receipt")
    target = frame
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return frame


@asynccontextmanager
async def connected(
    api: FakeAPI, *, trace: Trace | None = None
) -> AsyncIterator[tuple[FakeGateway, Session]]:
    async with FakeGateway() as gateway:
        async with NurClient(
            api_key="sk-eesi-test",
            base_url=gateway.base_url,
            transport=api.transport,
            timezone=None,
        ) as nur:
            session = nur.npc(name="Kael", greeting=False).connect("p42", trace=trace)
            await session.start()
            try:
                yield gateway, session
            finally:
                await session.close(learn=False)


async def waiter(
    session: Session,
    kind: str,
    *,
    predicate: Callable[[Event], bool] | None = None,
) -> asyncio.Task[Event]:
    task = asyncio.create_task(session.wait(kind, timeout=5, predicate=predicate))
    await asyncio.sleep(0)  # Register before sending frames on the real socket.
    return task


@pytest.mark.parametrize(("path", "value"), MALFORMED)
def test_malformed_receipt_is_not_a_typed_event(path, value):
    assert ResponseContext._from_wire(malformed(path, value)) is None


@pytest.mark.parametrize(
    ("provider_format", "turn_id", "turn_revision"),
    [
        ("responses", "turn_1", 0),
        ("chat_completions", None, 2),
        ("responses", "turn_2", None),
        ("chat_completions", None, None),
    ],
)
def test_formats_and_independently_nullable_turn_fields(
    provider_format, turn_id, turn_revision
):
    frame = receipt("r_receipt")
    frame["receipt"]["provider_format"] = provider_format
    frame["receipt"]["item_count"] = 0
    frame["turn_id"] = turn_id
    frame["turn_revision"] = turn_revision
    event = ResponseContext._from_wire(frame)
    assert isinstance(event, ResponseContext)
    assert (
        event.provider_format,
        event.item_count,
        event.turn_id,
        event.turn_revision,
    ) == (
        provider_format,
        0,
        turn_id,
        turn_revision,
    )


def test_omitted_turn_fields_and_exact_integer_upper_boundary():
    frame = receipt("r_receipt")
    del frame["turn_id"], frame["turn_revision"]
    frame["receipt"]["item_count"] = 2**53 - 1
    frame["receipt"]["model"] = "نُور"
    frame["receipt"]["receipt_id"] = "x" * 256
    event = ResponseContext._from_wire(frame)
    assert isinstance(event, ResponseContext)
    assert event.item_count == 2**53 - 1
    assert event.turn_id is event.turn_revision is None


@run_async
async def test_typed_callbacks_wait_stream_and_content_free_trace_keep_hashes(api):
    trace = Trace()
    async with connected(api, trace=trace) as (gateway, session):
        seen: list[ResponseContext] = []
        session.on("response.context", seen.append)
        stream = session.events_iter()
        first = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        try:
            waiting = await waiter(session, "response.context")
            await gateway.send(
                {"type": "response.created", "response": {"id": "r_receipt"}}
            )
            frame = receipt("r_receipt")
            frame["instructions"] = "excluded prompt marker"
            frame["authorization"] = "excluded credential marker"
            frame["receipt"]["words"] = "excluded words marker"
            frame["receipt"]["tool_outputs"] = [
                {"output": "excluded tool output marker"}
            ]
            await gateway.send(frame)
            event = await waiting
            assert isinstance(event, ResponseContext)
            assert seen == [event]
            assert (await first).type == "response.started"
            assert await anext(stream) == event
            data = dataclasses.asdict(event)
            assert data.pop("at") >= 0
            assert data == {
                "session_id": gateway.session_id,
                "player_id": "p42",
                "response_id": "r_receipt",
                "receipt_id": "receipt_1",
                "input_sha256": "a" * 64,
                "provider_format": "chat_completions",
                "model": "nur-llm-v1",
                "item_count": 1,
                "coverage": "serialized_provider_input",
                "turn_id": None,
                "turn_revision": None,
            }
            line = next(
                line for line in trace.lines if line["kind"] == "response.context"
            )
            assert line["input_sha256"] == frame["receipt"]["input_sha256"]
            assert line["receipt_id"] == "receipt_1"
            assert "excluded" not in json.dumps([data, trace.lines])
        finally:
            await stream.aclose()


@run_async
async def test_malformed_and_extra_fields_leave_two_segment_pcm_and_text_complete(api):
    trace = Trace()
    async with connected(api, trace=trace) as (gateway, session):
        seen: list[ResponseContext] = []
        audio: list[AudioOutputChunk] = []
        errors: list[Event] = []
        session.on("response.context", seen.append)
        session.on("audio.output.chunk", audio.append)
        session.on("error", errors.append)
        await gateway.send(
            {"type": "response.created", "response": {"id": "r_receipt"}}
        )
        for path, value in MALFORMED:
            await gateway.send(malformed(path, value))
        waiting = await waiter(session, "response.context")
        valid = receipt("r_receipt")
        valid["receipt"]["words"] = "excluded words marker"
        valid["receipt"]["tool_outputs"] = [{"output": "excluded tool output marker"}]
        await gateway.send(valid)
        await waiting
        completed = await waiter(session, "response.completed")
        segments = [
            (b"\x01\x00" * 24, "أنا هون، والباب مفتوح."),
            (b"\xfe\xff" * 36, "أهلاً فيك."),
        ]
        for index, (pcm, text) in enumerate(segments):
            await gateway.send(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "r_receipt",
                    "delta": base64.b64encode(pcm).decode(),
                }
            )
            await gateway.send(
                {
                    "type": "response.output_audio_transcript.done",
                    "response_id": "r_receipt",
                    "transcript": text,
                    "event_id": f"segment_{index}",
                }
            )
        await gateway.send(
            {
                "type": "response.done",
                "response": {"id": "r_receipt", "status": "completed"},
            }
        )
        event = await completed
        assert event.text == " ".join(text for _, text in segments)
        assert [chunk.pcm for chunk in audio] == [pcm for pcm, _ in segments]
        assert len(seen) == 1
        assert errors == []
        recorded = json.dumps(trace.lines, ensure_ascii=False)
        assert "excluded" not in recorded
        assert all(text not in recorded for _, text in segments)
        assert any(line.get("input_sha256") == "a" * 64 for line in trace.lines)


@run_async
async def test_unknown_internal_cancelled_completed_receipts_are_not_exposed(api):
    async with connected(api) as (gateway, session):
        seen: list[ResponseContext] = []
        session.on("response.context", seen.append)
        await gateway.send(receipt("unknown"))
        await gateway.send(
            {"type": "response.created", "response": {"id": "cancelled"}}
        )
        await gateway.send(
            {"type": "output_audio_buffer.cleared", "response_id": "cancelled"}
        )
        await gateway.send(receipt("cancelled"))
        await gateway.send(
            {
                "type": "response.created",
                "response": {"id": "internal", "metadata": {"eesi_sdk": "idle_check"}},
            }
        )
        await gateway.send(receipt("internal"))
        await gateway.send(
            {"type": "response.created", "response": {"id": "completed"}}
        )
        await gateway.send(
            {
                "type": "response.done",
                "response": {"id": "completed", "status": "completed"},
            }
        )
        await gateway.send(receipt("completed"))
        await gateway.send(
            {"type": "response.created", "response": {"id": "completed"}}
        )
        await gateway.send(receipt("completed"))
        waiting = await waiter(session, "response.context")
        await gateway.send({"type": "response.created", "response": {"id": "public"}})
        await gateway.send(receipt("public"))
        assert (await waiting).response_id == "public"
        assert [event.response_id for event in seen] == ["public"]


@run_async
async def test_invalid_response_ids_do_not_consume_receipt_capacity_or_get_normalized(
    api,
):
    async with connected(api) as (gateway, session):
        seen: list[ResponseContext] = []
        session.on("response.context", seen.append)
        for rid in [
            "",
            " padded",
            "padded ",
            " ",
            "x" * 257,
            "line\nbreak",
            "zero\u200bwidth",
        ]:
            await gateway.send({"type": "response.created", "response": {"id": rid}})
            await gateway.send(receipt(rid))
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": rid, "status": "completed"},
                }
            )
        rid = "valid internal space"
        waiting = await waiter(session, "response.context")
        await gateway.send({"type": "response.created", "response": {"id": rid}})
        frame = receipt(rid)
        frame["receipt"]["receipt_id"] = "receipt internal space"
        frame["receipt"]["model"] = "model internal space"
        frame["turn_id"] = "turn internal space"
        await gateway.send(frame)
        event = await waiting
        assert isinstance(event, ResponseContext)
        assert event.response_id == rid
        assert event.receipt_id == "receipt internal space"
        assert event.model == "model internal space"
        assert event.turn_id == "turn internal space"
        assert session._context_responses == {rid: True}
        assert not session._context_saturated
        assert seen == [event]


@run_async
async def test_reconnect_drops_prior_eligibility_and_allows_multiple_provider_passes(
    api,
):
    async with connected(api) as (gateway, session):
        seen: list[ResponseContext] = []
        session.on("response.context", seen.append)
        started = await waiter(session, "response.started")
        await gateway.send(
            {"type": "response.created", "response": {"id": "old_connection"}}
        )
        await started
        reconnected = await waiter(session, "reconnected")
        await gateway.drop()
        await reconnected
        await gateway.send(receipt("old_connection"))
        waiting = await waiter(session, "response.context")
        await gateway.send(
            {"type": "response.created", "response": {"id": "new_connection"}}
        )
        await gateway.send(receipt("new_connection"))
        assert (await waiting).response_id == "new_connection"
        second = await waiter(session, "response.context")
        frame = receipt("new_connection")
        frame["receipt"]["receipt_id"] = "receipt_2"
        frame["receipt"]["input_sha256"] = "b" * 64
        await gateway.send(frame)
        assert (await second).receipt_id == "receipt_2"
        assert [event.input_sha256 for event in seen] == ["a" * 64, "b" * 64]


@run_async
async def test_terminal_ids_survive_reply_pruning_and_internal_id_reuse(api):
    async with connected(api) as (gateway, session):
        seen: list[ResponseContext] = []
        session.on("response.context", seen.append)
        await gateway.send(
            {"type": "response.created", "response": {"id": "old_completed"}}
        )
        await gateway.send(
            {
                "type": "response.done",
                "response": {"id": "old_completed", "status": "completed"},
            }
        )
        for index in range(66):
            rid = f"prune_{index}"
            await gateway.send({"type": "response.created", "response": {"id": rid}})
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": rid, "status": "completed"},
                }
            )
        barrier = await waiter(
            session,
            "response.started",
            predicate=lambda event: event.response_id == "barrier",
        )
        await gateway.send({"type": "response.created", "response": {"id": "barrier"}})
        await barrier
        assert "old_completed" not in session._responses
        await gateway.send(
            {"type": "response.created", "response": {"id": "old_completed"}}
        )
        await gateway.send(receipt("old_completed"))
        await gateway.send(
            {"type": "response.created", "response": {"id": "reused_internal"}}
        )
        await gateway.send(
            {
                "type": "response.created",
                "response": {
                    "id": "reused_internal",
                    "metadata": {"eesi_sdk": "idle_check"},
                },
            }
        )
        await gateway.send(
            {
                "type": "response.done",
                "response": {"id": "reused_internal", "status": "completed"},
            }
        )
        await gateway.send(receipt("reused_internal"))
        waiting = await waiter(session, "response.context")
        await gateway.send({"type": "response.created", "response": {"id": "public"}})
        await gateway.send(receipt("public"))
        assert (await waiting).response_id == "public"
        assert [event.response_id for event in seen] == ["public"]


@run_async
async def test_cancelled_reply_remains_ineligible_after_deque_eviction_and_reconnect(
    api,
):
    async with connected(api) as (gateway, session):
        seen: list[ResponseContext] = []
        session.on("response.context", seen.append)
        for rid in ["old_cancelled", *(f"cut_{index}" for index in range(65))]:
            await gateway.send({"type": "response.created", "response": {"id": rid}})
            await gateway.send(
                {"type": "output_audio_buffer.cleared", "response_id": rid}
            )
        barrier = await waiter(
            session,
            "response.started",
            predicate=lambda event: event.response_id == "barrier",
        )
        await gateway.send({"type": "response.created", "response": {"id": "barrier"}})
        await barrier
        assert "old_cancelled" not in session._cancelled
        assert session._responses["old_cancelled"].cancelled
        reconnected = await waiter(session, "reconnected")
        await gateway.drop()
        await reconnected
        await gateway.send(
            {"type": "response.created", "response": {"id": "old_cancelled"}}
        )
        await gateway.send(receipt("old_cancelled"))
        waiting = await waiter(session, "response.context")
        await gateway.send({"type": "response.created", "response": {"id": "public"}})
        await gateway.send(receipt("public"))
        assert (await waiting).response_id == "public"
        assert [event.response_id for event in seen] == ["public"]


@run_async
async def test_cap_suppresses_receipts_until_reconnect_without_stopping_pcm_or_text(
    api,
):
    async with connected(api) as (gateway, session):
        seen: list[ResponseContext] = []
        audio: list[AudioOutputChunk] = []
        session.on("response.context", seen.append)
        session.on("audio.output.chunk", audio.append)
        first = await waiter(session, "response.context")
        await gateway.send(
            {"type": "response.created", "response": {"id": "still_active"}}
        )
        await gateway.send(receipt("still_active"))
        await first
        for index in range(2047):
            rid = f"cap_{index}"
            await gateway.send({"type": "response.created", "response": {"id": rid}})
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": rid, "status": "completed"},
                }
            )
        await gateway.send({"type": "response.created", "response": {"id": "over_cap"}})
        await gateway.send(receipt("over_cap"))
        await gateway.send(receipt("still_active"))
        completed = await waiter(
            session,
            "response.completed",
            predicate=lambda event: event.response_id == "after_cap",
        )
        await gateway.reply("after_cap", "Audio continues.", audio_ms=1)
        assert (await completed).text == "Audio continues."
        assert audio[0].pcm == b"\x10\x00" * 24
        assert len(session._context_responses) == 2048
        assert session._context_saturated
        assert [event.response_id for event in seen] == ["still_active"]
        reconnected = await waiter(session, "reconnected")
        await gateway.drop()
        await reconnected
        waiting = await waiter(session, "response.context")
        await gateway.send(
            {"type": "response.created", "response": {"id": "post_reconnect"}}
        )
        await gateway.send(receipt("post_reconnect"))
        assert (await waiting).response_id == "post_reconnect"
        assert [event.response_id for event in seen] == [
            "still_active",
            "post_reconnect",
        ]
        assert len(session._context_responses) == 1
        assert not session._context_saturated
