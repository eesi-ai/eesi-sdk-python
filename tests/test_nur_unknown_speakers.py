"""Explicitly unknown speech must not acquire a player through SDK fallbacks."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from conftest import run_async
from eesi import NurClient, Player, ToolContext
from eesi._protocol import parse_speaker_turns, split_speaker_turns
from eesi.events import SpeakerTurn, TranscriptFinal
from eesi.memory import ExtractionResult, InMemoryStore, Memory, MemoryPolicy
from eesi.multiplayer import SpeakerRoster
from eesi.trace import Trace, format_trace


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello\nthere.", [(None, "Hello\nthere.", False)]),
        ("Unattributed voice: Hello.", [(None, "Hello.", True)]),
        (
            "Speaker 1: Known.\n unattributed voice : Unknown.\nStill unknown.\nSpeaker 1: Known again.",
            [
                (1, "Known.", False),
                (None, "Unknown. Still unknown.", True),
                (1, "Known again.", False),
            ],
        ),
        (
            "Unattributed voice: First.\nUnattributed voice: Second.",
            [(None, "First.", True), (None, "Second.", True)],
        ),
    ],
)
def test_parser_keeps_explicit_unknown_boundaries(text, expected):
    parsed = parse_speaker_turns(text)
    assert [(turn.speaker, turn.text, turn.unattributed) for turn in parsed] == expected
    # Keep the private legacy helper's two-value unpacking contract.
    assert split_speaker_turns(text) == [
        (speaker, words) for speaker, words, _ in expected
    ]


def test_unknown_does_not_use_solo_identity_or_poison_introduction_order():
    solo = SpeakerRoster(["Ana"])
    assert solo.attribute("Hello.")[0].player == "Ana"
    unknown = solo.attribute("Unattributed voice: I'm Ana.")[0]
    assert unknown.unattributed and unknown.speaker is None and unknown.player is None
    assert solo.players() == {}

    roster = SpeakerRoster(["Ana", "Ben"])
    roster.attribute("Unattributed voice: I'm Ben.")
    known = roster.attribute("Speaker 1: Hello.")[0]
    assert known.player == "Ana", (
        "Unknown text must not seed the introduction-order identity heuristic"
    )
    turns = roster.attribute("Speaker 1: My words.\nUnattributed voice: Not mine.")
    assert [turn.player for turn in turns] == ["Ana", None]
    assert turns[1].unattributed and roster.players() == {1: "Ana"}


@pytest.mark.parametrize("content", [False, True])
def test_trace_preserves_explicit_unknown_without_inventing_player(tmp_path, content):
    path = tmp_path / "trace.jsonl"
    trace = Trace(path, content=content)
    trace.open()
    trace.event(
        TranscriptFinal(
            text="Unknown.", turns=(SpeakerTurn(text="Unknown.", unattributed=True),)
        )
    )
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    turn = next(line["turns"][0] for line in lines if "turns" in line)
    assert turn["unattributed"] and turn["speaker"] is None and turn["player"] is None
    assert "Unattributed voice:" in format_trace(lines)
    assert ("Unknown." in path.read_text()) == content


@run_async
async def test_solo_session_and_tools_do_not_reuse_the_player_for_unknown(api):
    async with NurClient(api_key="test", transport=api.transport) as nur:
        character = nur.npc(name="Guide", greeting=False, memory=False, party=True)
        session = character.connect("p1", player_name="Ana")
        finals = []
        session.on("transcript.final", finals.append)
        session._on_user_turn("My words.", "known")
        assert session._tool_context("before", {}).player_id == "p1"
        session._on_user_turn("Unattributed voice: Not Ana.", "unknown")
        assert finals[-1].turns[0].unattributed
        assert (
            finals[-1].turns[0].speaker is None
            and finals[-1].turns[0].player_id is None
        )
        assert session._tool_context("after", {}).player_id is None
        assert session.transcript[-1] == ("Unattributed voice", "Not Ana.")
        assert "Unattributed voice: Not Ana." in character.party.render()
        session._on_user_turn("Solo again.", "solo")
        assert session._tool_context("solo", {}).player_id == "p1"


@run_async
async def test_room_unknown_bypasses_loudest_player_and_clears_last_speaker(api):
    async with NurClient(api_key="test", transport=api.transport) as nur:
        room = nur.npc(name="Guide", greeting=False, memory=False).room(
            [Player("Ana", "p1")]
        )
        # Real room ranking from measured per-source levels: a loud source
        # still cannot override an explicitly unknown recognition label.
        room._timeline.append((0, {"p1": 0.5}))
        assert room.loudest(None, None) == ["p1"]
        room.session._on_user_turn("Speaker 1: Hello.", "known")
        assert room._bound == {1: "p1"} and "Ana spoke last." in room.render()
        room.session._on_user_turn("Unattributed voice: Not hers.", "unknown")
        assert room._bound == {1: "p1"}
        assert room._last_speaker is None and "Ana spoke last." not in room.render()
        assert room.session._tool_context("unknown", {}).player_id is None
        assert room.session.transcript[-1] == ("Unattributed voice", "Not hers.")
        unknown = room._attribute("Unattributed voice: Still unknown.")[0]
        assert (
            unknown.player is None and unknown.speaker is None and unknown.unattributed
        )


@run_async
async def test_unknown_words_do_not_query_or_become_a_known_players_memory(
    api, monkeypatch
):
    extracted = []

    async def extractor(request):
        extracted.append(request)
        return ExtractionResult()

    async with NurClient(api_key="test", transport=api.transport) as nur:
        character = nur.npc(
            name="Guide",
            greeting=False,
            memory=Memory(InMemoryStore(), policy=MemoryPolicy(extractor=extractor)),
        )
        session = character.connect("p1", player_name="Ana")
        recall = AsyncMock()
        monkeypatch.setattr(session, "_recall_for", recall)
        session._on_user_turn("I grow roses.", "known")
        await session._memory_task
        session._on_user_turn("Unattributed voice: My secret is unknown.", "unknown")
        assert recall.await_count == 1
        await session._learn()
        assert len(extracted) == 1 and extracted[0].player_id == "p1"
        assert list(extracted[0].transcript) == [("player", "I grow roses.")]


@pytest.mark.parametrize("mixed", [False, True])
@run_async
async def test_unknown_reply_is_excluded_from_memory_even_after_a_known_turn_resumes(
    api, mixed
):
    extracted = []

    async def extractor(request):
        extracted.append(request)
        return ExtractionResult()

    async with NurClient(api_key="test", transport=api.transport) as nur:
        character = nur.npc(
            name="Guide",
            greeting=False,
            memory=Memory(
                InMemoryStore(),
                policy=MemoryPolicy(extractor=extractor, recall_on_turn=False),
            ),
        )
        session = character.connect("p1", player_name="Ana")
        unknown = "Unattributed voice: I live in Cairo."
        if mixed:
            unknown += "\nSpeaker 1: I grow roses."
        session._on_user_turn(unknown, "unknown")
        await session._handle(
            {"type": "response.created", "response": {"id": "unknown-reply"}}
        )
        session._on_user_turn("I like gardening.", "known")
        assert session._tool_context("restored", {}).player_id == "p1"
        await session._handle(
            {
                "type": "response.output_text.done",
                "response_id": "unknown-reply",
                "text": "You live in Cairo.",
            }
        )
        await session._handle(
            {
                "type": "response.done",
                "response": {"id": "unknown-reply", "status": "completed"},
            }
        )
        await session._handle(
            {"type": "response.created", "response": {"id": "known-reply"}}
        )
        await session._handle(
            {
                "type": "response.output_text.done",
                "response_id": "known-reply",
                "text": "Gardening sounds good.",
            }
        )
        await session._handle(
            {
                "type": "response.done",
                "response": {"id": "known-reply", "status": "completed"},
            }
        )
        await session._learn()
        assert len(extracted) == 1
        assert list(extracted[0].transcript) == (
            ([("player", "I grow roses.")] if mixed else [])
            + [("player", "I like gardening."), ("Guide", "Gardening sounds good.")]
        )
        assert ("Guide", "You live in Cairo.") in session.transcript, (
            "Display keeps the complete transcript"
        )


@pytest.mark.parametrize("input_kind", ["unknown", "known", "mixed"])
@run_async
async def test_late_tool_and_party_reply_keep_the_responses_original_owner(
    api, monkeypatch, input_kind
):
    seen = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def permission(ctx: ToolContext):
        seen.append(("permission", ctx.player_id))
        entered.set()
        await release.wait()
        return True

    async with NurClient(api_key="test", transport=api.transport) as nur:
        character = nur.npc(name="Guide", greeting=False, memory=False, party=True)

        @character.tool(allow=permission)
        async def owner(ctx: ToolContext) -> str:
            """Report the response's owner."""
            seen.append(("body", ctx.player_id))
            return "Done."

        session = character.connect("p1", player_name="Ana")
        monkeypatch.setattr(session, "_send", AsyncMock())
        text = (
            "My words."
            if input_kind == "known"
            else "Unattributed voice: Unknown words."
        )
        if input_kind == "mixed":
            text += "\nSpeaker 1: Known words too."
        session._on_user_turn(text, "first")
        await session._handle({"type": "response.created", "response": {"id": "r1"}})
        expected = "p1" if input_kind == "known" else None
        assert session._responses["r1"].player_id == expected
        newer = "Unattributed voice: New unknown." if expected else "Known now."
        session._on_user_turn(newer, "newer")
        assert session._last_player != expected
        session._on_tool_call(
            {
                "type": "response.function_call_arguments.done",
                "response_id": "r1",
                "call_id": "c1",
                "name": "owner",
                "arguments": "{}",
            }
        )
        await asyncio.wait_for(entered.wait(), 1.0)
        # A pending permission callback owns the same immutable context as the
        # eventual tool body, even with later user events in the receive loop.
        session._on_user_turn(newer, "newer-again")
        release.set()
        await asyncio.gather(*list(session._tasks))
        assert seen == [("permission", expected), ("body", expected)]
        await session._handle(
            {
                "type": "response.output_text.done",
                "response_id": "r1",
                "text": "Old reply.",
            }
        )
        await session._handle(
            {"type": "response.done", "response": {"id": "r1", "status": "completed"}}
        )
        label = "Ana" if expected else "Unattributed voice"
        assert f"Guide to {label}: Old reply." in character.party.render()
        assert not session._responses["r1"].outstanding


@pytest.mark.parametrize("final_unknown", [False, True])
@run_async
async def test_early_response_before_final_stt_stays_unassigned(
    api, monkeypatch, final_unknown
):
    owners, extracted = [], []

    async def extractor(request):
        extracted.append(request)
        return ExtractionResult()

    async with NurClient(api_key="test", transport=api.transport) as nur:
        character = nur.npc(
            name="Guide",
            greeting=False,
            party=True,
            memory=Memory(
                InMemoryStore(),
                policy=MemoryPolicy(extractor=extractor, recall_on_turn=False),
            ),
        )

        @character.tool()
        async def owner(ctx: ToolContext) -> str:
            """Report the response's owner."""
            owners.append(ctx.player_id)
            return "Done."

        session = character.connect("p1", player_name="Ana")
        monkeypatch.setattr(session, "_send", AsyncMock())
        session._on_user_turn("Previous known words.", "previous")
        await session._handle(
            {"type": "response.created", "response": {"id": "established"}}
        )
        await session._handle(
            {
                "type": "input_audio_buffer.speech_started",
                "item_id": "pending",
                "interrupt_response": False,
            }
        )
        await session._handle(
            {
                "type": "response.created",
                "response": {"id": "early", "metadata": {"nur_act": "backchannel"}},
            }
        )
        await session._handle(
            {"type": "input_audio_buffer.speech_stopped", "item_id": "pending"}
        )
        await session._handle(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "pending",
                "transcript": "Unattributed voice: New words."
                if final_unknown
                else "New known words.",
            }
        )
        for rid in ("established", "early", "future"):
            session._on_tool_call(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": rid,
                    "call_id": rid,
                    "name": "owner",
                    "arguments": "{}",
                }
            )
            await asyncio.gather(*list(session._tasks))
        assert owners == ["p1", None, None if final_unknown else "p1"]
        await session._handle(
            {
                "type": "response.output_text.done",
                "response_id": "early",
                "text": "Early reply to the pending voice.",
            }
        )
        await session._handle(
            {
                "type": "response.done",
                "response": {"id": "early", "status": "completed"},
            }
        )
        assert "Guide to Unattributed voice: Early reply" in character.party.render()
        await session._learn()
        assert extracted
        assert all(
            words != "Early reply to the pending voice."
            for request in extracted
            for _, words in request.transcript
        )


@run_async
async def test_stale_finals_and_failures_cannot_replace_newer_input_owner(
    api, monkeypatch
):
    async with NurClient(api_key="test", transport=api.transport) as nur:
        session = nur.npc(name="Guide", greeting=False, memory=False).connect("p1")
        monkeypatch.setattr(session, "_send", AsyncMock())

        async def start(item):
            await session._handle(
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": item,
                    "interrupt_response": False,
                }
            )

        async def final(item, text):
            await session._handle(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item,
                    "transcript": text,
                }
            )

        async def check(rid, owner):
            await session._handle({"type": "response.created", "response": {"id": rid}})
            assert session._responses[rid].player_id == owner
            assert session._responses[rid].unattributed == (owner is None)

        await final("empty-before-start", " ")
        await check("empty-control-stream", None)
        await start("older")
        await start("newer")
        await final("older", "Older known voice.")
        await check("late-old", None)
        await final(None, "Unlabelled known voice.")
        await check("unlabelled", None)
        await session._handle(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "newer",
            }
        )
        await check("failed", None)
        await start("known")
        await final("known", "Known again.")
        await final("older", "Unattributed voice: Old unknown.")
        await check("known-kept", "p1")
        await final("known", " ")
        await check("empty-revision", None)
        await final("known", "Known again after empty revision.")
        await start("known")  # A speculative reopen can reuse this item.
        await check("reopened-early", None)
        await final("known", "Known after reopening.")
        await check("reopened-final", "p1")
        assert session._responses["reopened-early"].player_id is None
        assert session._responses["reopened-early"].unattributed
        await start("before-text")
        await session.send_text("Typed by the configured player.", respond=False)
        await check("typed", "p1")
        await final("before-text", "Unattributed voice: Late audio.")
        await session._handle(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "before-text",
            }
        )
        await check("typed-kept", "p1")
