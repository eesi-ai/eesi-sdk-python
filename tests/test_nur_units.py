"""The SDK's parts on their own: tools, context, memory, extraction, audio, multiplayer, traces."""

from __future__ import annotations

import array
import asyncio
import json
import time
import wave
from enum import Enum
from typing import Annotated, Literal

import pytest
from conftest import run_async
from eesi import ContextError, NurClient, ToolContext, ToolDefinitionError, tool
from eesi.audio import PlayoutClock, Resampler, WavInput, WavOutput
from eesi.context import ContextPublisher, LiveContext, render_fields
from eesi.events import ResponseCompleted, SpeakerTurn, ToolCompleted, TranscriptFinal
from eesi.memory import (
    ALL_PLAYERS,
    CharacterMemory,
    ExtractionResult,
    InMemoryStore,
    LocalMemoryStore,
    Memory,
    MemoryCandidate,
    MemoryPolicy,
    MemoryRecord,
    migrate,
)
from eesi.memory.extraction import ExtractionRequest, LLMExtractor, parse_result
from eesi.memory.recall import compose_block, rank, relative_age
from eesi.multiplayer import PartyLog, SpeakerRoster
from eesi.tools import ToolRunner, render_result
from eesi.trace import SessionMetrics, Trace, format_trace
from pydantic import BaseModel, Field

# ── tools ────────────────────────────────────────────────────────────────


class Mood(Enum):
    calm = "calm"
    angry = "angry"


class Item(BaseModel):
    name: str
    quantity: int = 1


def test_a_tool_describes_itself_from_its_signature_and_docstring():
    @tool
    def trade(
        item: Item,
        price: Annotated[int, Field(ge=1, le=999)],
        mood: Mood = Mood.calm,
        haggle: bool = False,
        ctx: ToolContext | None = None,
    ) -> str:
        """Trade an item with the player.

        Args:
            item: What changes hands.
            price: Gold pieces, agreed aloud.
        """
        return "ok"

    spec = trade.spec()
    params = spec["parameters"]
    assert spec["name"] == "trade"
    assert spec["description"] == "Trade an item with the player."
    assert set(params["properties"]) == {"item", "price", "mood", "haggle"}
    assert params["required"] == ["item", "price"]
    assert params["properties"]["price"]["description"] == "Gold pieces, agreed aloud."
    assert params["properties"]["price"]["minimum"] == 1
    assert params["properties"]["mood"]["enum"] == ["calm", "angry"]
    assert params["properties"]["item"]["properties"]["quantity"]["default"] == 1
    assert "$ref" not in json.dumps(params) and "title" not in json.dumps(params)


def test_a_context_parameter_is_not_described_to_the_model():
    @tool
    def where_am_i(ctx: ToolContext) -> str:
        """Where the player is."""
        return "here"

    assert where_am_i.parameters["properties"] == {}
    assert where_am_i._context_params == ["ctx"]


def test_tools_refuse_what_the_model_cannot_call():
    with pytest.raises(ToolDefinitionError):

        @tool
        def anything(*args: str) -> None: ...

    with pytest.raises(ToolDefinitionError):
        tool(lambda: None, name="has spaces")


def test_a_tool_is_still_the_plain_function():
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert add(2, 3) == 5


def _runner(*tools, events=None):
    return ToolRunner(
        tools,
        context=lambda call_id, args: ToolContext(
            session=None,
            character=None,
            player_id="p1",
            call_id=call_id,
            arguments=args,
        ),  # type: ignore[arg-type]
        on_event=lambda kind, data: (events if events is not None else []).append(
            (kind, data)
        ),
    )


@run_async
async def test_a_sync_tool_runs_off_the_event_loop():
    loop_thread = []

    @tool
    def slow() -> str:
        """Block for a moment."""
        import threading

        loop_thread.append(threading.current_thread().name)
        time.sleep(0.05)
        return "done"

    outcome = await _runner(slow).run("c1", "slow", "{}")
    assert outcome.output == "done"
    assert loop_thread and loop_thread[0] != "MainThread"


@run_async
async def test_a_tool_past_its_timeout_is_reported_as_still_running():
    release = asyncio.Event()

    @tool(timeout=0.05)
    async def dig() -> str:
        """Dig for treasure."""
        await release.wait()
        return "a chest"

    outcome = await _runner(dig).run("c1", "dig", "{}")
    assert "still working" in outcome.output and outcome.later is not None
    release.set()
    assert await outcome.later == "a chest"


@run_async
async def test_an_exception_becomes_a_result_the_character_can_explain():
    events: list = []

    @tool
    def explode() -> str:
        """Try something risky."""
        raise RuntimeError("the powder is wet")

    outcome = await _runner(explode, events=events).run("c1", "explode", "{}")
    assert outcome.output == "That failed: the powder is wet"
    assert events[-1][0] == "tool.failed" and events[-1][1]["reason"] == "exception"


@run_async
async def test_an_app_confirmation_gates_the_action():
    asked = []

    async def ask(ctx: ToolContext) -> bool:
        asked.append(ctx.arguments)
        return ctx.arguments.get("amount", 0) < 100

    @tool(confirm=ask)
    def pay(amount: int) -> str:
        """Pay gold."""
        return f"paid {amount}"

    runner = _runner(pay)
    assert (await runner.run("c1", "pay", '{"amount": 10}')).output == "paid 10"
    assert (
        "did not confirm" in (await runner.run("c2", "pay", '{"amount": 500}')).output
    )
    assert asked == [{"amount": 10}, {"amount": 500}]


def test_results_render_compactly():
    assert render_result(None) == "Done."
    assert render_result({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'
    assert render_result(Item(name="sword")) == '{"name":"sword","quantity":1}'


@run_async
async def test_a_huge_result_is_clipped_to_what_the_gateway_takes():
    @tool
    def dump() -> str:
        """Everything."""
        return "é" * 20_000

    outcome = await _runner(dump).run("c1", "dump", "{}")
    assert len(outcome.output.encode()) <= 16_000 and outcome.output.endswith("…")


# ── context ──────────────────────────────────────────────────────────────


def test_fields_render_as_lines_a_model_reads():
    text = render_fields(
        {
            "location": "Gate",
            "allies": ["Mira", "Tor"],
            "armed": True,
            "stats": {"hp": 6, "max": 20},
        }
    )
    assert (
        text
        == "location: Gate\nallies: Mira, Tor\narmed: yes\nstats.hp: 6\nstats.max: 20"
    )


@run_async
async def test_updates_merge_and_none_removes():
    context = LiveContext("scene")
    await context.update({"location": "Gate", "weather": "rain"})
    await context.update(weather=None, health="6/20")
    assert context.get() == {"location": "Gate", "health": "6/20"}
    assert context.render("scene") == "location: Gate\nhealth: 6/20"


@run_async
async def test_fields_expire():
    context = LiveContext("scene")
    changed = []
    context.subscribe(changed.append)
    await context.update({"alarm": "Guards chase the player"}, ttl=0.05)
    await context.update({"location": "Gate"})
    await asyncio.sleep(0.15)
    assert context.get() == {"location": "Gate"}
    assert changed.count("scene") >= 3


@run_async
async def test_a_block_too_large_or_badly_named_is_refused_with_why():
    context = LiveContext("scene")
    with pytest.raises(ContextError, match="8,000"):
        await context.update({"save": "x" * 9_000})
    with pytest.raises(ContextError, match="lowercase"):
        await context.update({"a": 1}, block="World State")


@run_async
async def test_typed_state_models_are_accepted():
    class Player(BaseModel):
        level: int
        reputation: Literal["trusted", "untrusted"]

    context = LiveContext("scene")
    await context.update(Player(level=12, reputation="untrusted"))
    assert context.render("scene") == "level: 12\nreputation: untrusted"


@run_async
async def test_the_publisher_sends_only_changes_in_a_stable_order_and_throttles():
    sent: list[tuple[str, str]] = []

    async def send(key: str, text: str) -> None:
        sent.append((key, text))

    world, scene = LiveContext("world"), LiveContext("scene")
    publisher = ContextPublisher(send, [world, scene], min_interval=0.1)
    await scene.update({"hp": 20})
    await world.update({"time": "dusk"})
    publisher.set_block("memory", "You remember them.")
    await publisher.start()
    assert [key for key, _ in sent] == ["memory", "world", "scene"]

    sent.clear()
    for hp in range(19, 9, -1):  # a game updating every frame
        await scene.update({"hp": hp})
    await asyncio.sleep(0.25)
    assert sent[-1] == ("scene", "hp: 10")
    assert len(sent) <= 2

    sent.clear()
    await scene.update({"hp": 10})  # same text: nothing to send
    await asyncio.sleep(0.15)
    assert sent == []

    sent.clear()
    await publisher.start()  # a resume: everything again
    assert [key for key, _ in sent] == ["memory", "world", "scene"]
    await publisher.close()


def test_the_publisher_refuses_more_blocks_than_a_session_holds():
    async def send(key: str, text: str) -> None: ...

    publisher = ContextPublisher(send, [])
    with pytest.raises(ContextError, match="at most 8"):
        publisher.validate({f"k{i}": "x" for i in range(9)})
    with pytest.raises(ContextError, match="24,000"):
        publisher.validate({f"k{i}": "x" * 7_000 for i in range(4)})


# ── memory ───────────────────────────────────────────────────────────────


def _memory(store=None, **policy) -> CharacterMemory:
    return CharacterMemory(
        Memory(store or InMemoryStore(), policy=MemoryPolicy(**policy)),
        character_id="kael",
        character_name="Kael",
    )


@pytest.fixture(params=["sqlite", "dict"])
def store(request, tmp_path):
    if request.param == "sqlite":
        made = LocalMemoryStore(tmp_path / "memory")
        yield made
        asyncio.run(made.close())
    else:
        yield InMemoryStore()


@run_async
async def test_players_characters_and_namespaces_never_mix(store):
    kael = _memory(store)
    mira = CharacterMemory(Memory(store), character_id="mira", character_name="Mira")
    other_tenant = CharacterMemory(
        Memory(store, namespace="studio-b"), character_id="kael", character_name="Kael"
    )
    await kael.remember("Ana paid her debt.", player_id="ana")
    await kael.remember("Ben stole bread.", player_id="ben")
    await kael.remember("The bridge fell.")
    await mira.remember("Ana sings well.", player_id="ana")
    await other_tenant.remember("Ana is a stranger here.", player_id="ana")

    assert {r.text for r in await kael.list(player_id="ana")} == {"Ana paid her debt."}
    assert {
        r.text for r, _ in await kael.search("Ana bread debt", player_id="ana")
    } == {"Ana paid her debt."}
    assert {r.text for r, _ in await kael.recall(player_id="ben")} == {
        "Ben stole bread.",
        "The bridge fell.",
    }
    assert {r.text for r in await kael.list()} == {
        "Ana paid her debt.",
        "Ben stole bread.",
        "The bridge fell.",
    }
    assert {r.text for r in await mira.list()} == {"Ana sings well."}
    assert {r.text for r in await other_tenant.list()} == {"Ana is a stranger here."}


@run_async
async def test_a_keyed_preference_replaces_the_last_one_and_keeps_history(store):
    memory = _memory(store)
    first = await memory.remember(
        "Speak French with them.", player_id="p", kind="preference", key="language"
    )
    second = await memory.remember(
        "Speak Arabic with them.", player_id="p", kind="preference", key="language"
    )
    active = await memory.list(player_id="p")
    every = await memory.list(player_id="p", include_inactive=True)
    assert [r.text for r in active] == ["Speak Arabic with them."]
    assert {r.id: r.superseded_by for r in every}[first.id] == second.id


@run_async
async def test_a_correction_supersedes_and_forget_deletes(store):
    memory = _memory(store)
    wrong = await memory.remember("They like coffee.", player_id="p")
    right = await memory.correct(wrong.id, "They like tea, not coffee.")
    assert right is not None and right.source == "correction"
    assert [r.text for r in await memory.list(player_id="p")] == [
        "They like tea, not coffee."
    ]
    assert await memory.forget(right.id) == 1
    assert await memory.get(right.id) is None


@run_async
async def test_erasing_a_player_removes_them_from_the_file(tmp_path):
    store = LocalMemoryStore(tmp_path / "memory")
    memory = _memory(store)
    await memory.remember("Their secret name is Zephyrine.", player_id="p")
    await memory.remember("The bridge fell.")
    assert await memory.erase(player_id="p") == 1
    await store.close()
    raw = (tmp_path / "memory" / "memory.db").read_bytes()
    for suffix in ("-wal", "-shm"):
        wal = tmp_path / "memory" / f"memory.db{suffix}"
        if wal.exists():
            raw += wal.read_bytes()
    assert b"Zephyrine" not in raw
    reopened = _memory(LocalMemoryStore(tmp_path / "memory"))
    assert [r.text for r in await reopened.list()] == ["The bridge fell."]


@run_async
async def test_export_import_and_migration_round_trip(tmp_path):
    source = _memory(LocalMemoryStore(tmp_path / "a"))
    await source.remember(
        "They owe ten gold.", player_id="p", kind="promise", importance=0.9
    )
    await source.remember("The bridge fell.")
    data = await source.export(tmp_path / "export.json")
    assert data["format"] == "eesi.memory" and len(data["records"]) == 2

    target = _memory(InMemoryStore())
    assert await target.import_records(tmp_path / "export.json") == 2
    assert {r.text for r in await target.list()} == {
        "They owe ten gold.",
        "The bridge fell.",
    }

    moved = InMemoryStore()
    assert await migrate(source.store, moved) == 2
    assert (
        len(
            await moved.query(
                namespace="default", character_id="kael", player_ids=ALL_PLAYERS
            )
        )
        == 2
    )


@run_async
async def test_retention_expires_episodes_but_not_facts():
    memory = _memory(retention_days=1)
    event = await memory.remember(
        "They won the archery contest.", player_id="p", kind="event"
    )
    fact = await memory.remember("Their name is Sam.", player_id="p")
    assert event.expires_at is not None and fact.expires_at is None


def test_the_block_keeps_preferences_and_last_time_and_stays_in_budget():
    now = time.time()
    records = [
        MemoryRecord(
            "Keep your answers brief.",
            kind="preference",
            key="verbosity",
            player_id="p",
            created_at=now,
        ),
        MemoryRecord(
            "They are wary of you.", kind="relationship", player_id="p", created_at=now
        ),
        MemoryRecord(
            "They asked about the crown.",
            kind="summary",
            player_id="p",
            created_at=now - 3 * 86_400,
        ),
        *[
            MemoryRecord(
                f"Fact number {i} about them, padded to take some room in the block.",
                player_id="p",
                created_at=now,
            )
            for i in range(200)
        ],
        MemoryRecord("The bridge fell.", kind="lore", created_at=now),
    ]
    text, used = compose_block(
        rank(records),
        character_name="Kael",
        player_name="Sam",
        first_meeting=True,
        budget_bytes=1_500,
    )
    assert len(text.encode()) <= 1_500
    assert "How they like to be spoken to: Keep your answers brief." in text
    assert "Your relationship: They are wary of you." in text
    assert "Last conversation (3 days ago): They asked about the crown." in text
    assert "You have not met" not in text
    assert 3 < len(used) < 200


def test_a_first_meeting_says_so():
    text, used = compose_block(
        [], character_name="Kael", player_name="Sam", first_meeting=True
    )
    assert text.startswith("You have not met Sam before") and used == []


def test_ages_read_like_speech():
    now = time.time()
    assert relative_age(now - 3 * 86_400, now) == "3 days ago"
    assert relative_age(now - 40 * 86_400, now) == "5 weeks ago"
    assert relative_age(now - 400 * 86_400, now) == "13 months ago"


# ── extraction ───────────────────────────────────────────────────────────


def test_an_extraction_is_read_from_a_chatty_fenced_answer():
    ids = {"m1": "mem_old"}
    reply = """Sure! Here it is:
```json
{"summary": "They bought a sword.", "memories": [
  {"text": "They bought a sword from Kael.", "kind": "event", "importance": "high", "key": null, "replaces": null},
  {"text": "Call them Sam.", "kind": "preference", "importance": "medium", "key": "name", "replaces": "m1"},
  {"text": "", "kind": "fact"},
  {"text": "Weird kind", "kind": "secret", "importance": 0.4}
], "forget": ["m1", "m9"]}
```"""
    result = parse_result(reply, ids)
    assert result.summary == "They bought a sword."
    assert [(m.kind, m.importance) for m in result.memories] == [
        ("event", 0.9),
        ("preference", 0.6),
        ("fact", 0.4),
    ]
    assert result.memories[1].key == "name" and result.memories[1].replaces == "mem_old"
    assert result.forget == ("mem_old",)


def test_garbage_is_no_memories_not_a_crash():
    assert parse_result("I cannot help with that.", {}).memories == ()
    assert parse_result("{not json", {}).memories == ()


@run_async
async def test_the_extractor_asks_with_the_transcript_and_existing_memories():
    asked = []

    async def chat(messages):
        asked.append(messages)
        return json.dumps({"summary": "Hi.", "memories": [], "forget": []})

    existing = MemoryRecord("They like tea.", player_id="p")
    result = await LLMExtractor(chat)(
        ExtractionRequest(
            character_name="Luna",
            player_id="p",
            player_name="Sam",
            transcript=[
                ("player", "Hi Luna, I'm Sam."),
                ("Luna", "Hello Sam!"),
                ("event", "Sam picked up a lantern."),
            ],
            existing=[existing],
        )
    )
    content = asked[0][1]["content"]
    assert "m1: They like tea." in content
    assert (
        "Sam: Hi Luna, I'm Sam." in content
        and "Luna: Hello Sam!" in content
        and "[event]: Sam picked up a lantern." in content
    )
    assert result.summary == "Hi."


@run_async
async def test_applying_an_extraction_dedupes_supersedes_and_forgets():
    memory = _memory()
    tea = await memory.remember("They like tea.", player_id="p")
    gone = await memory.remember("They live in Oslo.", player_id="p")
    added, superseded, forgotten = await memory.apply(
        ExtractionResult(
            summary="We talked about drinks.",
            memories=(
                MemoryCandidate("They like tea."),  # already known
                MemoryCandidate("They like green tea most.", replaces=tea.id),
                MemoryCandidate("They have a cat named Miso.", importance=0.9),
            ),
            forget=(gone.id,),
        ),
        player_id="p",
    )
    texts = sorted(r.text for r in await memory.list(player_id="p"))
    assert texts == [
        "They have a cat named Miso.",
        "They like green tea most.",
        "We talked about drinks.",
    ]
    assert [r.id for r in superseded] == [tea.id]
    assert forgotten == [gone.id]
    assert len(added) == 3


@run_async
async def test_the_oldest_least_important_memories_go_first_past_the_cap():
    memory = _memory(max_records_per_player=5)
    for index in range(8):
        await memory.remember(f"Trivia {index}.", player_id="p", importance=0.1)
    keep = await memory.remember("Their mother is ill.", player_id="p", importance=1.0)
    await memory._prune("p")
    remaining = await memory.list(player_id="p", include_inactive=True)
    assert len(remaining) == 5 and keep.id in {r.id for r in remaining}


@run_async
async def test_redaction_runs_before_anything_is_stored():
    memory = _memory(redact=lambda text: text.replace("sam@example.com", "[email]"))
    record = await memory.remember("Their email is sam@example.com.", player_id="p")
    assert record.text == "Their email is [email]."


# ── audio ────────────────────────────────────────────────────────────────


def _pcm(values) -> bytes:
    return array.array("h", values).tobytes()


@pytest.mark.parametrize(
    "src, dst", [(48_000, 16_000), (16_000, 24_000), (44_100, 16_000), (24_000, 48_000)]
)
def test_resampling_does_not_depend_on_chunking(src, dst):
    import math

    signal = _pcm(int(8_000 * math.sin(i / 7.0)) for i in range(src // 10))
    whole = Resampler(src, dst).process(signal)
    chunked = Resampler(src, dst)
    pieces = b"".join(
        chunked.process(signal[i : i + 2 * 137]) for i in range(0, len(signal), 2 * 137)
    )
    assert whole == pieces
    assert abs(len(whole) // 2 - (len(signal) // 2) * dst / src) <= 2


def test_the_pure_python_resampler_matches_numpy(monkeypatch):
    import eesi.audio as audio_module

    signal = _pcm((i * 37) % 2000 - 1000 for i in range(4_800))
    with_numpy = Resampler(48_000, 16_000).process(signal)
    monkeypatch.setattr(audio_module, "_np", None)
    without = Resampler(48_000, 16_000).process(signal)
    assert with_numpy == without


def test_the_playout_clock_counts_down_and_flushes():
    now = [100.0]
    clock = PlayoutClock(24_000, clock=lambda: now[0])
    clock.push(24_000)
    now[0] += 0.25
    assert clock.buffered_seconds() == pytest.approx(0.75)
    clock.flush()
    assert clock.buffered_seconds() == 0.0


@run_async
async def test_a_wav_is_heard_in_real_time_chunks_and_a_reply_is_written(tmp_path):
    path = tmp_path / "in.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(_pcm([100, 300] * 1_600))  # 0.1 s stereo
    source = WavInput(path, tail=0.04)
    started = time.monotonic()
    frames = [frame async for frame in source.frames()]
    assert len(frames) == 7 and all(len(frame) == 640 for frame in frames)
    assert array.array("h", frames[0])[0] == 200  # stereo averaged
    assert time.monotonic() - started >= 0.1

    out = WavOutput(tmp_path / "out.wav")
    out.play(_pcm([500] * 2_400))
    out.close()
    with wave.open(str(tmp_path / "out.wav"), "rb") as wav:
        # The reply sits where it played on the wall clock: a few samples after the file began.
        assert wav.getframerate() == 24_000 and 2_400 <= wav.getnframes() <= 2_400 + 240


# ── multiplayer ──────────────────────────────────────────────────────────


def test_voices_are_bound_to_players_by_introduction_and_order():
    roster = SpeakerRoster(["Layla", "Sam"])
    turns = roster.attribute("Speaker 1: Hi, I'm Sam.\nSpeaker 2: And I'm Leila!")
    assert [(t.speaker, t.player) for t in turns] == [(1, "Sam"), (2, "Layla")]
    turns = roster.attribute("Speaker 2: Where is the key?")
    assert turns[0].player == "Layla"
    assert "Speaker 1 is Sam. Speaker 2 is Layla." in roster.render()


def test_a_voice_addressing_someone_is_not_them():
    roster = SpeakerRoster(["Layla", "Sam"])
    turns = roster.attribute("Speaker 1: Layla, you go up the stairs.")
    assert turns[0].player == "Sam"


def test_the_party_log_shows_each_session_the_others():
    log = PartyLog(max_lines=3)
    log.heard("Sam", "Where did Ana go?", origin="s1")
    log.said("Kael", "Ana", "Wait by the mill.", origin="s2")
    assert "Kael to Ana: Wait by the mill." in log.render(exclude="s1")
    assert "Where did Ana go?" not in log.render(exclude="s1")
    for index in range(5):
        log.add(f"line {index}")
    assert log.render().count("\n") == 3


# ── traces ───────────────────────────────────────────────────────────────


def test_a_trace_holds_no_words_unless_asked(tmp_path):
    private = Trace(tmp_path / "private.jsonl")
    private.open()
    private.event(
        TranscriptFinal(
            text="my password is swordfish",
            turns=(SpeakerTurn(text="my password is swordfish"),),
        )
    )
    private.event(
        ToolCompleted(
            name="open_gate", arguments={"gate": "north"}, output="open", result="open"
        )
    )
    private.close()
    written = (tmp_path / "private.jsonl").read_text()
    assert (
        "swordfish" not in written
        and "north" not in written
        and '"chars": 24' in written
    )

    debug = Trace(tmp_path / "debug.jsonl", content=True)
    debug.open()
    debug.event(
        TranscriptFinal(
            text="hello", turns=(SpeakerTurn(text="hello", player_id="p1"),)
        )
    )
    debug.event(ResponseCompleted(text="Hi there.", latency=0.61))
    debug.close()
    lines = [
        json.loads(line) for line in (tmp_path / "debug.jsonl").read_text().splitlines()
    ]
    rendered = format_trace(lines)
    assert (
        "heard     p1: hello" in rendered
        and "said (0.61s)" in rendered
        and "Hi there." in rendered
    )


def test_metrics_summarise_latency():
    metrics = SessionMetrics(latencies=[0.5, 0.7, 0.6, 2.0])
    summary = metrics.summary()
    assert summary["latency_p50"] == 0.7 and summary["latency_p95"] == 2.0


# ── the client ───────────────────────────────────────────────────────────


def test_the_client_needs_a_key_and_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("EESI_API_KEY", raising=False)
    from eesi import ConfigurationError

    with pytest.raises(ConfigurationError, match="EESI_API_KEY"):
        NurClient()


def test_urls_are_derived_from_the_base_url():
    nur = NurClient(api_key="k", base_url="https://api.example.test/v1/")
    assert nur.base_url == "https://api.example.test"
    assert nur.realtime_url == "wss://api.example.test/v1/realtime"
    local = NurClient(api_key="k", base_url="http://localhost:8000")
    assert local.realtime_url == "ws://localhost:8000/v1/realtime"


@run_async
async def test_an_unknown_voice_lists_the_real_ones(api):
    from eesi import ConfigurationError

    nur = NurClient(api_key="k", base_url="https://x.test", transport=api.transport)
    assert await nur.resolve_voice("atlas") == "ev_atlas001"
    with pytest.raises(ConfigurationError, match="Available: Atlas, Nur"):
        await nur.resolve_voice("fantasy_male")


def test_a_character_saves_and_loads(tmp_path):
    nur = NurClient(api_key="k", base_url="https://x.test")

    @tool
    def open_gate() -> str:
        """Open the gate."""
        return "open"

    kael = nur.npc(
        name="Kael",
        persona="An elven merchant.",
        traits=["wry"],
        boundaries=["Never reveal the vault."],
        tools=[open_gate],
        context={"realm": "North"},
        version="1.2",
    )
    kael.save(tmp_path / "kael.json")
    loaded = nur.load_character(tmp_path / "kael.json", tools=[open_gate])
    assert loaded.instructions_text() == kael.instructions_text()
    assert [t.name for t in loaded.tools] == ["open_gate"]
    assert loaded.context.get("world") == {"realm": "North"}
    variant = kael.clone(name="Kaela", id="kaela")
    assert (
        variant.instructions_text().startswith("You are Kaela")
        and variant.tools == kael.tools
    )


def test_instructions_compose_persona_rules_and_spoken_style(tmp_path):
    nur = NurClient(api_key="k", base_url="https://x.test", memory_dir=tmp_path)
    luna = nur.companion(
        name="Luna",
        persona="A curious, thoughtful companion.",
        language=["en", "ar"],
        goals=["Help them keep their running habit."],
        knowledge=["Their running club meets on Sundays."],
        examples=["Oh? Tell me more."],
    )
    text = luna.instructions_text()
    assert text.startswith(
        "You are Luna, a companion the person you're talking with comes back to often. A curious"
    )
    assert "You speak English and Arabic" in text
    assert "What you want:\n- Help them keep their running habit." in text
    assert "no lists, markdown" in text
    assert luna.memory is not None, "a companion remembers by default"


# ── a game's own memory rules, and learning from a webhook ──────────────


@run_async
async def test_memory_rules_are_the_last_word_in_the_extraction_prompt():
    from eesi.memory.extraction import ExtractionRequest, LLMExtractor

    seen: list[list[dict]] = []

    async def chat(messages):
        seen.append(messages)
        return '{"summary": "", "memories": [], "forget": []}'

    request = ExtractionRequest(
        character_name="Sage",
        player_id="p1",
        player_name=None,
        transcript=[("player", "I'm building a treehouse.")],
        existing=[],
    )
    await LLMExtractor(chat, rules="Only game facts. Never age, school or location.")(
        request
    )
    await LLMExtractor(chat)(request)

    assert seen[0][0]["content"].endswith(
        "\n\nWhat Sage may remember, set by the game, which wins over the rules above "
        "where they differ:\nOnly game facts. Never age, school or location."
    )
    assert "set by the game" not in seen[1][0]["content"]


@run_async
async def test_a_finished_call_is_learned_from_its_webhook():
    seen: list = []

    async def extractor(request):
        seen.append(request)
        return ExtractionResult(
            summary="Talked about a treehouse.",
            memories=(
                MemoryCandidate(
                    text="They are building a treehouse.", kind="fact", importance=0.6
                ),
            ),
        )

    nur = NurClient(api_key="sk-eesi-test", base_url="http://127.0.0.1:9")
    sage = nur.npc(
        name="Sage",
        memory=Memory(InMemoryStore(), policy=MemoryPolicy(extractor=extractor)),
    )
    event = {
        "type": "session.ended",
        "session_id": "conv_1",
        "transcript": [
            {"role": "player", "text": "I'm building a treehouse."},
            {"role": "character", "text": "Use oak for the floor."},
        ],
    }

    added, _, _ = await sage.learn_from_session(event, player_id="p-123")

    assert list(seen[0].transcript) == [
        ("player", "I'm building a treehouse."),
        ("Sage", "Use oak for the floor."),
    ]
    assert {record.text for record in added} == {
        "They are building a treehouse.",
        "Talked about a treehouse.",
    }
    assert all(record.player_id == "p-123" for record in added)
    assert await sage.learn_from_session(
        {"type": "response.completed"}, player_id="p-123"
    ) == ([], [], [])
