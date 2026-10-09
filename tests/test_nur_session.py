"""A session against a fake Nur Live gateway: the wire the SDK actually speaks.

These tests run the real WebSocket client against a local server that
behaves like the gateway's protocol surface. They prove what the SDK sends
and how it reacts to events; they cannot prove anything about speech, the
model, echo, or network conditions. ``test_nur_live.py`` does that against a
real deployment when credentials are present.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Literal

import pytest
from conftest import FakeAPI, run_async
from eesi import AuthenticationError, NurClient, Started, ToolContext
from eesi.audio import PlayoutClock
from fake_gateway import FakeGateway


class FakeSpeaker:
    """An output that plays in real time, for barge-in tests."""

    sample_rate = 24_000

    def __init__(self) -> None:
        self.clock = PlayoutClock(self.sample_rate)
        self.played = 0
        self.flushes = 0
        self.ducked: list[bool] = []

    def play(self, pcm: bytes) -> None:
        self.played += len(pcm) // 2
        self.clock.push(len(pcm) // 2)

    def flush(self) -> None:
        self.flushes += 1
        self.clock.flush()

    def duck(self, ducked: bool = True) -> None:
        self.ducked.append(ducked)

    def buffered_seconds(self) -> float:
        return self.clock.buffered_seconds()


def client(gateway: FakeGateway, api: FakeAPI) -> NurClient:
    return NurClient(
        api_key="sk-eesi-test",
        base_url=gateway.base_url,
        transport=api.transport,
        timezone="Europe/Berlin",
    )


@run_async
async def test_connect_sends_the_character_then_its_state_then_a_greeting(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(
            name="Kael",
            persona="A mysterious elven merchant who distrusts kings.",
            voice="Atlas",
            memory=nur.memory.in_memory(),
            context={"realm": "The Forgotten Kingdom"},
            hotwords=["Mira"],
        )

        @kael.tool
        def open_gate(gate: Literal["north", "south"] = "north") -> str:
            """Open a castle gate."""
            return f"The {gate} gate is open."

        session = kael.connect(
            "player_42", player_name="Sam", context={"location": "Northern Gate"}
        )
        started = []
        session.on("session.started", started.append)
        async with session:
            greeting = await gateway.wait_type("response.create")
            update = gateway.of_type("session.update")[0]["session"]
            contexts = gateway.of_type("eesi.context.update")
            await session.close(learn=False)

        assert gateway.headers[0]["Authorization"] == "Bearer sk-eesi-test"
        assert (
            "model=nur-live-v1" in gateway.paths[0] and "source=sdk" in gateway.paths[0]
        )
        assert "timezone=Europe%2FBerlin" in gateway.paths[0]
        assert update["instructions"].startswith(
            "You are Kael, a character who lives in this world."
        )
        assert update["audio"]["output"]["voice"] == "ev_atlas001"
        assert update["audio"]["output"]["format"] == {
            "type": "audio/pcm",
            "rate": 24000,
        }
        assert update["audio"]["input"]["turn_detection"]["interrupt_response"] is True
        assert "create_response" not in update["audio"]["input"]["turn_detection"]
        assert update["audio"]["input"]["transcription"]["prompt"] == "Kael, Mira, Sam"
        assert [tool["name"] for tool in update["tools"]] == ["open_gate"]
        assert update["tool_choice"] == "auto"
        assert "reasoning" not in update
        # Slow blocks first, so the prompt in front of the fast ones stays cached.
        assert [frame["key"] for frame in contexts] == ["memory", "world", "scene"]
        assert "You have not met Sam before" in contexts[0]["text"]
        assert contexts[1]["text"] == "realm: The Forgotten Kingdom"
        assert contexts[2]["text"] == "location: Northern Gate"
        assert "greet" in greeting["response"]["instructions"]
        assert gateway.frames.index(greeting) > gateway.frames.index(contexts[-1])
        assert started and started[0].session_id == gateway.session_id


@run_async
async def test_a_refused_key_raises_at_connect(api):
    async with FakeGateway(refuse_status=401) as gateway:
        nur = client(gateway, api)
        with pytest.raises(AuthenticationError):
            async with nur.character("Luna", greeting=False).connect():
                pass


@run_async
async def test_a_tool_call_runs_answers_once_and_continues_once(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        seen: list[ToolContext] = []

        @kael.tool
        async def open_gate(gate: str, ctx: ToolContext) -> dict:
            """Open a castle gate."""
            seen.append(ctx)
            return {"opened": gate}

        events = []
        async with kael.connect("p1") as session:
            session.on("tool.*", events.append)
            await gateway.send({"type": "response.created", "response": {"id": "r1"}})
            for _ in range(2):  # the same call delivered twice must run once
                await gateway.send(
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": "r1",
                        "call_id": "c1",
                        "name": "open_gate",
                        "arguments": '{"gate": "north"}',
                    }
                )
            output = await gateway.wait_type("conversation.item.create")
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": "completed"},
                }
            )
            follow = await gateway.wait_type("response.create")
            await asyncio.sleep(0.2)

        assert output["item"] == {
            "type": "function_call_output",
            "call_id": "c1",
            "output": '{"opened":"north"}',
        }
        assert follow.get("response") is None, (
            "a plain create resumes the request that asked for the tool"
        )
        assert len(gateway.of_type("response.create")) == 1
        assert [e.type for e in events] == ["tool.started", "tool.completed"]
        assert seen[0].player_id == "p1" and seen[0].call_id == "c1"


@run_async
async def test_a_spoken_action_is_not_narrated_twice(api):
    """A reply that already said 'opening it' and called a tool returning None needs no follow-up."""
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)

        @kael.tool
        def wave() -> None:
            """Wave at the player."""

        async with kael.connect("p1"):
            await gateway.reply("r1", "Hello there!", done=False)
            await gateway.send(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "r1",
                    "call_id": "c1",
                    "name": "wave",
                    "arguments": "{}",
                }
            )
            await gateway.wait_type("conversation.item.create")
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": "completed"},
                }
            )
            await asyncio.sleep(0.3)

        assert gateway.of_type("response.create") == []


@run_async
async def test_bad_arguments_and_refusals_are_answered_not_run(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        ran = []

        @kael.tool(
            allow=lambda ctx: (
                "the gate is for allies" if ctx.player_id == "thief" else True
            )
        )
        def open_gate(gate: Literal["north", "south"]) -> str:
            """Open a castle gate."""
            ran.append(gate)
            return "open"

        @kael.tool(confirm=True)
        def sell_sword(price: int) -> str:
            """Sell the player the sword."""
            ran.append(price)
            return "sold"

        failures = []
        async with kael.connect("thief") as session:
            session.on("tool.failed", failures.append)
            calls = [
                ("c1", "open_gate", '{"gate": "east"}'),
                ("c2", "open_gate", '{"gate": "north"}'),
                ("c3", "sell_sword", '{"price": 10}'),
                ("c4", "sell_sword", '{"price": "ten", "confirmed": true}'),
                ("c5", "fly", "{}"),
            ]
            for call_id, name, args in calls:
                await gateway.send(
                    {
                        "type": "response.function_call_arguments.done",
                        "call_id": call_id,
                        "name": name,
                        "arguments": args,
                    }
                )
            for _ in range(5):
                await gateway.wait_for(
                    lambda f: (
                        f.get("type") == "conversation.item.create"
                        and len(gateway.of_type("conversation.item.create")) >= 5
                    )
                )
            outputs = {
                f["item"]["call_id"]: f["item"]["output"]
                for f in gateway.of_type("conversation.item.create")
            }

        assert ran == []
        assert "not valid" in outputs["c1"] and "gate" in outputs["c1"]
        assert outputs["c2"].startswith("Not allowed: the gate is for allies")
        assert "confirm" in outputs["c3"].lower()
        assert "not valid" in outputs["c4"]
        assert "No tool named fly" in outputs["c5"]
        assert [f.reason for f in failures] == [
            "invalid_arguments",
            "not_allowed",
            "not_confirmed",
            "invalid_arguments",
            "unknown_tool",
        ]


@run_async
async def test_a_slow_tool_says_started_and_reports_back_later(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        finish = asyncio.Event()

        @kael.tool
        async def fetch_mead() -> Started:
            """Fetch a mead from the cellar."""

            async def walk() -> str:
                await finish.wait()
                return "The mead is on the counter."

            return Started("Off to the cellar.", asyncio.ensure_future(walk()))

        async with kael.connect("p1"):
            await gateway.send({"type": "response.created", "response": {"id": "r1"}})
            await gateway.send(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "r1",
                    "call_id": "c1",
                    "name": "fetch_mead",
                    "arguments": "{}",
                }
            )
            first = await gateway.wait_type("conversation.item.create")
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": "completed"},
                }
            )
            await gateway.wait_type("response.create")
            await gateway.send({"type": "response.created", "response": {"id": "r2"}})
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r2", "status": "completed"},
                }
            )
            count = len(gateway.frames)
            finish.set()
            news = await gateway.wait_for(
                lambda f: (
                    f.get("type") == "response.create"
                    and gateway.frames.index(f) >= count
                )
            )

        assert first["item"]["output"] == "Off to the cellar."
        event = news["response"]["input"][0]["content"][0]["text"]
        assert (
            "fetch_mead you started earlier finished" in event
            and "on the counter" in event
        )


@run_async
async def test_barge_in_drops_queued_audio_and_late_audio(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(name="Luna", greeting=False, memory=False)
        speaker = FakeSpeaker()
        interrupted = []
        async with luna.connect("p1", output=speaker) as session:
            session.on("speech.interrupted", interrupted.append)
            await gateway.reply(
                "r1", "Let me tell you a long story", audio_ms=2000, done=False
            )
            await asyncio.sleep(0.1)
            assert speaker.buffered_seconds() > 1.0
            await gateway.send(
                {"type": "input_audio_buffer.speech_started", "item_id": "i2"}
            )
            await asyncio.sleep(0.1)
            assert speaker.flushes >= 1 and speaker.buffered_seconds() == 0.0
            played = speaker.played
            # Audio still in flight for the cut reply is never played.
            await gateway.send(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "r1",
                    "delta": base64.b64encode(b"\x01\x00" * 4800).decode(),
                }
            )
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {
                        "id": "r1",
                        "status": "cancelled",
                        "status_details": {"reason": "turn_detected"},
                    },
                }
            )
            cancelled = await session.wait("response.cancelled", timeout=2)
            assert speaker.played == played
            status = await gateway.wait_type("eesi.output.status", timeout=2)

        assert len(interrupted) == 1 and interrupted[0].response_id == "r1"
        assert cancelled.reason == "turn_detected" and "long story" in cancelled.text
        assert (
            status["playback"]["response_id"] == "r1"
            and status["playback"]["flushed"] is True
        )


@run_async
async def test_a_held_overlap_ducks_instead_of_cutting(api):
    """interrupt_response false means the runtime is deciding: duck, then resume."""
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(name="Luna", greeting=False, memory=False)
        speaker = FakeSpeaker()
        async with luna.connect("p1", output=speaker):
            await gateway.reply("r1", "So I said to him", audio_ms=1500, done=False)
            await asyncio.sleep(0.1)
            await gateway.send(
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "i2",
                    "interrupt_response": False,
                }
            )
            await gateway.send({"type": "eesi.output.resume"})
            await asyncio.sleep(0.1)
            assert speaker.flushes == 0, "a held overlap must not drop the reply"
            assert speaker.buffered_seconds() > 0.5

        assert speaker.ducked[:2] == [True, False]


@run_async
async def test_memory_is_learned_after_a_session_and_recalled_in_the_next(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        memory = nur.memory.in_memory()
        luna = nur.companion(name="Luna", greeting=False, memory=memory)
        api.chat_replies.append(
            json.dumps(
                {
                    "summary": "Sam told me about their new job at the observatory.",
                    "memories": [
                        {
                            "text": "Sam started a new job at the observatory.",
                            "kind": "fact",
                            "importance": "high",
                            "key": None,
                            "replaces": None,
                        },
                        {
                            "text": "Sam prefers short answers.",
                            "kind": "preference",
                            "importance": "medium",
                            "key": "verbosity",
                            "replaces": None,
                        },
                    ],
                    "forget": [],
                }
            )
        )
        learned = []
        session = luna.connect("sam", player_name="Sam")
        session.on("memory.updated", learned.append)
        async with session:
            await gateway.player_says("I started at the observatory today!")
            await gateway.reply("r1", "That's wonderful, congratulations!")
            await session.wait("response.completed")

        assert learned and {r.text for r in learned[0].added} >= {
            "Sam started a new job at the observatory."
        }
        chat_body = json.loads(api.requests[-1].content)
        assert (
            "I started at the observatory today!" in chat_body["messages"][1]["content"]
        )

        frames_before = len(gateway.frames)
        async with luna.connect("sam", player_name="Sam"):
            block = (
                await gateway.wait_for(
                    lambda f: (
                        f.get("type") == "eesi.context.update"
                        and gateway.frames.index(f) >= frames_before
                    )
                )
            )["text"]
        assert "observatory" in block
        assert (
            "Keep your answers brief" not in block
        )  # the extracted phrase is kept as written
        assert "Sam prefers short answers." in block
        assert "Last conversation" in block

        frames_before = len(gateway.frames)
        async with luna.connect("someone_else", player_name="Ada"):
            other = (
                await gateway.wait_for(
                    lambda f: (
                        f.get("type") == "eesi.context.update"
                        and gateway.frames.index(f) >= frames_before
                    )
                )
            )["text"]
        assert "observatory" not in other and "You have not met Ada before" in other


@run_async
async def test_what_the_player_says_brings_a_memory_to_mind(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False, memory=nur.memory.in_memory())
        # More than the memory block holds, so the low-importance one is left out at the start.
        for index in range(120):
            await kael.remember(
                f"Small talk number {index}: they mentioned that the weather in the valley was mild that day.",
                player_id="p1",
                importance=0.9,
            )
        await kael.remember(
            "They betrayed the northern alliance at the river ford.",
            player_id="p1",
            importance=0.2,
        )
        retrieved = []
        async with kael.connect("p1") as session:
            session.on("memory.retrieved", retrieved.append)
            await gateway.player_says("Do you remember the northern alliance?")
            recall = await gateway.wait_for(
                lambda f: (
                    f.get("type") == "eesi.context.update" and f.get("key") == "recall"
                )
            )
        assert "betrayed the northern alliance" in recall["text"]
        assert any("alliance" in r.text for event in retrieved for r in event.records)


@run_async
async def test_app_turn_taking_asks_for_each_reply_after_the_turn_settles(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False, turn_taking="app")
        async with kael.connect("p1"):
            update = gateway.of_type("session.update")[0]["session"]
            await gateway.player_says("Open the gate")
            await asyncio.sleep(0.2)
            assert gateway.of_type("response.create") == []
            create = await gateway.wait_type("response.create", timeout=2)
        assert update["audio"]["input"]["turn_detection"]["create_response"] is False
        assert create["response"] == {"tool_choice": "auto"}


@run_async
async def test_a_lost_connection_resumes_and_restores_the_character_and_state(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        events = []
        async with kael.connect("p1") as session:
            session.on("*", events.append)
            await session.context.update({"location": "Northern Gate"})
            await asyncio.sleep(0.4)
            await gateway.drop(1011)
            await session.wait("reconnected", timeout=10)
            updates = gateway.of_type("session.update")
            contexts = [
                f for f in gateway.of_type("eesi.context.update") if f["key"] == "scene"
            ]

        assert gateway.connections == 2
        assert (
            f"resume={gateway.session_id}" in gateway.paths[1]
            and "resume_token=resume-secret" in gateway.paths[1]
        )
        assert len(updates) == 2 and updates[0] == updates[1]
        assert contexts[-1]["text"] == "location: Northern Gate"
        kinds = [e.type for e in events]
        assert kinds.index("reconnecting") < kinds.index("reconnected")
        assert next(e for e in events if e.type == "reconnected").resumed is True


@run_async
async def test_a_partial_update_keeps_the_voice(api):
    """The gateway pins the default voice on a session.update that names none."""
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(
            name="Luna", greeting=False, voice="Atlas", memory=nur.memory.in_memory()
        )
        async with luna.connect("p1") as session:
            await session.preferences.set(pace="slow")
            await session.set_tools_enabled("anything", False)
            await asyncio.sleep(0.1)
        updates = [f["session"] for f in gateway.of_type("session.update")]
        assert len(updates) == 3
        assert all(
            update["audio"]["output"]["voice"] == "ev_atlas001" for update in updates
        )
        assert updates[1]["audio"]["output"]["speed"] == 0.9


@run_async
async def test_cues_wait_for_the_floor(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        async with kael.connect("p1") as session:
            await gateway.send(
                {"type": "input_audio_buffer.speech_started", "item_id": "i1"}
            )
            await asyncio.sleep(0.05)
            await session.cue("The player drew a sword.")
            await asyncio.sleep(0.3)
            assert gateway.of_type("response.create") == []
            await gateway.send(
                {"type": "input_audio_buffer.speech_stopped", "item_id": "i1"}
            )
            create = await gateway.wait_type("response.create", timeout=3)
        text = create["response"]["input"][0]["content"][0]["text"]
        assert text == "[Game event, not the player speaking] The player drew a sword."


@run_async
async def test_two_players_of_one_character_never_share_memories(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False, memory=nur.memory.in_memory())
        await kael.remember("Ana paid her debt in gold.", player_id="ana")
        await kael.remember("Ben stole a loaf of bread.", player_id="ben")
        await kael.remember("The bridge to the north collapsed.")
        blocks = {}
        for player in ("ana", "ben"):
            count = len(gateway.frames)
            async with kael.connect(player):
                frame = await gateway.wait_for(
                    lambda f, after=count: (
                        f.get("type") == "eesi.context.update"
                        and f.get("key") == "memory"
                        and gateway.frames.index(f) >= after
                    )
                )
                blocks[player] = frame["text"]
        assert "Ana paid" in blocks["ana"] and "Ben stole" not in blocks["ana"]
        assert "Ben stole" in blocks["ben"] and "Ana paid" not in blocks["ben"]
        assert (
            "bridge to the north collapsed" in blocks["ana"]
            and "bridge to the north collapsed" in blocks["ben"]
        )


@run_async
async def test_a_hangup_ends_the_session_after_the_goodbye_plays(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(name="Luna", greeting=False, memory=False)
        speaker = FakeSpeaker()
        session = luna.connect("p1", output=speaker)
        ended = []
        session.on("session.ended", ended.append)
        async with session:
            await gateway.reply("r1", "Goodbye, friend.", audio_ms=500)
            await gateway.send({"type": "eesi.hangup", "reason": "model_requested"})
            await asyncio.wait_for(session.run(), 5)
        assert ended and ended[0].reason == "hangup"


@run_async
async def test_client_secret_carries_the_character_memory_and_context(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(
            name="Kael",
            persona="An elven merchant.",
            voice="Atlas",
            memory=nur.memory.in_memory(),
            context={"realm": "North"},
        )
        await kael.remember("They owe Kael ten gold.", player_id="p42")

        secret = await kael.client_secret(
            "p42", context={"location": "Gate"}, expires_in=90
        )
        locked = await kael.client_secret("p43", lock="all")
        assert kael.context._listeners == [], (
            "minting a secret leaves nothing subscribed"
        )
        assert secret.lock == ("instructions", "tools")
        assert api.secrets[1]["eesi_lock"] == "all"
        assert "eesi_lock" not in api.secrets[0]
        assert locked.lock == ("all",)

        body = api.secrets[0]
        assert body["expires_after"] == {"anchor": "created_at", "seconds": 90}
        assert body["session"]["instructions"].startswith("You are Kael")
        assert body["session"]["audio"]["output"]["voice"] == "ev_atlas001"
        assert list(body["eesi_context"]) == ["memory", "world", "scene"]
        assert "ten gold" in body["eesi_context"]["memory"]
        assert body["eesi_context"]["scene"] == "location: Gate"
        assert secret.url.startswith(f"ws://127.0.0.1:{gateway.port}/v1/realtime?")
        assert "token=rt_secret" in secret.url


@run_async
async def test_a_party_character_tells_each_player_what_it_told_the_others(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.multiplayer_npc(name="Kael", greeting=False)
        async with kael.connect("ana", player_name="Ana") as ana:
            await gateway.player_says("Where can I find the smith?")
            await gateway.reply("r1", "The smith works by the mill, Ana.")
            await ana.wait("response.completed")
            count = len(gateway.frames)
            async with kael.connect("ben", player_name="Ben"):
                party = await gateway.wait_for(
                    lambda f: (
                        f.get("type") == "eesi.context.update"
                        and f.get("key") == "party"
                        and gateway.frames.index(f) >= count
                    )
                )
        assert "Ana: Where can I find the smith?" in party["text"]
        assert "Kael to Ana: The smith works by the mill, Ana." in party["text"]


@run_async
async def test_shared_microphone_turns_are_attributed_to_players(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False, memory=nur.memory.in_memory())
        callers = []

        @kael.tool
        def whoami(ctx: ToolContext) -> str:
            """Say who is asking."""
            callers.append(ctx.player_id)
            return ctx.player_id or "unknown"

        async with kael.connect(players={"Layla": "p1", "Sam": "p2"}) as session:
            await gateway.player_says(
                "Speaker 1: Hi, I'm Sam.\nSpeaker 2: And I'm Layla."
            )
            final = await session.wait("transcript.final")
            roster = await gateway.wait_for(
                lambda f: (
                    f.get("type") == "eesi.context.update" and f.get("key") == "roster"
                )
            )
            await gateway.player_says("Speaker 2: Who am I?", item_id="i2")
            await session.wait("transcript.final")
            await gateway.send(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "c1",
                    "name": "whoami",
                    "arguments": "{}",
                }
            )
            await gateway.wait_type("conversation.item.create")
        assert [(t.speaker, t.player_id) for t in final.turns] == [(1, "p2"), (2, "p1")]
        assert "Speaker 1 is Sam. Speaker 2 is Layla." in roster["text"]
        assert callers == ["p1"]


@run_async
async def test_a_line_typed_during_a_reply_is_sent_at_once_and_superseding_is_quiet(
    api,
):
    """The runtime holds a request made during a reply (one speaker); the SDK need not."""
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        rowan = nur.support_agent(name="Rowan", memory=False)
        async with rowan.connect("p1") as session:
            errors = []
            session.on("error", errors.append)
            await gateway.wait_type("response.create")
            await gateway.send({"type": "response.created", "response": {"id": "r1"}})
            await session.send_text("I need a refund.")
            await gateway.wait_for(
                lambda f: (
                    f.get("type") == "response.create"
                    and len(gateway.of_type("response.create")) == 2
                )
            )
            await gateway.send(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "response_request_superseded",
                        "message": "A newer continuation replaced this request.",
                    },
                }
            )
            await asyncio.sleep(0.1)
        assert errors == []


@run_async
async def test_ask_returns_the_answer_not_the_greeting(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(name="Luna", memory=False)
        async with luna.connect("p1") as session:
            await gateway.wait_type("response.create")

            async def model() -> None:
                item = await gateway.wait_type("conversation.item.create")
                assert item["item"]["id"].startswith("item_sdk_")
                # The greeting, begun before the line was typed, finishes after it.
                await gateway.send(
                    {
                        "type": "response.output_audio_transcript.done",
                        "response_id": "greet",
                        "transcript": "Hello there!",
                    }
                )
                await gateway.send(
                    {
                        "type": "response.done",
                        "response": {"id": "greet", "status": "completed"},
                    }
                )
                await gateway.send(
                    {"type": "response.created", "response": {"id": "tools_only"}}
                )
                await gateway.send(
                    {
                        "type": "response.done",
                        "response": {"id": "tools_only", "status": "completed"},
                    }
                )
                await gateway.reply("answer", "Miso is a lovely name.")

            # The greeting began before the line was added.
            await gateway.send(
                {"type": "response.created", "response": {"id": "greet"}}
            )
            await asyncio.sleep(0.05)
            task = asyncio.create_task(model())
            reply = await session.ask("I adopted a cat named Miso.")
            await task
        assert reply.response_id == "answer" and reply.text == "Miso is a lovely name."


@run_async
async def test_every_successful_reconnect_restores_the_reconnect_budget(
    api, monkeypatch
):
    import eesi.session as session_module

    monkeypatch.setattr(session_module, "MAX_RECONNECT_ATTEMPTS", 1)
    monkeypatch.setattr(session_module, "_reconnect_delay", lambda attempt: 0.01)
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        async with kael.connect("p1") as session:
            for _ in range(
                3
            ):  # more drops than one attempt each would allow without a reset
                waiter = asyncio.ensure_future(session.wait("reconnected", timeout=5))
                await gateway.drop(1011)
                await waiter
            assert not session.ended
        assert gateway.connections == 4
