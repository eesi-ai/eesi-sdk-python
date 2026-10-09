"""Control channels, speaking first, languages and rooms, against the fake gateway."""

from __future__ import annotations

import array
import asyncio
import math

from conftest import FakeAPI, run_async
from eesi import ClientSecret, NurClient, Player, Proactive, ToolContext
from fake_gateway import FakeGateway


def client(gateway: FakeGateway, api: FakeAPI) -> NurClient:
    return NurClient(
        api_key="sk-eesi-test", base_url=gateway.base_url, transport=api.transport
    )


def secret(control_tools: str = "server") -> ClientSecret:
    return ClientSecret(
        value="rt_x",
        expires_at=0,
        url="wss://x",
        session={},
        control_id="ctl_abcdefghijklmnopqrstuvwx",
        control_tools=control_tools,
        player_id="p42",
        player_name="Sam",
    )


# ── a server steering a session its player's client opened ──────────────


@run_async
async def test_a_controlled_session_steers_without_audio(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(
            name="Kael", memory=nur.memory.in_memory(), context={"realm": "North"}
        )
        ran = []

        @kael.tool
        def open_gate(ctx: ToolContext) -> str:
            """Open the gate."""
            ran.append(ctx.player_id)
            return "The gate is open."

        session = kael.control(secret())
        started = []
        session.on("session.started", started.append)
        async with session:
            assert (
                gateway.paths[0] == "/v1/realtime/control/ctl_abcdefghijklmnopqrstuvwx"
            )
            assert gateway.headers[0]["Authorization"] == "Bearer sk-eesi-test"
            greeting = await gateway.wait_type("response.create")
            await session.context.update({"location": "Northern Gate"})
            await gateway.wait_for(
                lambda f: (
                    f.get("type") == "eesi.context.update" and f.get("key") == "scene"
                )
            )
            # The client's session events arrive here, and its tools run here.
            await gateway.send({"type": "response.created", "response": {"id": "r1"}})
            await gateway.send(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": "r1",
                    "call_id": "c1",
                    "name": "open_gate",
                    "arguments": "{}",
                    "eesi_executor": "server",
                }
            )
            output = await gateway.wait_type("conversation.item.create")
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": "completed"},
                }
            )
            await gateway.wait_for(
                lambda f: f.get("type") == "response.create" and f is not greeting
            )
            try:
                await session.send_audio(b"\x00" * 640)
                raised = False
            except Exception:
                raised = True
            await gateway.end_control()
            await asyncio.wait_for(session.run(), 5)

        assert started and started[0].session_id == gateway.session_id
        assert not any(
            f.get("type")
            in (
                "session.update",
                "input_audio_buffer.append",
                "eesi.output.status",
                "eesi.keepalive",
            )
            for f in gateway.frames
        )
        assert output["item"]["output"] == "The gate is open." and ran == ["p42"]
        assert raised, "a controlled session has no microphone"
        assert "greet" in greeting["response"]["instructions"]


@run_async
async def test_tools_left_to_the_client_are_not_run_by_the_server(api):
    async with FakeGateway() as gateway:
        gateway.control_tools = "client"
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)

        @kael.tool
        def open_gate() -> str:
            """Open the gate."""
            return "open"

        async with kael.control(secret("client")):
            await gateway.send(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "c1",
                    "name": "open_gate",
                    "arguments": "{}",
                }
            )
            await asyncio.sleep(0.2)
        assert gateway.of_type("conversation.item.create") == []


@run_async
async def test_a_secret_minted_with_control_carries_the_channel(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael")
        minted = await kael.client_secret(
            "p42", player_name="Sam", control=True, lock="all"
        )
        body = api.secrets[-1]
        assert body["eesi_control"] == {"tools": "server"}
        assert minted.player_id == "p42" and minted.player_name == "Sam"
        assert minted.control_id == "ctl_0123456789abcdef"
        # What the player's client gets: how to behave, never the key or the channel.
        join = minted.join
        assert join["url"] == minted.url and join["controlled"] is True
        assert join["tools"] == "server" and join["greeting"] is None
        assert join["playerId"] == "p42" and join["outputSampleRate"] == 24_000
        assert "ctl_" not in str(join) and "sk-" not in str(join)


@run_async
async def test_a_secret_without_control_lets_the_client_greet(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(name="Luna", memory=False)
        minted = await luna.client_secret("p7", player_name="Ada", language="es")
        body = api.secrets[-1]
        assert body["session"]["eesi_language_name"] == "Spanish"
        join = minted.join
        assert join["controlled"] is False and join["tools"] == "client"
        assert join["greeting"].startswith("Open the conversation")
        assert join["proactive"]["idleAfterMs"] == 30_000


# ── speaking first ───────────────────────────────────────────────────────


@run_async
async def test_a_trigger_fires_once_as_the_state_turns_true(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        kael.when(
            lambda state: state.get("health", 100) < 20,
            "The player is badly hurt.",
            kind="hurt",
            urgent=True,
        )
        async with kael.connect("p1") as session:
            await session.context.update({"health": 80})
            await asyncio.sleep(0.1)
            await session.context.update({"health": 12})
            create = await gateway.wait_type("response.create")
            await gateway.send({"type": "response.created", "response": {"id": "r1"}})
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": "completed"},
                }
            )
            await session.context.update({"health": 10})  # still true: no second remark
            await asyncio.sleep(0.3)
        assert "badly hurt" in create["response"]["input"][0]["content"][0]["text"]
        assert len(gateway.of_type("response.create")) == 1


@run_async
async def test_a_quiet_moment_is_considered_and_spoken_only_if_worth_it(
    api, monkeypatch
):
    import eesi.session as session_module

    monkeypatch.setattr(session_module, "IDLE_POLL_S", 0.05)
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(
            name="Luna",
            greeting=False,
            memory=False,
            proactive=Proactive(idle_after=0.2, cooldown=0.0, max_remarks=1),
        )
        texts = []
        async with luna.connect("p1") as session:
            session.on("response.text.delta", texts.append)
            session.on("response.completed", texts.append)
            check = await gateway.wait_type("response.create", timeout=3)
            # The decision: out of band, text only, never shown as a reply.
            await gateway.send(
                {
                    "type": "response.created",
                    "response": {
                        "id": "oob1",
                        "conversation_id": None,
                        "metadata": {"eesi_sdk": "idle_check"},
                    },
                }
            )
            await gateway.send(
                {
                    "type": "response.output_text.delta",
                    "response_id": "oob1",
                    "delta": "How did the interview go?",
                }
            )
            await gateway.send(
                {
                    "type": "response.output_text.done",
                    "response_id": "oob1",
                    "text": "How did the interview go?",
                }
            )
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "oob1", "status": "completed"},
                }
            )
            spoken = await gateway.wait_for(
                lambda f: f.get("type") == "response.create" and f is not check
            )
            await asyncio.sleep(0.4)
        assert check["response"]["conversation"] == "none"
        assert check["response"]["output_modalities"] == ["text"]
        assert "How did the interview go?" in spoken["response"]["instructions"]
        assert texts == [], "the decision is the SDK's own, not a reply"
        assert len(gateway.of_type("response.create")) == 2, (
            "max_remarks caps the check-ins"
        )


@run_async
async def test_a_quiet_moment_not_worth_a_remark_stays_quiet(api, monkeypatch):
    import eesi.session as session_module

    monkeypatch.setattr(session_module, "IDLE_POLL_S", 0.05)
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        luna = nur.companion(
            name="Luna",
            greeting=False,
            memory=False,
            proactive=Proactive(idle_after=0.2, cooldown=10.0),
        )
        async with luna.connect("p1"):
            await gateway.wait_type("response.create", timeout=3)
            await gateway.send(
                {
                    "type": "response.created",
                    "response": {
                        "id": "oob1",
                        "conversation_id": None,
                        "metadata": {"eesi_sdk": "idle_check"},
                    },
                }
            )
            await gateway.send(
                {
                    "type": "response.output_text.done",
                    "response_id": "oob1",
                    "text": "SKIP",
                }
            )
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "oob1", "status": "completed"},
                }
            )
            await asyncio.sleep(0.5)
        assert len(gateway.of_type("response.create")) == 1


# ── languages ────────────────────────────────────────────────────────────


@run_async
async def test_a_session_pinned_to_a_language_greets_and_answers_in_it(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False)
        async with kael.connect("p1", language="ar-EG"):
            update = gateway.of_type("session.update")[0]["session"]
        assert update["audio"]["input"]["transcription"]["language"] == "ar"
        assert update["eesi_language_name"] == "Egyptian Arabic"
        assert "Speak Egyptian Arabic." in update["instructions"]


@run_async
async def test_a_character_that_follows_the_player_is_not_pinned(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False, language=["en", "ar"])
        async with kael.connect("p1"):
            update = gateway.of_type("session.update")[0]["session"]
        assert "language" not in update["audio"]["input"]["transcription"]
        assert "eesi_language_name" not in update
        assert "You speak English and Arabic" in update["instructions"]


# ── rooms ────────────────────────────────────────────────────────────────


def tone(level: float, ms: int = 20) -> bytes:
    samples = int(16 * ms)
    return array.array(
        "h", (int(level * 32767 * math.sin(i / 3.0)) for i in range(samples))
    ).tobytes()


@run_async
async def test_a_room_mixes_players_and_knows_who_spoke(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        kael = nur.npc(name="Kael", greeting=False, memory=False)
        room = kael.room(
            [Player("Ana", "p1", language="ar"), Player("Ben", "p2", language="en")],
            greet_joiners=True,
        )
        heard = []
        room.on("transcript.final", heard.append)
        async with room:
            # Ana talks for half a second while Ben is quiet.
            for _ in range(25):
                room.feed("p1", tone(0.3))
                room.feed("p2", tone(0.001))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
            await gateway.send(
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "i1",
                    "audio_start_ms": 0,
                }
            )
            await gateway.send(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": "i1",
                    "audio_end_ms": 500,
                }
            )
            await gateway.send(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "i1",
                    "transcript": "Marhaba Kael!",
                }
            )
            await room.session.wait("transcript.final", timeout=3)
            roster = await gateway.wait_for(
                lambda f: (
                    f.get("type") == "eesi.context.update" and f.get("key") == "roster"
                )
            )
            await room.add_player(Player("Cy", "p3"))
            cue = await gateway.wait_type("response.create", timeout=3)
            appends = gateway.of_type("input_audio_buffer.append")
        assert heard[0].turns[0].player_id == "p1"
        assert "Ana (speaks Arabic), Ben (speaks English)" in roster["text"]
        assert (
            "Ana spoke last." in roster["text"]
            and "Answer each person in their own language." in roster["text"]
        )
        assert (
            "Cy just joined the group."
            in cue["response"]["input"][0]["content"][0]["text"]
        )
        assert len(appends) >= 20, "the mixer keeps sending 20 ms frames"
