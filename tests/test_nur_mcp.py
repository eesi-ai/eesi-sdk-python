"""Remote MCP servers on a character: what the SDK declares and how it reads
the gateway's MCP events. The gateway runs the calls; nothing here answers one.

Against the fake gateway, so these prove the wire, not a real server.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import FakeAPI, run_async
from eesi import ConfigurationError, McpServer, NurClient
from fake_gateway import FakeGateway

DEEPWIKI = McpServer(
    "https://mcp.deepwiki.com/mcp",
    label="deepwiki",
    description="Answers questions about public GitHub repositories.",
    allowed_tools=["ask_wiki_question"],
    authorization="sk-mcp-token",
    require_approval=["delete_page"],
)


def client(gateway: FakeGateway, api: FakeAPI) -> NurClient:
    return NurClient(
        api_key="sk-eesi-test", base_url=gateway.base_url, transport=api.transport
    )


def test_a_server_is_the_wires_mcp_entry_and_never_prints_its_token():
    assert DEEPWIKI.to_wire() == {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "server_description": "Answers questions about public GitHub repositories.",
        "allowed_tools": ["ask_wiki_question"],
        "authorization": "sk-mcp-token",
        "require_approval": {"always": {"tool_names": ["delete_page"]}},
    }
    assert "sk-mcp-token" not in repr(DEEPWIKI)
    with pytest.raises(ConfigurationError):
        McpServer("https://x.example.com/mcp", label="has space")
    with pytest.raises(ConfigurationError):
        McpServer("ftp://x.example.com/mcp", label="x")


@run_async
async def test_a_character_declares_its_servers_and_restates_them(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        sage = nur.character(name="Sage", greeting=False, mcp_servers=[DEEPWIKI])
        async with sage.connect("p1"):
            first = (await gateway.wait_type("session.update"))["session"]

            @sage.tool
            def note(text: str) -> str:
                """Write a note."""
                return "noted"

            again = await gateway.wait_for(
                lambda frame: (
                    frame.get("type") == "session.update"
                    and gateway.frames.index(frame)
                    > gateway.frames.index(gateway.of_type("session.update")[0])
                )
            )

        # With no function tools, the server alone still turns tools on.
        assert first["tools"] == [DEEPWIKI.to_wire()]
        assert first["tool_choice"] == "auto"
        # A tools update never drops the server.
        assert [
            tool.get("name") or tool["server_label"]
            for tool in again["session"]["tools"]
        ] == [
            "note",
            "deepwiki",
        ]


def test_a_saved_character_names_its_servers_by_label_only(api, tmp_path):
    nur = NurClient(api_key="sk-eesi-test", transport=api.transport)
    sage = nur.character(name="Sage", mcp_servers=[DEEPWIKI])
    saved = sage.to_dict()
    assert saved["mcp_servers"] == ["deepwiki"]
    assert "sk-mcp-token" not in json.dumps(saved, default=str)
    path = tmp_path / "sage.json"
    sage.save(path)
    loaded = nur.load_character(path, mcp_servers=[DEEPWIKI])
    assert loaded.mcp_servers == [DEEPWIKI]
    assert sage.clone(name="Sage II").mcp_servers == [DEEPWIKI]


@run_async
async def test_mcp_events_are_read_and_calls_are_left_to_the_gateway(api):
    async with FakeGateway() as gateway:
        nur = client(gateway, api)
        sage = nur.character(name="Sage", greeting=False, mcp_servers=[DEEPWIKI])
        events = []
        async with sage.connect("p1") as session:
            session.on("mcp.*", events.append)
            await gateway.wait_type("session.update")
            listing = {
                "id": "mcpl_1",
                "type": "mcp_list_tools",
                "server_label": "deepwiki",
            }
            await gateway.send(
                {"type": "conversation.item.added", "item": {**listing, "tools": []}}
            )
            await gateway.send(
                {"type": "mcp_list_tools.in_progress", "item_id": "mcpl_1"}
            )
            await gateway.send(
                {
                    "type": "conversation.item.done",
                    "item": {
                        **listing,
                        "tools": [{"name": "ask_wiki_question", "input_schema": {}}],
                    },
                }
            )
            await gateway.send(
                {"type": "mcp_list_tools.completed", "item_id": "mcpl_1"}
            )

            call = {
                "id": "item_1",
                "type": "mcp_call",
                "server_label": "deepwiki",
                "name": "ask_wiki_question",
                "arguments": '{"repoName": "a/b", "question": "What?"}',
            }
            await gateway.send({"type": "response.created", "response": {"id": "r1"}})
            await gateway.send(
                {
                    "type": "response.output_item.added",
                    "response_id": "r1",
                    "output_index": 0,
                    "item": call,
                }
            )
            await gateway.send(
                {
                    "type": "response.mcp_call_arguments.done",
                    "item_id": "item_1",
                    "arguments": call["arguments"],
                }
            )
            await gateway.send(
                {
                    "type": "response.mcp_call.in_progress",
                    "item_id": "item_1",
                    "output_index": 0,
                }
            )
            await gateway.send(
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": "completed", "output": [call]},
                }
            )
            await gateway.send(
                {
                    "type": "conversation.item.done",
                    "item": {**call, "output": "It is a wiki.", "error": None},
                }
            )
            await gateway.send(
                {
                    "type": "response.mcp_call.completed",
                    "item_id": "item_1",
                    "output_index": 0,
                }
            )

            failed = {**call, "id": "item_2"}
            await gateway.send(
                {
                    "type": "conversation.item.done",
                    "item": {
                        **failed,
                        "error": {
                            "type": "tool_execution_error",
                            "message": "too slow",
                        },
                    },
                }
            )
            await gateway.send(
                {
                    "type": "response.mcp_call.failed",
                    "item_id": "item_2",
                    "output_index": 0,
                }
            )

            request = {
                "id": "ask_1",
                "type": "mcp_approval_request",
                "server_label": "deepwiki",
                "name": "delete_page",
                "arguments": '{"page": "Home"}',
            }
            for kind in (
                "response.output_item.added",
                "conversation.item.added",
                "response.output_item.done",
            ):
                await gateway.send({"type": kind, "item": request})
            await gateway.send(
                {
                    "type": "mcp_list_tools.failed",
                    "server_label": "other",
                    "error": {
                        "code": "not_public",
                        "message": "The server is not on the public internet.",
                    },
                }
            )
            for _ in range(50):
                if len(events) >= 6:
                    break
                await asyncio.sleep(0.02)
            await session.approve_mcp("ask_1")
            approval = await gateway.wait_type("conversation.item.create")
            await asyncio.sleep(0.1)

        kinds = [event.type for event in events]
        assert kinds == [
            "mcp.tools.listed",
            "mcp.call.started",
            "mcp.call.completed",
            "mcp.call.failed",
            "mcp.approval.requested",
            "mcp.tools.failed",
        ]
        listed, started, completed, failed_event, asked, refused = events
        assert listed.server_label == "deepwiki" and listed.tools == (
            "ask_wiki_question",
        )
        assert (
            started.name == "ask_wiki_question"
            and started.arguments == call["arguments"]
        )
        assert completed.output == "It is a wiki."
        assert (
            failed_event.kind == "tool_execution_error"
            and failed_event.error == "too slow"
        )
        assert asked.request_id == "ask_1" and asked.name == "delete_page"
        assert refused.server_label == "other" and refused.code == "not_public"
        assert approval["item"] == {
            "type": "mcp_approval_response",
            "approval_request_id": "ask_1",
            "approve": True,
        }
        # The gateway ran the call: the SDK never answered it or asked to go on.
        outputs = [
            frame
            for frame in gateway.of_type("conversation.item.create")
            if frame["item"].get("type") == "function_call_output"
        ]
        assert outputs == []
        assert gateway.of_type("response.create") == []
