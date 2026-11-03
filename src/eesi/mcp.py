"""Remote MCP servers a character can use. EESI connects to them and runs the calls.

::

    deepwiki = McpServer(
        "https://mcp.deepwiki.com/mcp",
        label="deepwiki",
        description="Answers questions about public GitHub repositories.",
        allowed_tools=["ask_wiki_question", "read_wiki_structure"],
    )
    sage = nur.character(name="Sage", mcp_servers=[deepwiki])

When a session starts, EESI lists the server's tools and offers them to the
character as ``<label>__<tool>``. Every call runs on EESI's side: your code
never runs or answers them, and watches them as events instead
(``mcp.tools.listed``, ``mcp.call.started``, ``mcp.call.completed`` …). A tool
the server marks destructive, or one named in ``require_approval``, waits for
the player's yes: the character asks, and the player answers out loud, or your
UI calls ``session.approve_mcp(request_id)`` on ``mcp.approval.requested``.

``authorization`` and ``headers`` go to the server and nowhere else; they are
never printed, and :meth:`Character.to_dict` names a server only by its label.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from .errors import ConfigurationError

__all__ = ["McpServer", "as_mcp_servers"]

_LABEL = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


@dataclass(frozen=True)
class McpServer:
    """One remote MCP server, as a session's ``{"type": "mcp"}`` tool entry."""

    url: str
    label: str
    #: What the server is for; the character reads it beside each tool.
    description: str | None = None
    #: The tool names to offer; None (or empty) offers all of them.
    allowed_tools: Sequence[str] | None = None
    #: Sent as ``Authorization: Bearer <token>``.
    authorization: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] | None = field(default=None, repr=False)
    #: ``"always"``, or tool names, to make the character ask the player first.
    #: A tool the server marks destructive always waits for a yes.
    require_approval: Literal["always", "never"] | Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not _LABEL.match(self.label):
            raise ConfigurationError(
                "An MCP server's label is 1 to 32 letters, digits, '_' or '-'."
            )
        parts = urlsplit(self.url)
        if parts.scheme not in ("https", "http") or not parts.hostname:
            raise ConfigurationError(f"{self.label}: the URL must be https://…")

    def to_wire(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "type": "mcp",
            "server_label": self.label,
            "server_url": self.url,
        }
        if self.description:
            entry["server_description"] = self.description
        if self.allowed_tools is not None:
            entry["allowed_tools"] = list(self.allowed_tools)
        if self.authorization:
            entry["authorization"] = self.authorization
        if self.headers:
            entry["headers"] = dict(self.headers)
        if isinstance(self.require_approval, str):
            entry["require_approval"] = self.require_approval
        elif self.require_approval is not None:
            entry["require_approval"] = {
                "always": {"tool_names": list(self.require_approval)}
            }
        return entry


def as_mcp_servers(
    servers: Iterable[McpServer | Mapping[str, Any]],
) -> list[McpServer]:
    """Servers as given: :class:`McpServer`, or the wire's ``{"type": "mcp"}`` dict."""
    made: list[McpServer] = []
    for server in servers:
        if isinstance(server, McpServer):
            made.append(server)
            continue
        if not isinstance(server, Mapping) or server.get("type", "mcp") != "mcp":
            raise ConfigurationError(
                "An MCP server is an McpServer or an mcp tool entry."
            )
        approval = server.get("require_approval")
        if isinstance(approval, Mapping):
            always = approval.get("always")
            approval = (
                list(always.get("tool_names") or [])
                if isinstance(always, Mapping)
                else None
            )
        made.append(
            McpServer(
                url=str(server.get("server_url") or server.get("url") or ""),
                label=str(server.get("server_label") or server.get("label") or ""),
                description=server.get("server_description"),
                allowed_tools=server.get("allowed_tools"),
                authorization=server.get("authorization"),
                headers=server.get("headers"),
                require_approval=approval,
            )
        )
    labels = [server.label for server in made]
    if len(labels) != len(set(labels)):
        raise ConfigurationError("Two MCP servers share a label.")
    return made


_ITEM_EVENTS = frozenset(
    {
        "response.output_item.added",
        "response.output_item.done",
        "conversation.item.added",
        "conversation.item.done",
    }
)


class McpEvents:
    """Reads the MCP part of a session's server events into typed events.

    The wire is OpenAI's: ``mcp_list_tools`` / ``mcp_call`` /
    ``mcp_approval_request`` items, and ``mcp_list_tools.*`` and
    ``response.mcp_call.*`` events that name an item by its id.
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._asked: set[str] = set()

    def read(self, kind: str, event: Mapping[str, Any]) -> list[Any]:
        from .events import (
            McpApprovalRequested,
            McpCallCompleted,
            McpCallFailed,
            McpCallStarted,
            McpToolsFailed,
            McpToolsListed,
        )

        if kind in _ITEM_EVENTS:
            item = event.get("item")
            if not isinstance(item, dict) or not str(item.get("type") or "").startswith(
                "mcp_"
            ):
                return []
            identity = str(item.get("id") or "")
            if not identity:
                return []
            self._items[identity] = dict(item)
            while len(self._items) > 256:
                del self._items[next(iter(self._items))]
            if (
                item.get("type") == "mcp_approval_request"
                and identity not in self._asked
            ):
                self._asked.add(identity)
                return [
                    McpApprovalRequested(
                        request_id=identity,
                        server_label=str(item.get("server_label") or ""),
                        name=str(item.get("name") or ""),
                        arguments=str(item.get("arguments") or ""),
                    )
                ]
            return []
        if not kind.startswith(("mcp_list_tools.", "response.mcp_call.")):
            return []
        item = self._items.get(str(event.get("item_id") or ""), {})
        label = str(event.get("server_label") or item.get("server_label") or "")
        if kind == "mcp_list_tools.completed":
            tools = item.get("tools") if isinstance(item.get("tools"), list) else []
            return [
                McpToolsListed(
                    server_label=label,
                    tools=tuple(
                        str(tool.get("name"))
                        for tool in tools
                        if isinstance(tool, dict) and tool.get("name")
                    ),
                )
            ]
        if kind == "mcp_list_tools.failed":
            error = event.get("error") if isinstance(event.get("error"), dict) else {}
            return [
                McpToolsFailed(
                    server_label=label,
                    code=str(error.get("code") or ""),
                    message=str(error.get("message") or ""),
                )
            ]
        call = {
            "item_id": str(event.get("item_id") or ""),
            "server_label": label,
            "name": str(item.get("name") or ""),
            "arguments": str(item.get("arguments") or ""),
        }
        if kind == "response.mcp_call.in_progress":
            return [McpCallStarted(**call)]
        if kind == "response.mcp_call.completed":
            return [McpCallCompleted(**call, output=str(item.get("output") or ""))]
        if kind == "response.mcp_call.failed":
            error = item.get("error") if isinstance(item.get("error"), dict) else {}
            return [
                McpCallFailed(
                    **call,
                    kind=str(error.get("type") or ""),
                    error=str(error.get("message") or ""),
                )
            ]
        return []
