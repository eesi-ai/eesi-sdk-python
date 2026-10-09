"""Shared helpers for the SDK tests: async tests without a plugin, and a fake HTTP API."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

VOICES = [
    {
        "voice_id": "ev_nur00001",
        "name": "Nur",
        "is_builtin": True,
        "status": "ready",
        "language": "ar",
    },
    {
        "voice_id": "ev_atlas001",
        "name": "Atlas",
        "is_builtin": True,
        "status": "ready",
        "language": "en",
    },
]


def run_async(fn: Callable[..., Any] | None = None, *, timeout: float = 30.0) -> Any:
    """Run an async test on its own event loop (the SDK tests need no pytest plugin)."""

    def wrap(test: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(test)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return asyncio.run(asyncio.wait_for(test(*args, **kwargs), timeout))

        wrapper.__signature__ = inspect.signature(test)  # type: ignore[attr-defined]
        return wrapper

    return wrap(fn) if fn is not None else wrap


class FakeAPI:
    """The REST endpoints the Nur SDK calls, answered in memory."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.chat_replies: list[str] = []
        self.secrets: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/voices" and request.method == "GET":
            return httpx.Response(200, json={"data": VOICES})
        if path == "/v1/chat/completions":
            reply = (
                self.chat_replies.pop(0)
                if self.chat_replies
                else '{"summary": "", "memories": [], "forget": []}'
            )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": reply}}]
                },
            )
        if path == "/v1/realtime/client_secrets":
            body = json.loads(request.content)
            self.secrets.append(body)
            lock = body.get("eesi_lock") or ["instructions", "tools"]
            return httpx.Response(
                200,
                json={
                    "value": "rt_secret",
                    "expires_at": 1_900_000_000,
                    "session": body["session"],
                    "eesi_lock": lock if isinstance(lock, list) else ["all"],
                    "eesi_control": (
                        {
                            "id": "ctl_0123456789abcdef",
                            "tools": body["eesi_control"]["tools"],
                        }
                        if body.get("eesi_control")
                        else None
                    ),
                },
            )
        if path == "/v1/realtime/ticket":
            return httpx.Response(200, json={"ticket": "rt_plain", "expires_in": 30})
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()
