"""Errors raised by the Nur SDK.

Every error is an :class:`EESIError`. The subclasses say what to fix: a
missing key, a refused credential, a character the gateway cannot serve, a
context block over its limit. REST failures keep the REST SDK's
:class:`eesi_sdk.ApiError`, which carries the server's status and message.
"""

from __future__ import annotations

from eesi_sdk.errors import ApiError

__all__ = [
    "ApiError",
    "AudioDeviceError",
    "AuthenticationError",
    "ConfigurationError",
    "ContextError",
    "EESIError",
    "MemoryStoreError",
    "RealtimeConnectionError",
    "SessionClosedError",
    "ToolDefinitionError",
]


class EESIError(Exception):
    """Base class for everything the Nur SDK raises on purpose."""


class ConfigurationError(EESIError, ValueError):
    """An option is missing or cannot work: no API key, an unknown voice."""


class AuthenticationError(EESIError):
    """The gateway refused the credential (HTTP 401/403 or close 1008 at dial)."""


class RealtimeConnectionError(EESIError):
    """The realtime session could not be reached or was lost for good.

    ``code`` is the WebSocket close code when there was one. ``kind`` says
    whether retrying can help: ``"at-limit"`` (capacity or quota; try again
    later), ``"unavailable"`` (the backend is down), or ``"dropped"``.
    """

    def __init__(self, message: str, *, code: int | None = None, kind: str = "dropped"):
        super().__init__(message)
        self.code = code
        self.kind = kind


class SessionClosedError(EESIError):
    """An operation needed a live session, and this one has ended."""


class ContextError(EESIError, ValueError):
    """A live-context update the gateway would refuse: a bad key or a block over its limit."""


class ToolDefinitionError(EESIError, TypeError):
    """A function cannot be offered as a tool: an unsupported parameter, a duplicate name."""


class MemoryStoreError(EESIError):
    """A memory store could not read or write."""


class AudioDeviceError(EESIError):
    """The local microphone or speaker could not be opened."""
