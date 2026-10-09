"""SDK-level exceptions.

All errors raised from `eesi_sdk` are subclasses of `EESISdkError` so
calling code can catch them as one category.
"""

from __future__ import annotations

from typing import Any


class EESISdkError(Exception):
    """Base class for all SDK errors."""


class RequestTimeoutError(EESISdkError):
    """Raised when a request exceeds the client's connect or read timeout."""


class InvalidRequestError(EESISdkError):
    """Raised when a request body cannot be built from what was passed.

    Nothing has been sent when this is raised: the body did not satisfy the
    model the method's type hint names. `errors` is pydantic's own list of
    `{loc, msg, type}` problems when there was a body to validate, and empty
    when the argument was not a body at all.
    """

    def __init__(self, message: str, *, errors: list[Any] | None = None):
        super().__init__(message)
        self.errors = errors or []


def _describe_detail_item(item: Any) -> str | None:
    """One item of a FastAPI/backend validation array, or None when it says
    nothing."""
    if isinstance(item, str):
        return item or None
    if not isinstance(item, dict):
        return None
    message = item.get("message")
    if not isinstance(message, str) or not message:
        message = item.get("msg")
    if not isinstance(message, str) or not message:
        return None
    model = item.get("model")
    return f"{model}: {message}" if isinstance(model, str) and model else message


class ApiError(EESISdkError):
    """Raised when the EESI backend returns a non-2xx response.

    The backend answers in two envelopes and both are read here (the same
    reading the console uses, `packages/api-client/src/apiError.ts`):

    - FastAPI's `{"detail": ...}` from the control-plane routes — a string,
      or on 422 an array of `{loc, msg, type}` / `{model, message}` items,
      which is flattened one message per line.
    - OpenAI's `{"error": {"message", "type", "code", "param"}}` from the
      speech routes (`/v1/audio`, `/v1/chat`, `/v1/models`, `/v1/realtime`,
      `/v1/voices`), including auth failures on those paths.

    `message` is the server's own sentence; `code` and `type` are set only
    when the envelope carries them. `body` is the parsed JSON, or the raw
    text of a non-JSON body.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        body: object = None,
        *,
        code: str | None = None,
        type: str | None = None,
    ):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message
        self.body = body
        self.code = code
        self.type = type

    @classmethod
    def from_body(
        cls, status_code: int, body: object, fallback: str = "Request failed"
    ) -> ApiError:
        """Build the error from a failed response's body, whichever envelope
        it wears. `fallback` is used when the payload carries no sentence at
        all — never an empty message."""
        message: str | None = None
        code: str | None = None
        error_type: str | None = None
        if isinstance(body, str):
            message = body or None
        elif isinstance(body, dict):
            enveloped = body.get("error")
            if isinstance(enveloped, dict):
                if isinstance(enveloped.get("message"), str) and enveloped["message"]:
                    message = enveloped["message"]
                if isinstance(enveloped.get("code"), str):
                    code = enveloped["code"]
                if isinstance(enveloped.get("type"), str):
                    error_type = enveloped["type"]
            if message is None:
                detail = body.get("detail")
                if isinstance(detail, str):
                    message = detail or None
                elif isinstance(detail, list):
                    messages = [
                        m for m in map(_describe_detail_item, detail) if m is not None
                    ]
                    message = "\n".join(messages) or None
        return cls(status_code, message or fallback, body, code=code, type=error_type)
