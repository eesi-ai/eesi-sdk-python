"""HTTP client for the EESI REST API.

Every endpoint method comes from `_GeneratedClient` (auto-generated from
the FastAPI OpenAPI spec — see `scripts/generate_sdk.sh`). This class only
adds the session/auth/transport surface around that mixin.

The SDK surface on the backend is controlled by decorating routes with
`@sdk_expose(method="...")`; anything else is invisible here.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from ._generated_client import _GeneratedClient
from .errors import ApiError, RequestTimeoutError

# Shared with the TypeScript SDK — change both or neither.
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 60.0


class EESIClient(_GeneratedClient):
    """Sync HTTP client. Suitable for scripts, pytest, and the LLM SDK
    exec sandbox.

    Auth precedence:
        1. `api_key` kwarg
        2. `EESI_API_KEY` env var
        3. unauthenticated (most endpoints will 401)

    The key (an `sk-eesi-...` API key created on the dashboard's
    /api-keys page) is sent as `Authorization: Bearer <key>`.

    Timeouts and retries (identical in the TypeScript SDK):
        - connect 10 s, read 60 s (write 60 s, pool 10 s). Pass `timeout` to
          override: a float applies to every phase, an `httpx.Timeout` is
          used as-is. A timeout raises `RequestTimeoutError`.
        - `GET` is retried once, after a 100–300 ms jittered pause, when the
          connection cannot be established (refused, DNS, connect timeout).
          Nothing else is retried: a `POST`/`PUT`/`PATCH`/`DELETE` that
          failed on the wire may still have reached the server, and a read
          timeout means the server took the request. Connection failures
          that survive the retry propagate as `httpx.ConnectError`.
        - Non-2xx responses raise `ApiError` (see `errors.py` for the two
          envelopes it reads).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        resolved_url = base_url or os.environ.get("EESI_API_URL", "https://api.eesi.ai")
        self.base_url = resolved_url.rstrip("/")
        self.api_key = api_key or os.environ.get("EESI_API_KEY")

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if timeout is None:
            timeout = httpx.Timeout(
                connect=DEFAULT_CONNECT_TIMEOUT,
                read=DEFAULT_READ_TIMEOUT,
                write=DEFAULT_READ_TIMEOUT,
                pool=DEFAULT_CONNECT_TIMEOUT,
            )
        # `transport` exists so tests can drive the client through
        # `httpx.MockTransport` without a server.
        self._http = httpx.Client(
            base_url=f"{self.base_url}/v1",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EESIClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── low-level ──────────────────────────────────────────────────

    def _request(
        self, method: str, path: str, *, raw: bool = False, **kwargs: Any
    ) -> Any:
        """Send one request. `raw=True` returns the response bytes untouched
        (audio); otherwise JSON is parsed and a non-JSON body comes back as
        text. See the class docstring for the timeout and retry policy."""
        retries_left = 1 if method == "GET" else 0
        while True:
            try:
                resp = self._http.request(method, path, **kwargs)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if retries_left:
                    retries_left -= 1
                    time.sleep(random.uniform(0.1, 0.3))
                    continue
                if isinstance(exc, httpx.ConnectTimeout):
                    raise RequestTimeoutError(
                        f"{method} {path} timed out (connect)"
                    ) from exc
                raise
            except httpx.TimeoutException as exc:
                phase = type(exc).__name__.removesuffix("Timeout").lower()
                raise RequestTimeoutError(
                    f"{method} {path} timed out ({phase})"
                ) from exc

        if resp.status_code >= 400:
            try:
                body: object = resp.json()
            except ValueError:
                body = resp.text
            raise ApiError.from_body(
                resp.status_code, body, fallback=resp.reason_phrase or "Request failed"
            )
        if raw:
            return resp.content
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text
