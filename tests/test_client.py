"""HTTP-layer tests for `eesi_sdk.EESIClient`, driven through
`httpx.MockTransport` — no server.

The endpoint methods themselves are generated, so what is worth testing is
the wire: error envelopes, the multipart call shape the generator emits,
binary responses, and the timeout/retry policy the TypeScript SDK mirrors.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx
import pytest
from eesi_sdk import (
    ApiError,
    EESIClient,
    InvalidRequestError,
    RequestTimeoutError,
)
from eesi_sdk.models import SpeechRequest


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> EESIClient:
    return EESIClient(
        base_url="http://api.test",
        api_key="sk-eesi-test",
        transport=httpx.MockTransport(handler),
    )


def respond_json(
    status: int, body: object
) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: httpx.Response(status, json=body)


# ── error envelopes ──────────────────────────────────────────────────────


def test_openai_envelope_is_read_with_code_and_type():
    client = make_client(
        respond_json(
            400,
            {
                "error": {
                    "message": "Unknown model: nur-tts-v9",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": None,
                }
            },
        )
    )
    with pytest.raises(ApiError) as exc:
        client.list_voices()
    err = exc.value
    assert err.status_code == 400
    assert err.message == "Unknown model: nur-tts-v9"
    assert err.type == "invalid_request_error"
    assert err.code is None
    assert str(err) == "[400] Unknown model: nur-tts-v9"


def test_auth_failure_wearing_the_openai_envelope():
    client = make_client(
        respond_json(
            401,
            {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
    )
    with pytest.raises(ApiError) as exc:
        client.list_voices()
    assert exc.value.message == "Invalid API key"
    assert exc.value.code == "invalid_api_key"


def test_fastapi_string_detail():
    client = make_client(respond_json(404, {"detail": "Voice not found"}))
    with pytest.raises(ApiError) as exc:
        client.get_voice("voice_1")
    assert exc.value.message == "Voice not found"
    assert exc.value.code is None
    assert exc.value.type is None
    assert exc.value.body == {"detail": "Voice not found"}


def test_422_array_is_flattened_one_message_per_line():
    client = make_client(
        respond_json(
            422,
            {
                "detail": [
                    {
                        "loc": ["body", "name"],
                        "msg": "Field required",
                        "type": "missing",
                    },
                    {"model": "nur-tts-v1", "message": "Backend unavailable"},
                    "plain string item",
                ]
            },
        )
    )
    with pytest.raises(ApiError) as exc:
        client.get_voice("voice_1")
    assert exc.value.status_code == 422
    assert exc.value.message == (
        "Field required\nnur-tts-v1: Backend unavailable\nplain string item"
    )


def test_envelope_wins_when_a_body_carries_both_shapes():
    err = ApiError.from_body(
        400, {"error": {"message": "from envelope"}, "detail": "from detail"}
    )
    assert err.message == "from envelope"


def test_non_json_body_is_its_own_message():
    client = make_client(lambda _request: httpx.Response(502, text="Bad Gateway"))
    with pytest.raises(ApiError) as exc:
        client.get_voice("voice_1")
    assert exc.value.message == "Bad Gateway"
    assert exc.value.body == "Bad Gateway"


def test_falls_back_when_the_payload_says_nothing():
    for body in ({}, {"error": {}}, {"error": {"message": ""}}, {"detail": []}, ""):
        assert ApiError.from_body(500, body, "x").message == "x"
    assert ApiError.from_body(500, {"detail": [{"loc": ["body"]}]}, "x").message == "x"


# ── multipart call signature ─────────────────────────────────────────────


def test_create_transcription_sends_file_and_fields_as_multipart():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "hello"})

    client = make_client(handler)
    result = client.create_transcription(
        file=("clip.wav", b"RIFF....", "audio/wav"),
        model="nur-stt-v1",
        timestamp_granularities=["word", "segment"],
        temperature=0.2,
    )
    assert result == {"text": "hello"}

    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer sk-eesi-test"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    body = request.read().decode("latin-1")
    assert 'name="file"; filename="clip.wav"' in body
    assert "Content-Type: audio/wav" in body
    assert 'name="model"\r\n\r\nnur-stt-v1' in body
    assert 'name="temperature"\r\n\r\n0.2' in body
    # The list field is repeated under its wire name, `[]` suffix included.
    assert body.count('name="timestamp_granularities[]"') == 2
    # Omitted optionals are not sent, so the server applies its own defaults.
    assert 'name="language"' not in body
    assert 'name="stream"' not in body


def test_create_voice_takes_raw_bytes_and_booleans():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "voice_id": "voice_1",
                "name": "Sarah",
                "engine": "nur-tts-v1",
                "status": "ready",
                "has_ref_text": False,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    client = make_client(handler)
    voice = client.create_voice(file=b"RIFF....", name="Sarah", consent_attested=True)
    assert voice.name == "Sarah"

    body = seen[0].read().decode("latin-1")
    assert 'name="file"; filename="upload"' in body
    assert 'name="name"\r\n\r\nSarah' in body
    assert 'name="consent_attested"\r\n\r\ntrue' in body


def test_chat_completions_forwards_the_json_body_verbatim():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": []})

    client = make_client(handler)
    payload = {"model": "nur-llm-v1", "messages": [{"role": "user", "content": "hi"}]}
    assert client.chat_completions(body=payload) == {"choices": []}
    assert seen[0].headers["content-type"] == "application/json"
    assert seen[0].read() == httpx.Request("POST", "http://x", json=payload).read()


def test_create_speech_returns_the_audio_bytes():
    audio = b"\xff\xfb\x90\x00" * 8
    client = make_client(
        lambda _request: httpx.Response(
            200, content=audio, headers={"content-type": "audio/mpeg"}
        )
    )
    out = client.create_speech(
        body=SpeechRequest(model="nur-tts-v1", input="Hello", voice="ev_1a2b3c4d")
    )
    assert out == audio


# ── request bodies ───────────────────────────────────────────────────────


def test_a_mapping_body_is_validated_and_sent():
    """The README's first call passes a dict. `body.model_dump()` used to
    meet it with `AttributeError: 'dict' object has no attribute
    'model_dump'` — not an SDK error, and not a sentence about the request."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"audio")

    client = make_client(handler)
    out = client.create_speech(
        body={
            "model": "nur-tts-v1",
            "voice": "ev_1a2b3c4d",
            "input": "Hello",
            "response_format": "wav",
        }
    )

    assert out == b"audio"
    sent = json.loads(seen[0].read())
    assert sent["voice"] == "ev_1a2b3c4d"
    assert sent["response_format"] == "wav"
    # Server defaults the caller did not set still travel, exactly as they do
    # when the model is built by hand.
    assert sent["speed"] == 1.0


def test_a_body_missing_a_field_names_the_field_before_the_wire():
    client = make_client(_never_called)

    with pytest.raises(InvalidRequestError) as exc:
        client.create_speech(body={"model": "nur-tts-v1", "input": "Hello"})

    message = str(exc.value)
    assert "SpeechRequest" in message
    assert "voice: Field required" in message
    assert [error["loc"] for error in exc.value.errors] == [("voice",)]


def test_a_body_that_is_not_a_body_at_all_says_what_was_passed():
    client = make_client(_never_called)

    with pytest.raises(InvalidRequestError) as exc:
        client.create_speech(body="nur-tts-v1")  # type: ignore[arg-type]

    assert "must be a SpeechRequest or a mapping of its fields, not str" in str(
        exc.value
    )


def test_the_readme_quickstart_reaches_a_usable_voice_id():
    """`list_voices().data[*].voice_id` is what `voice` takes — the README
    walks a reader from one to the other, so both halves are pinned here."""
    client = make_client(
        respond_json(
            200,
            {
                "data": [
                    {
                        "voice_id": "ev_1a2b3c4d",
                        "name": "Nur",
                        "engine": "omnivoice",
                        "status": "ready",
                        "has_ref_text": True,
                        "created_at": "2026-01-01T00:00:00Z",
                        "is_builtin": True,
                    }
                ]
            },
        )
    )

    voices = client.list_voices()

    assert [(v.voice_id, v.name) for v in voices.data] == [("ev_1a2b3c4d", "Nur")]


def _never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a rejected body still reached the wire: {request.url}")


# ── timeouts and retries ─────────────────────────────────────────────────


def test_default_base_url_is_production(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EESI_API_URL", raising=False)
    client = EESIClient(transport=httpx.MockTransport(respond_json(200, {})))
    try:
        assert client.base_url == "https://api.eesi.ai"
    finally:
        client.close()


def test_default_timeouts_are_connect_10s_read_60s():
    client = make_client(respond_json(200, {}))
    assert client._http.timeout == httpx.Timeout(
        connect=10.0, read=60.0, write=60.0, pool=10.0
    )


def test_float_timeout_applies_to_every_phase():
    client = EESIClient(base_url="http://api.test", api_key="k", timeout=5.0)
    assert client._http.timeout == httpx.Timeout(5.0)


def test_read_timeout_raises_request_timeout_error_without_retry():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    client = make_client(handler)
    with pytest.raises(
        RequestTimeoutError, match=r"GET /user/api-keys timed out \(read\)"
    ):
        client.get_api_keys()
    assert calls == 1


def test_get_is_retried_once_on_connect_errors(monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=[])

    client = make_client(handler)
    assert client.get_api_keys() == []
    assert calls == 2
    assert len(sleeps) == 1 and 0.1 <= sleeps[0] <= 0.3


def test_get_connect_failure_propagates_after_the_single_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("slow", request=request)

    client = make_client(handler)
    with pytest.raises(RequestTimeoutError, match=r"timed out \(connect\)"):
        client.get_api_keys()
    assert calls == 2


def test_post_is_never_retried(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with pytest.raises(httpx.ConnectError):
        client.chat_completions(body={"model": "nur-llm-v1", "messages": []})
    assert calls == 1
