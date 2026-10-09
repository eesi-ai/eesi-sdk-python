"""GENERATED — do not edit. Source: filtered OpenAPI from `api.app`.

Regenerate with `./scripts/generate_sdk.sh`.

`EESIClient` mixes in this class to get HTTP methods for every route
decorated with `sdk_expose(...)` on the backend. Request/response types
come from `_generated_models` (datamodel-codegen output).

A `body` argument takes its model or a plain mapping of the same fields;
`json_body` validates the mapping and raises `InvalidRequestError` naming
the offending field rather than letting a bad body reach the wire.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import IO, Any, Literal, Union

from eesi_sdk._body import json_body
from eesi_sdk._generated_models import (
    APIKeyResponse,
    BillingCreditsResponse,
    ClientSecretRequest,
    ClientSecretResponse,
    CurrentUsageResponse,
    DailyUsageBreakdownResponse,
    GenerationListResponse,
    InteractionFailure,
    InteractionFailureList,
    InteractionFailureUpdate,
    ListenerBrief,
    ListenerEvents,
    ListenerState,
    OrganizationTurnMetricsSummaryResponse,
    PhoneCall,
    PhoneCallRequest,
    PhoneNumber,
    PhoneNumberConfirmation,
    PhoneNumberList,
    PhoneNumberRequest,
    ScorecardResponse,
    SeparatedClip,
    SpeechModelsResponse,
    SpeechRequest,
    TextRequest,
    UsageBreakdownResponse,
    UsageHistoryResponse,
    VoiceIsolation,
    VoiceIsolationRequest,
    VoiceResponse,
    VoicesListResponse,
    VoiceUpdateRequest,
)

# An upload for a multipart field, in the forms httpx accepts: raw bytes or a
# binary stream (sent as `upload`, `application/octet-stream`), or a
# `(filename, content)` / `(filename, content, content_type)` tuple. Pass the
# tuple form when the server keys off the extension or media type — it does
# for voice references.
FileInput = Union[
    bytes,
    IO[bytes],
    tuple[str, Union[bytes, IO[bytes]]],
    tuple[str, Union[bytes, IO[bytes]], str],
]


class _GeneratedClient:
    # `EESIClient.__init__` installs `self._request` (see client.py).

    def archive_api_key(self, api_key_id: int) -> Any:
        """Archive an API key (soft delete)."""
        return self._request("DELETE", f"/user/api-keys/{api_key_id}")

    def chat_completions(self, *, body: Mapping[str, Any]) -> Any:
        """Chat completions. `body` is an OpenAI chat-completions request and is forwarded verbatim; the parsed completion is returned. Streaming (`stream: true`) is not supported by this method — use an OpenAI client for that."""
        return self._request("POST", "/chat/completions", json=dict(body))

    def confirm_phone_number(
        self, *, body: PhoneNumberConfirmation | Mapping[str, Any]
    ) -> PhoneNumber:
        """Enter the code Nur phoned in; the number is confirmed for its calls."""
        data = self._request(
            "POST",
            "/phone/numbers/confirm",
            json=json_body(body, PhoneNumberConfirmation),
        )
        return PhoneNumber.model_validate(data)

    def create_phone_call(self, *, body: PhoneCallRequest | Mapping[str, Any]) -> Any:
        """Have Nur call a phone number: say what the call is about, and Nur has the conversation. Fetch the call afterwards for its transcript."""
        return self._request(
            "POST", "/phone/calls", json=json_body(body, PhoneCallRequest)
        )

    def create_realtime_client_secret(
        self, *, body: ClientSecretRequest | Mapping[str, Any]
    ) -> ClientSecretResponse:
        """Mint a one-use credential that opens a realtime session already configured as you specify, for a client that must not hold your key."""
        data = self._request(
            "POST",
            "/realtime/client_secrets",
            json=json_body(body, ClientSecretRequest),
        )
        return ClientSecretResponse.model_validate(data)

    def create_speech(self, *, body: SpeechRequest | Mapping[str, Any]) -> bytes:
        """Create speech."""
        return self._request(
            "POST", "/audio/speech", json=json_body(body, SpeechRequest), raw=True
        )

    def create_transcription(
        self,
        *,
        file: FileInput,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: Literal["json", "text", "verbose_json", "srt", "vtt"]
        | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
        stream: bool | None = None,
    ) -> Any:
        """Create transcription."""
        files: dict[str, FileInput] = {"file": file}
        data: dict[str, Any] = {"model": model}
        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt
        if response_format is not None:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature
        if timestamp_granularities is not None:
            data["timestamp_granularities[]"] = timestamp_granularities
        if stream is not None:
            data["stream"] = stream
        return self._request("POST", "/audio/transcriptions", files=files, data=data)

    def create_voice(
        self,
        *,
        file: FileInput,
        name: str,
        consent_attested: bool | None = None,
        ref_text: str | None = None,
        engine: Literal["nur-tts-v1", "omnivoice"] | None = None,
        category: Literal[
            "general",
            "conversational",
            "narrative",
            "podcast",
            "customer-service",
            "corporate",
            "advertising",
            "characters",
        ]
        | None = None,
        description: str | None = None,
        language: str | None = None,
        gender: Literal["female", "male", "neutral"] | None = None,
        accent: str | None = None,
        prefer_isolated: bool | None = None,
    ) -> VoiceResponse:
        """Clone a voice from 3-10 seconds of reference audio."""
        files: dict[str, FileInput] = {"file": file}
        data: dict[str, Any] = {"name": name}
        if consent_attested is not None:
            data["consent_attested"] = consent_attested
        if ref_text is not None:
            data["ref_text"] = ref_text
        if engine is not None:
            data["engine"] = engine
        if category is not None:
            data["category"] = category
        if description is not None:
            data["description"] = description
        if language is not None:
            data["language"] = language
        if gender is not None:
            data["gender"] = gender
        if accent is not None:
            data["accent"] = accent
        if prefer_isolated is not None:
            data["prefer_isolated"] = prefer_isolated
        data = self._request("POST", "/voices", files=files, data=data)
        return VoiceResponse.model_validate(data)

    def delete_voice(self, voice_id: str) -> VoiceResponse:
        """Delete a cloned voice."""
        data = self._request("DELETE", f"/voices/{voice_id}")
        return VoiceResponse.model_validate(data)

    def forget_phone_number(self, number: str) -> Any:
        """Stop Nur's calls to a number on the shared line until it is confirmed again."""
        return self._request("DELETE", f"/phone/numbers/{number}")

    def get_api_keys(
        self, *, include_archived: bool | None = None
    ) -> list[APIKeyResponse]:
        """Get all API keys for the user's selected organization."""
        params: dict[str, Any] = {}
        if include_archived is not None:
            params["include_archived"] = include_archived
        data = self._request("GET", "/user/api-keys", params=params)
        return [APIKeyResponse.model_validate(x) for x in data]

    def get_billing_credits(
        self, *, page: int | None = None, limit: int | None = None
    ) -> BillingCreditsResponse:
        """Return the organization's prepaid credit balance and paginated ledger."""
        params: dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if limit is not None:
            params["limit"] = limit
        data = self._request("GET", "/organizations/billing/credits", params=params)
        return BillingCreditsResponse.model_validate(data)

    def get_current_period_usage(self) -> CurrentUsageResponse:
        """Get current reporting-period usage for the user's organization."""
        data = self._request("GET", "/organizations/usage/current-period")
        return CurrentUsageResponse.model_validate(data)

    def get_daily_usage_breakdown(
        self, *, days: int | None = None
    ) -> DailyUsageBreakdownResponse:
        """Get daily usage breakdown for the last N days. Only available for organizations with pricing."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        data = self._request(
            "GET", "/organizations/usage/daily-breakdown", params=params
        )
        return DailyUsageBreakdownResponse.model_validate(data)

    def get_failure(self, failure_id: str) -> InteractionFailure:
        """One interaction failure."""
        data = self._request("GET", f"/failures/{failure_id}")
        return InteractionFailure.model_validate(data)

    def get_failure_scorecard(self, *, days: int | None = None) -> ScorecardResponse:
        """Per runtime bundle: sessions, how they ended, evidence coverage, reply latency and failures — the scorecard a release is judged by."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        data = self._request("GET", "/failures/scorecard", params=params)
        return ScorecardResponse.model_validate(data)

    def get_organization_turn_metrics_summary(
        self, *, days: int | None = None
    ) -> OrganizationTurnMetricsSummaryResponse:
        """Organization-wide per-turn latency summary over an optional day window."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        data = self._request(
            "GET", "/organizations/turn-metrics/summary", params=params
        )
        return OrganizationTurnMetricsSummaryResponse.model_validate(data)

    def get_phone_call(self, call_id: str) -> PhoneCall:
        """A call Nur placed: its status, and once it has ended, its transcript."""
        data = self._request("GET", f"/phone/calls/{call_id}")
        return PhoneCall.model_validate(data)

    def get_usage_breakdown(self, *, days: int | None = None) -> UsageBreakdownResponse:
        """Get usage cost broken down by kind, model and API key for the organization."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        data = self._request("GET", "/organizations/usage/breakdown", params=params)
        return UsageBreakdownResponse.model_validate(data)

    def get_usage_history(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int | None = None,
        limit: int | None = None,
    ) -> UsageHistoryResponse:
        """Get the organization's paginated realtime session history."""
        params: dict[str, Any] = {}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if page is not None:
            params["page"] = page
        if limit is not None:
            params["limit"] = limit
        data = self._request("GET", "/organizations/usage/runs", params=params)
        return UsageHistoryResponse.model_validate(data)

    def get_voice(self, voice_id: str) -> VoiceResponse:
        """Get voice."""
        data = self._request("GET", f"/voices/{voice_id}")
        return VoiceResponse.model_validate(data)

    def get_voice_isolation(self, voice_id: str) -> VoiceIsolation:
        """The recording and the voice separated out of it, to compare."""
        data = self._request("GET", f"/voices/{voice_id}/isolation")
        return VoiceIsolation.model_validate(data)

    def list_failures(
        self,
        *,
        status: str | None = None,
        signal: str | None = None,
        severity: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> InteractionFailureList:
        """Interaction failures raised from this organization's sessions, ranked by severity × confidence."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if signal is not None:
            params["signal"] = signal
        if severity is not None:
            params["severity"] = severity
        if session_id is not None:
            params["session_id"] = session_id
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        data = self._request("GET", "/failures", params=params)
        return InteractionFailureList.model_validate(data)

    def list_generations(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        voice: str | None = None,
        search: str | None = None,
    ) -> GenerationListResponse:
        """List past text-to-speech generations."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if voice is not None:
            params["voice"] = voice
        if search is not None:
            params["search"] = search
        data = self._request("GET", "/generations", params=params)
        return GenerationListResponse.model_validate(data)

    def list_phone_numbers(self) -> PhoneNumberList:
        """The numbers confirmed for Nur's calls on the shared EESI line."""
        data = self._request("GET", "/phone/numbers")
        return PhoneNumberList.model_validate(data)

    def list_speech_models(self) -> SpeechModelsResponse:
        """List speech models."""
        data = self._request("GET", "/audio/models")
        return SpeechModelsResponse.model_validate(data)

    def list_voices(self) -> VoicesListResponse:
        """List the organization's cloned voices."""
        data = self._request("GET", "/voices")
        return VoicesListResponse.model_validate(data)

    def phone_listener_events(self, *, wait: float | None = None) -> ListenerEvents:
        """Calls to the code line that have ended, and texts to it, since you last asked; waits up to `wait` seconds for one."""
        params: dict[str, Any] = {}
        if wait is not None:
            params["wait"] = wait
        data = self._request("GET", "/phone/listener/events", params=params)
        return ListenerEvents.model_validate(data)

    def publish_phone_listener(
        self, *, body: ListenerBrief | Mapping[str, Any]
    ) -> ListenerState:
        """Brief the code line: what a call about your code should know. Publish again before it expires; calls to the code line use the latest."""
        data = self._request(
            "PUT", "/phone/listener", json=json_body(body, ListenerBrief)
        )
        return ListenerState.model_validate(data)

    def send_text(self, *, body: TextRequest | Mapping[str, Any]) -> Any:
        """Text a number. On the shared EESI line: a confirmed number, from the code line, so a reply reaches Nur."""
        return self._request(
            "POST", "/phone/messages", json=json_body(body, TextRequest)
        )

    def separate_reference_clip(self, *, file: FileInput) -> SeparatedClip:
        """Separate a voice from what was recorded around it, without storing either."""
        files: dict[str, FileInput] = {"file": file}
        data: dict[str, Any] = {}
        data = self._request("POST", "/voices/separate", files=files, data=data)
        return SeparatedClip.model_validate(data)

    def set_voice_isolation(
        self, voice_id: str, *, body: VoiceIsolationRequest | Mapping[str, Any]
    ) -> VoiceResponse:
        """Speak from the recording, or from the voice isolated out of it."""
        data = self._request(
            "POST",
            f"/voices/{voice_id}/isolation",
            json=json_body(body, VoiceIsolationRequest),
        )
        return VoiceResponse.model_validate(data)

    def update_failure(
        self, failure_id: str, *, body: InteractionFailureUpdate | Mapping[str, Any]
    ) -> InteractionFailure:
        """Move a failure through triage (status) and record what was found (notes)."""
        data = self._request(
            "PATCH",
            f"/failures/{failure_id}",
            json=json_body(body, InteractionFailureUpdate),
        )
        return InteractionFailure.model_validate(data)

    def update_voice(
        self, voice_id: str, *, body: VoiceUpdateRequest | Mapping[str, Any]
    ) -> VoiceResponse:
        """Rename a cloned voice."""
        data = self._request(
            "PATCH", f"/voices/{voice_id}", json=json_body(body, VoiceUpdateRequest)
        )
        return VoiceResponse.model_validate(data)

    def verify_phone_number(
        self, *, body: PhoneNumberRequest | Mapping[str, Any]
    ) -> PhoneNumber:
        """Have Nur phone a number a six-digit code; entering it confirms the number for calls on the shared EESI line."""
        data = self._request(
            "POST", "/phone/numbers", json=json_body(body, PhoneNumberRequest)
        )
        return PhoneNumber.model_validate(data)

    def voice_design_options(self) -> Any:
        """The serving backend's accepted voice-design vocabulary."""
        return self._request("GET", "/audio/voice-design")
