"""EESI SDK — typed Python client for the EESI REST API.

Every method on `EESIClient` is generated from the backend's own OpenAPI
schema (see `scripts/generate_sdk.sh`), so the surface here is exactly the
set of routes carrying an `@sdk_expose` decorator — care, voices,
generations, organization usage and the speech endpoints.

    from eesi_sdk import EESIClient

    with EESIClient(base_url="https://api.eesi.ai", api_key="sk-eesi-...") as client:
        voices = client.list_voices()
"""

from ._generated_client import FileInput
from .client import EESIClient
from .errors import (
    ApiError,
    EESISdkError,
    InvalidRequestError,
    RequestTimeoutError,
)

__all__ = [
    "ApiError",
    "EESIClient",
    "EESISdkError",
    "FileInput",
    "InvalidRequestError",
    "RequestTimeoutError",
]
