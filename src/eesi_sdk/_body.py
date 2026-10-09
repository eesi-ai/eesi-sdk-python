"""Turning whatever a caller passed as `body` into the JSON the API wants.

The generated methods used to call `body.model_dump(...)` straight, so the
first thing anyone who passed a dict — which is what the TypeScript SDK
takes, and what every example of every other API looks like — got back was
`AttributeError: 'dict' object has no attribute 'model_dump'`. That is not
an SDK error, so `except EESISdkError` did not catch it, and it names
nothing about the request.

A dict is now validated against the same model the type hint names. The
request either goes out correct or fails here, before the wire, with the
field and the reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import InvalidRequestError

M = TypeVar("M", bound=BaseModel)


def json_body(body: M | Mapping[str, Any], model: type[M]) -> dict[str, Any]:
    """The request body as JSON, from the model or a mapping of its fields."""
    if isinstance(body, model):
        return body.model_dump(mode="json", exclude_none=True)
    if isinstance(body, Mapping):
        try:
            validated = model.model_validate(dict(body))
        except ValidationError as exc:
            raise InvalidRequestError(
                f"{model.__name__} is not satisfied by the body you passed:\n"
                f"{_describe(exc)}",
                errors=exc.errors(),
            ) from None
        return validated.model_dump(mode="json", exclude_none=True)
    raise InvalidRequestError(
        f"body must be a {model.__name__} or a mapping of its fields, "
        f"not {type(body).__name__}."
    )


def _describe(exc: ValidationError) -> str:
    """One line per problem: the field, then what is wrong with it."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "body"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
