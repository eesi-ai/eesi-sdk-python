"""Functions a character can call.

Decorate a function and the character can use it::

    @npc.tool
    async def open_gate(gate: Literal["north", "south"] = "north") -> str:
        \"\"\"Open a castle gate.

        Args:
            gate: Which gate to open.
        \"\"\"
        return await game.open_gate(gate)

The name, the description and a JSON schema of the arguments come from the
function itself: its name, its docstring, its type hints and defaults. The
arguments the model sends are validated against that schema before the
function runs; a bad call is answered with what was wrong, so the model can
try again, and the function never sees it.

What a tool may do is decided in code, never by its description:

- ``allow=`` is checked before every call (the player's reputation, the
  session's permissions). Return False or a reason to refuse.
- ``confirm=True`` makes the character ask the player before acting and call
  again with ``confirmed=true``. That is a conversational check, not an
  authorization: use ``allow=`` for what must be enforced. ``confirm=`` may
  also be an async function, for a confirmation your own UI asks.
- A parameter annotated :class:`ToolContext` receives the calling session,
  player and character instead of a model argument.

A tool that takes a while answers at once with :class:`Started`, and the
character keeps talking; when the work finishes, the character is told and
speaks about it. A tool that runs past ``timeout`` is reported as still
running the same way. Sync functions run in a worker thread, so no tool can
hold up the audio.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import types
import typing
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ._protocol import MAX_TOOL_RESULT_BYTES, clip_bytes
from .errors import ToolDefinitionError

if TYPE_CHECKING:
    from .character import Character
    from .session import Session

log = logging.getLogger("eesi")

__all__ = ["Started", "Tool", "ToolContext", "tool"]

#: The gateway answers for an unanswered tool after 20 s; a quick tool that
#: has not come back well before then is reported as still running instead.
DEFAULT_TIMEOUT_S = 8.0
#: The longest a tool's background work is kept alive after its first answer.
DEFAULT_MAX_DURATION_S = 300.0
_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class ToolContext:
    """Who is asking. Annotate a parameter with this type to receive it."""

    session: Session
    character: Character
    player_id: str | None
    call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Started:
    """Work that has begun and will finish later.

    The character is told *message* at once and keeps talking. When *done*
    settles, its result (or failure) is told to the character as news, and it
    speaks about it when nobody is talking.
    """

    message: str
    done: Awaitable[Any]


Allow = Callable[[ToolContext], "bool | str | None | Awaitable[bool | str | None]"]
Confirm = Callable[[ToolContext], "bool | Awaitable[bool]"]


class Tool:
    """A function the character can call. Build one with :func:`tool`."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_duration: float = DEFAULT_MAX_DURATION_S,
        allow: Allow | None = None,
        confirm: bool | Confirm = False,
        speak_after: bool | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        if not _NAME.match(self.name):
            raise ToolDefinitionError(
                f"Tool name {self.name!r} must be 1-64 letters, digits, '_' or '-'."
            )
        summary, arg_docs = _parse_docstring(inspect.getdoc(fn) or "")
        self.description = (
            description or summary or self.name.replace("_", " ")
        ).strip()
        self.timeout = timeout
        self.max_duration = max_duration
        self.allow = allow
        self.confirm = confirm
        self.speak_after = speak_after
        self._context_params: list[str] = []
        self._model: type[BaseModel] | None = None
        if parameters is not None:
            self.parameters = parameters
        else:
            self._model = self._build_model(arg_docs)
            self.parameters = _clean_schema(self._model.model_json_schema())
        if confirm is True:
            properties = self.parameters.setdefault("properties", {})
            properties["confirmed"] = {
                "type": "boolean",
                "description": "true only after the player clearly agreed to this action in the conversation.",
            }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Still the plain function: a tool can be called from your own code."""
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def _build_model(self, arg_docs: dict[str, str]) -> type[BaseModel]:
        signature = inspect.signature(self.fn)
        try:
            hints = typing.get_type_hints(self.fn, include_extras=True)
        except Exception as error:  # an annotation that cannot be resolved
            raise ToolDefinitionError(
                f"Tool {self.name!r}: cannot read its type hints ({error})."
            ) from error
        fields: dict[str, Any] = {}
        for param in signature.parameters.values():
            annotation = hints.get(param.name, Any)
            if _is_context(annotation):
                self._context_params.append(param.name)
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise ToolDefinitionError(
                    f"Tool {self.name!r}: *args and **kwargs cannot be described to the model; name each argument."
                )
            if param.name == "confirmed" and self.confirm is True:
                continue
            default = ... if param.default is inspect.Parameter.empty else param.default
            description = arg_docs.get(param.name)
            fields[param.name] = (
                annotation,
                Field(default, description=description) if description else default,
            )
        try:
            return create_model(  # type: ignore[call-overload,no-any-return]
                f"{self.name}_arguments",
                __config__=ConfigDict(extra="ignore"),
                **fields,
            )
        except Exception as error:
            raise ToolDefinitionError(
                f"Tool {self.name!r}: its arguments cannot be described as JSON ({error})."
            ) from error

    def bind(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Validated keyword arguments for the function. Raises ValidationError."""
        if self._model is None:
            return dict(raw)
        values = {key: value for key, value in raw.items() if key != "confirmed"}
        model = self._model.model_validate(values)
        return {name: getattr(model, name) for name in type(model).model_fields}


def _is_context(annotation: Any) -> bool:
    """``ToolContext``, also as ``ToolContext | None``."""
    if annotation is ToolContext:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return ToolContext in typing.get_args(annotation)
    return False


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    allow: Allow | None = None,
    confirm: bool | Confirm = False,
    speak_after: bool | None = None,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Make *fn* a :class:`Tool`. Use bare (``@tool``) or with options (``@tool(confirm=True)``).

    ``speak_after`` says whether the character speaks again once the result
    is in. By default it does when the tool returned something and not when
    it returned None (an action the character already announced). A reply
    that was nothing but tool calls is always followed up.
    """

    def wrap(function: Callable[..., Any]) -> Tool:
        if isinstance(function, Tool):
            return function
        return Tool(
            function,
            name=name,
            description=description,
            timeout=timeout,
            max_duration=max_duration,
            allow=allow,
            confirm=confirm,
            speak_after=speak_after,
            parameters=parameters,
        )

    return wrap(fn) if fn is not None else wrap


def as_tools(items: Iterable[Tool | Callable[..., Any]]) -> list[Tool]:
    tools: list[Tool] = []
    names: set[str] = set()
    for item in items:
        made = item if isinstance(item, Tool) else tool(item)
        if made.name in names:
            raise ToolDefinitionError(f"Two tools are named {made.name!r}.")
        names.add(made.name)
        tools.append(made)
    return tools


# ── running calls ────────────────────────────────────────────────────────


@dataclass
class Outcome:
    """What a call produced: the text the character reads and whether it should speak about it."""

    output: str
    speak_after: bool
    later: Awaitable[Any] | None = None


def render_result(value: Any) -> str:
    """A tool's return value as the text the model reads."""
    if value is None:
        return "Done."
    if isinstance(value, str):
        return value or "Done."
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


class ToolRunner:
    """One session's calls: validate, authorize, run, time out, answer once."""

    def __init__(
        self,
        tools: Iterable[Tool],
        *,
        context: Callable[[str, dict[str, Any]], ToolContext],
        on_event: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.tools = {item.name: item for item in tools}
        self.disabled: set[str] = set()
        self._context = context
        self._on_event = on_event
        self._seen: dict[str, None] = {}
        self._background: set[asyncio.Task[Any]] = set()

    def specs(self) -> list[dict[str, Any]]:
        return [
            item.spec()
            for name, item in self.tools.items()
            if name not in self.disabled
        ]

    def seen(self, call_id: str) -> bool:
        """Record *call_id*; True when it was already handled (a duplicate frame)."""
        if call_id in self._seen:
            return True
        self._seen[call_id] = None
        if len(self._seen) > 512:
            self._seen.pop(next(iter(self._seen)))
        return False

    async def run(self, call_id: str, name: str, raw_arguments: str) -> Outcome:
        """Answer one call. Never raises."""
        started = time.monotonic()
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            if not isinstance(arguments, dict):
                raise ValueError
        except ValueError:
            return self._fail(
                call_id,
                name,
                {},
                "invalid_arguments",
                "The arguments were not a JSON object.",
            )
        item = self.tools.get(name)
        if item is None or name in self.disabled:
            return self._fail(
                call_id,
                name,
                arguments,
                "unknown_tool",
                f"No tool named {name} is available.",
            )
        try:
            kwargs = item.bind(arguments)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc']) or 'arguments'}: {issue['msg']}"
                for issue in error.errors()[:5]
            )
            return self._fail(
                call_id,
                name,
                arguments,
                "invalid_arguments",
                f"The arguments were not valid: {problems}.",
            )
        ctx = self._context(call_id, arguments)
        if item.allow is not None:
            try:
                verdict = item.allow(ctx)
                if inspect.isawaitable(verdict):
                    verdict = await verdict
            except Exception as error:
                log.exception("allow check for %s failed", name)
                verdict = f"the permission check failed ({error})"
            if verdict is False or isinstance(verdict, str):
                reason = (
                    verdict
                    if isinstance(verdict, str)
                    else "not permitted for this player right now"
                )
                return self._fail(
                    call_id,
                    name,
                    arguments,
                    "not_allowed",
                    f"Not allowed: {reason}. Nothing was done.",
                )
        if item.confirm is True and arguments.get("confirmed") is not True:
            return self._fail(
                call_id,
                name,
                arguments,
                "not_confirmed",
                f"Nothing was done yet. Ask the player to confirm ({item.description.rstrip('.')}) "
                f"and call {name} again with confirmed=true only if they clearly agree.",
                speak_after=True,
            )
        if callable(item.confirm):
            try:
                approved = item.confirm(ctx)
                if inspect.isawaitable(approved):
                    approved = await asyncio.wait_for(approved, item.timeout)
            except Exception:
                approved = False
            if approved is not True:
                return self._fail(
                    call_id,
                    name,
                    arguments,
                    "not_confirmed",
                    "The player did not confirm. Nothing was done.",
                )
        for param in item._context_params:
            kwargs[param] = ctx
        self._on_event(
            "tool.started", {"call_id": call_id, "name": name, "arguments": arguments}
        )
        work = asyncio.ensure_future(self._invoke(item, kwargs))
        try:
            value = await asyncio.wait_for(asyncio.shield(work), item.timeout)
        except asyncio.TimeoutError:
            self._keep(work, item, call_id, arguments, started)
            return Outcome(
                f"{name} is still working; you will be told when it finishes. Do not claim it is done.",
                speak_after=False,
                later=work,
            )
        except Exception as error:
            return self._fail(
                call_id,
                name,
                arguments,
                "exception",
                f"That failed: {error}",
                error=error,
            )
        if isinstance(value, Started):
            later = asyncio.ensure_future(_await(value.done))
            self._keep(later, item, call_id, arguments, started)
            output = clip_bytes(value.message or "Started.", MAX_TOOL_RESULT_BYTES)
            self._on_event(
                "tool.completed",
                {
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "result": value,
                    "output": output,
                    "duration": time.monotonic() - started,
                },
            )
            return Outcome(output, speak_after=item.speak_after is True, later=later)
        output = clip_bytes(render_result(value), MAX_TOOL_RESULT_BYTES)
        self._on_event(
            "tool.completed",
            {
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "result": value,
                "output": output,
                "duration": time.monotonic() - started,
            },
        )
        speak = item.speak_after if item.speak_after is not None else value is not None
        return Outcome(output, speak_after=speak)

    def _fail(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        reason: str,
        message: str,
        *,
        speak_after: bool = True,
        error: BaseException | None = None,
    ) -> Outcome:
        if error is not None:
            log.warning("tool %s failed: %s", name, error, exc_info=error)
        self._on_event(
            "tool.failed",
            {
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "reason": reason,
                "error": message,
            },
        )
        return Outcome(
            clip_bytes(message, MAX_TOOL_RESULT_BYTES), speak_after=speak_after
        )

    async def _invoke(self, item: Tool, kwargs: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(item.fn):
            return await item.fn(**kwargs)
        result = await asyncio.to_thread(item.fn, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _keep(
        self,
        work: asyncio.Future[Any],
        item: Tool,
        call_id: str,
        arguments: dict[str, Any],
        started: float,
    ) -> None:
        """Let background work finish (or be cut off at max_duration)."""

        async def watch() -> None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(work),
                    max(0.0, item.max_duration - (time.monotonic() - started)),
                )
            except asyncio.TimeoutError:
                work.cancel()
            except Exception:
                pass

        task = asyncio.ensure_future(watch())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def close(self) -> None:
        for task in list(self._background):
            task.cancel()
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)


async def _await(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


# ── schema helpers ───────────────────────────────────────────────────────

_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Params|Returns|Raises|Yields|Example|Examples|Note|Notes):\s*$"
)
_ARG = re.compile(r"^\s{2,}(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """The summary and the per-argument descriptions of a Google-style docstring."""
    summary: list[str] = []
    args: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    for line in doc.splitlines():
        header = _SECTION.match(line.strip())
        if header and not line.startswith(" "):
            section = header.group(1)
            current = None
            continue
        if section is None:
            summary.append(line)
            continue
        if section in ("Args", "Arguments", "Parameters", "Params"):
            match = _ARG.match(line)
            if match:
                current = match.group(1).lstrip("*")
                args[current] = match.group(3).strip()
            elif current and line.strip():
                args[current] = f"{args[current]} {line.strip()}".strip()
    text = " ".join(" ".join(summary).split())
    return text, args


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A pydantic schema as a compact, ref-free JSON schema for the model."""
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                target = defs.get(node["$ref"].split("/")[-1], {})
                merged = {
                    **resolve(target),
                    **{k: v for k, v in node.items() if k != "$ref"},
                }
                return resolve(merged)
            return {
                key: resolve(value) for key, value in node.items() if key != "title"
            }
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    cleaned = resolve(schema)
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    if "required" not in cleaned:
        cleaned["required"] = []
    return cleaned
