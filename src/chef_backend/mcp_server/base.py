"""ChefMCP: FastMCP with per-user resource lists and no internal error messages.

FastMCP only lists resources registered at startup, but a user's recipes differ per
user, so resource listers registered with `@mcp.resource_lister()` add resources
for the current request's user. And FastMCP passes exception messages to the client
in both tool results and resource read errors, which could expose infrastructure
details; here only errors raised on purpose reach the client (tools, resources and
prompts alike).
"""

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Protocol

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import FastMCPError, ResourceError, ToolError
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    Completion,
    CompletionArgument,
    CompletionContext,
    ContentBlock,
    GetPromptResult,
    PromptReference,
    ResourceTemplateReference,
)
from mcp.types import Resource as MCPResource
from pydantic import AnyUrl, ValidationError

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "internal_error"
UNKNOWN_RESOURCE = "unknown_resource"
INVALID_ARGUMENTS = "invalid_arguments"
# Messages FastMCP itself raises for prompts that are safe to show: they name the
# prompt or the missing argument, never a value or an internal detail.
_SAFE_PROMPT_ERRORS = ("Unknown prompt:", "Missing required arguments:")

ResourceLister = Callable[[], Iterable[MCPResource]]
CompletionHandler = Callable[
    [PromptReference | ResourceTemplateReference, CompletionArgument, CompletionContext | None],
    Awaitable[Completion | None],
]


class CompletionDecorator(Protocol):
    def __call__(self, func: CompletionHandler) -> CompletionHandler: ...


class PromptError(FastMCPError):
    """Raised by a prompt function on purpose; its message is shown to the client."""


def _find_in_chain[E: BaseException](error: BaseException, kind: type[E]) -> E | None:
    """The first exception of `kind` in the cause/context chain, under FastMCP's re-wrapping."""
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, kind):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


def _deliberate_resource_error(error: BaseException) -> ResourceError | None:
    return _find_in_chain(error, ResourceError)


class ChefMCP(FastMCP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._resource_listers: list[ResourceLister] = []

    def resource_lister(self) -> Callable[[ResourceLister], ResourceLister]:
        """Register a function that lists resources for the current request's user."""

        def register(fn: ResourceLister) -> ResourceLister:
            self._resource_listers.append(fn)
            return fn

        return register

    def completion(self) -> CompletionDecorator:
        """Register the server's one completion handler, with internal errors hidden.

        Completion is only a convenience: a failing handler is logged and answers with
        no suggestions instead of an error message.
        """
        register = super().completion()

        def wrap(func: CompletionHandler) -> CompletionHandler:
            async def safe(
                ref: PromptReference | ResourceTemplateReference,
                argument: CompletionArgument,
                context: CompletionContext | None,
            ) -> Completion | None:
                try:
                    return await func(ref, argument, context)
                except Exception:
                    logger.exception("MCP completion failed")
                    return None

            register(safe)
            return func

        return wrap

    async def list_resources(self) -> list[MCPResource]:
        listed = await super().list_resources()
        for lister in self._resource_listers:
            try:
                listed.extend(lister())
            except Exception:
                # A broken lister must not hide the tools: list what we can.
                logger.exception("MCP resource lister failed")
        return listed

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        """Tool errors raised on purpose (ToolError) and argument validation errors
        (about the client's own input) pass through; anything else is logged and
        reported as "internal_error"."""
        try:
            return await super().call_tool(name, arguments)
        except ToolError as error:
            if error.__cause__ is None or isinstance(error.__cause__, ToolError | ValidationError):
                raise
            logger.error("MCP tool %s failed", name, exc_info=error.__cause__)
            raise ToolError(INTERNAL_ERROR) from None

    async def read_resource(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        """A ResourceError raised by a resource function passes through with its own
        message; an unknown URI is "unknown_resource"; anything else is logged and
        reported as "internal_error"."""
        try:
            return await super().read_resource(uri)
        except Exception as error:
            deliberate = _deliberate_resource_error(error)
            if deliberate is not None:
                raise ResourceError(str(deliberate)) from None
            if str(error).startswith("Unknown resource"):
                raise ResourceError(UNKNOWN_RESOURCE) from None
            logger.error("MCP resource read failed", exc_info=error)
            raise ResourceError(INTERNAL_ERROR) from None

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """A PromptError raised by a prompt function passes through with its own message, as
        do FastMCP's unknown-prompt and missing-argument errors; invalid arguments are
        "invalid_arguments"; anything else is logged and reported as "internal_error"."""
        try:
            return await super().get_prompt(name, arguments)
        except Exception as error:
            if (deliberate := _find_in_chain(error, PromptError)) is not None:
                raise PromptError(str(deliberate)) from None
            if _find_in_chain(error, ValidationError) is not None:
                raise PromptError(INVALID_ARGUMENTS) from None
            if str(error).startswith(_SAFE_PROMPT_ERRORS):
                raise PromptError(str(error)) from None
            logger.error("MCP prompt %s failed", name, exc_info=error)
            raise PromptError(INTERNAL_ERROR) from None
