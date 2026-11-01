"""ChefMCP: FastMCP with per-user resource lists and no internal error messages.

FastMCP only lists resources registered at startup, but a user's recipes differ per
user, so resource listers registered with `@mcp.resource_lister()` add resources
for the current request's user. And FastMCP passes exception messages to the client
in both tool results and resource read errors, which could expose infrastructure
details; here only errors raised on purpose reach the client.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import ContentBlock
from mcp.types import Resource as MCPResource
from pydantic import AnyUrl, ValidationError

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "internal_error"
UNKNOWN_RESOURCE = "unknown_resource"

ResourceLister = Callable[[], Iterable[MCPResource]]


def _deliberate_resource_error(error: BaseException) -> ResourceError | None:
    """The ResourceError a resource function raised, under FastMCP's re-wrapping."""
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, ResourceError):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


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
