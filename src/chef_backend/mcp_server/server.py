"""Builds the FastMCP server and its ASGI app, mounted into the FastAPI app."""

import logging
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ContentBlock
from pydantic import AnyHttpUrl, ValidationError

from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE
from chef_backend.mcp_server.tools import register_tools
from chef_backend.recipes.service import RecipeService

logger = logging.getLogger(__name__)

MCP_PATH = "/mcp"
INTERNAL_ERROR = "internal_error"


class _ChefMCP(FastMCP):
    """FastMCP that does not hand internal error messages to the client.

    FastMCP turns every exception in a tool into a tool error carrying its message,
    which could expose infrastructure details. Errors a tool raises on purpose
    (ToolError) and argument validation errors (about the client's own input) pass
    through; anything else is logged here and reported as "internal_error".
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        try:
            return await super().call_tool(name, arguments)
        except ToolError as error:
            if error.__cause__ is None or isinstance(error.__cause__, ToolError | ValidationError):
                raise
            logger.error("MCP tool %s failed", name, exc_info=error.__cause__)
            raise ToolError(INTERNAL_ERROR) from None


class _LazyTokenVerifier:
    """Resolves the verifier per request, so tests can swap it after the app is built."""

    def __init__(self, get_verifier: Callable[[], TokenVerifier]) -> None:
        self._get_verifier = get_verifier

    async def verify_token(self, token: str) -> AccessToken | None:
        return await self._get_verifier().verify_token(token)


def build_mcp(
    base_url: str,
    token_verifier: Callable[[], TokenVerifier],
    recipe_service: Callable[[], RecipeService],
) -> FastMCP:
    host = urlsplit(base_url).netloc
    mcp = _ChefMCP(
        name="Chef",
        instructions="Search and read the user's recipes and the community's shared recipes.",
        # Cloud Run scales to zero and may run two instances: no server-side sessions.
        stateless_http=True,
        json_response=True,
        streamable_http_path=MCP_PATH,
        token_verifier=_LazyTokenVerifier(token_verifier),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(base_url),
            resource_server_url=AnyHttpUrl(base_url + MCP_PATH),
            required_scopes=[RECIPES_READ_SCOPE],
            validate_token_resource=True,
        ),
        transport_security=TransportSecuritySettings(
            allowed_hosts=[host], allowed_origins=[f"{urlsplit(base_url).scheme}://{host}"]
        ),
    )
    register_tools(mcp, recipe_service)
    return mcp
