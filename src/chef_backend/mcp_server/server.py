"""Builds the FastMCP server and its ASGI app, mounted into the FastAPI app."""

import logging
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenVerifier,
)
from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.types import ContentBlock
from pydantic import AnyHttpUrl, ValidationError
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE
from chef_backend.mcp_server.oauth import ChefOAuthProvider
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


class _LazyOAuthProvider:
    """Resolves the provider per call, so tests can swap it after the app is built."""

    def __init__(self, get_provider: Callable[[], ChefOAuthProvider]) -> None:
        self._get = get_provider

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return await self._get().get_client(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        await self._get().register_client(client_info)

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        return await self._get().authorize(client, params)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return await self._get().load_authorization_code(client, authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        return await self._get().exchange_authorization_code(client, authorization_code)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return await self._get().load_refresh_token(client, refresh_token)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        return await self._get().exchange_refresh_token(client, refresh_token, scopes)

    async def load_access_token(self, token: str) -> AccessToken | None:
        return await self._get().load_access_token(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        await self._get().revoke_token(token)


class StatelessOnly:
    """Answers GET and DELETE on /mcp with 405, as the transport spec allows a server to.

    A stateless server can never send anything on the GET event stream, but the SDK
    would hold it open until Cloud Run's request timeout. An open request keeps the
    instance billed and stops it scaling to zero for as long as a client is connected.
    DELETE ends a session, and there are none. Everything else goes to the MCP app.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["path"] == MCP_PATH
            and scope["method"] in ("GET", "DELETE")
        ):
            response = Response(status_code=405, headers={"Allow": "POST"})
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


def build_oauth_routes(
    base_url: str, oauth_provider: Callable[[], ChefOAuthProvider]
) -> list[Route]:
    """The authorization server endpoints: metadata, /authorize, /token, /register, /revoke.

    Registration is open, as the MCP spec expects, but limited to allowlisted
    redirect URIs (ChefOAuthProvider.register_client).
    """
    return create_auth_routes(
        provider=_LazyOAuthProvider(oauth_provider),
        issuer_url=AnyHttpUrl(base_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[RECIPES_READ_SCOPE],
            default_scopes=[RECIPES_READ_SCOPE],
        ),
        revocation_options=RevocationOptions(enabled=True),
    )


def build_mcp(
    base_url: str,
    token_verifier: Callable[[], TokenVerifier],
    recipe_service: Callable[[], RecipeService],
) -> FastMCP:
    host = urlsplit(base_url).netloc
    mcp = _ChefMCP(
        name="Chef",
        instructions=(
            "Chef holds the user's recipes and recipes shared by the community. "
            "Recipes are stored in English: translate search words to English, but answer "
            "the user in their own language. Typical flow: search_recipes to find "
            "candidates, then get_recipe for the one or few recipes you need in full."
        ),
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
