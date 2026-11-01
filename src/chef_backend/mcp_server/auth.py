"""Who is calling a tool: the OAuth access token behind the current MCP request.

MCP clients cannot produce App Check tokens, so /mcp is protected by OAuth access
tokens issued by this backend for this server only (MCP authorization spec). The
token's `subject` is the Firebase uid of the user who signed in and consented.
"""

from functools import lru_cache

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import ProviderTokenVerifier, TokenVerifier
from mcp.server.fastmcp.exceptions import ToolError

RECIPES_READ_SCOPE = "recipes:read"
# Saving recipes. Requested at sign-in together with read; every /mcp request needs
# read, and tools that write check for this one themselves (require_scope).
RECIPES_WRITE_SCOPE = "recipes:write"
SCOPES = [RECIPES_READ_SCOPE, RECIPES_WRITE_SCOPE]

INSUFFICIENT_SCOPE = (
    "insufficient_scope: this connection may not do this yet. "
    "Ask the user to reconnect Chef and allow it on the sign-in page."
)


def require_scope(scope: str) -> None:
    """Raise a tool error the model can relay unless the access token grants `scope`.

    Connections made before a scope existed lack it; reconnecting asks for it.
    """
    token = get_access_token()
    if token is None or scope not in token.scopes:
        raise ToolError(INSUFFICIENT_SCOPE)


class NotAuthenticatedError(Exception):
    """A tool ran without an authenticated user: a wiring bug, never a client error."""


def current_uid() -> str:
    """The Firebase uid of the user the current MCP request acts for."""
    token = get_access_token()
    if token is None or not token.subject:
        raise NotAuthenticatedError("no authenticated user in this MCP request")
    return token.subject


@lru_cache
def get_mcp_token_verifier() -> TokenVerifier:
    """Access tokens are the ones our own authorization server issued (oauth.py)."""
    from chef_backend.mcp_server.oauth import get_oauth_provider

    return ProviderTokenVerifier(get_oauth_provider())
