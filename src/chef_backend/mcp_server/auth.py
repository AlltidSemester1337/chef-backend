"""Who is calling a tool: the OAuth access token behind the current MCP request.

MCP clients cannot produce App Check tokens, so /mcp is protected by OAuth access
tokens issued by this backend for this server only (MCP authorization spec). The
token's `subject` is the Firebase uid of the user who signed in and consented.
"""

from functools import lru_cache

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier

RECIPES_READ_SCOPE = "recipes:read"


class NotAuthenticatedError(Exception):
    """A tool ran without an authenticated user: a wiring bug, never a client error."""


def current_uid() -> str:
    """The Firebase uid of the user the current MCP request acts for."""
    token = get_access_token()
    if token is None or not token.subject:
        raise NotAuthenticatedError("no authenticated user in this MCP request")
    return token.subject


class RejectAllTokenVerifier:
    """Until the authorization server exists (M2), no token is valid: /mcp is closed."""

    async def verify_token(self, token: str) -> AccessToken | None:
        return None


@lru_cache
def get_mcp_token_verifier() -> TokenVerifier:
    return RejectAllTokenVerifier()
