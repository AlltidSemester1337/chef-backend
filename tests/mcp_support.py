"""Helpers for MCP tests: fakes for tokens, recipes and OAuth, and a JSON-RPC client.

The server is stateless and answers with plain JSON, so each call is a single POST.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken
from mcp.shared.auth import OAuthClientInformationFull

from chef_backend.auth import Claims, InvalidTokenError
from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE
from chef_backend.mcp_server.oauth import OAuthConfig
from chef_backend.mcp_server.store import (
    PendingAuthorization,
    StoredCode,
    StoredToken,
    secret_hash,
)
from chef_backend.recipes.models import Ingredient, Recipe, RecipeVariant

BASE_URL = "http://localhost:8080"
RESOURCE = f"{BASE_URL}/mcp"
ME, OTHER = "me-uid", "other-secret-uid"
T0 = datetime(2026, 10, 1, tzinfo=UTC)
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


@dataclass
class FakeMcpTokenVerifier:
    """Accepts the tokens in `tokens`; anything else is invalid."""

    tokens: dict[str, AccessToken] = field(default_factory=dict[str, AccessToken])

    def issue(
        self,
        token: str = "valid-token",
        *,
        uid: str = ME,
        scopes: list[str] | None = None,
        resource: str = RESOURCE,
    ) -> str:
        self.tokens[token] = AccessToken(
            token=token,
            client_id="test-client",
            scopes=[RECIPES_READ_SCOPE] if scopes is None else scopes,
            resource=resource,
            subject=uid,
        )
        return token

    async def verify_token(self, token: str) -> AccessToken | None:
        return self.tokens.get(token)


def recipe(recipe_id: str, uid: str | None, *, shared: bool = False, day: int = 0) -> Recipe:
    return Recipe.model_validate(
        {
            "id": recipe_id,
            "uid": uid,
            "title": f"Title {recipe_id}",
            "tags": ["vegetarian"],
            "isFavourite": shared,
            "ingredients": [Ingredient(name="onion", quantity="1", unit="pc")],
            "instructions": ["Chop."],
            "updatedAt": T0 + timedelta(days=day),
        }
    )


@dataclass
class FakeRecipeRepository:
    recipes: list[Recipe] = field(
        default_factory=lambda: [
            recipe("mine", ME, day=1),
            recipe("theirs-shared", OTHER, shared=True, day=2),
            recipe("theirs-private", OTHER, day=3),
        ]
    )

    def list_by_owner(self, uid: str) -> list[Recipe]:
        return [r for r in reversed(self.recipes) if r.uid == uid]

    def list_community(self) -> list[Recipe]:
        return [r for r in reversed(self.recipes) if r.is_favourite]

    def get(self, recipe_id: str) -> Recipe | None:
        return next((r for r in self.recipes if r.id == recipe_id), None)

    def list_variants(self, recipe_id: str) -> list[RecipeVariant]:
        return [
            RecipeVariant.model_validate(
                {"id": "v1", "label": "Mild", "title": "Mild", "createdAt": T0, "isPinned": True}
            )
        ]


class McpClient:
    """Minimal JSON-RPC over the streamable HTTP transport."""

    def __init__(self, http: TestClient, token: str | None) -> None:
        self._http = http
        self._token = token
        self._ids = count(1)

    def post(self, method: str, params: dict[str, Any] | None = None) -> Any:
        headers = dict(MCP_HEADERS)
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            body["params"] = params
        return self._http.post("/mcp", json=body, headers=headers)

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.post("tools/list")
        assert response.status_code == 200, response.text
        return response.json()["result"]["tools"]

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        """The tool result: `isError`, `content` and (on success) `structuredContent`."""
        response = self.post("tools/call", {"name": name, "arguments": arguments})
        assert response.status_code == 200, response.text
        return response.json()["result"]


# --- OAuth: an in-memory store and a fake Firebase ID token verifier --------------


class InMemoryOAuthStore:
    """OAuthStore with the same semantics as FirestoreOAuthStore, keyed by hash."""

    def __init__(self) -> None:
        self.clients: dict[str, tuple[OAuthClientInformationFull, float]] = {}
        self.pending: dict[str, PendingAuthorization] = {}
        self.codes: dict[str, StoredCode] = {}
        self.tokens: dict[str, StoredToken] = {}

    def save_client(self, client: OAuthClientInformationFull, expires_at: float) -> None:
        assert client.client_id is not None
        self.clients[client.client_id] = (client, expires_at)

    def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        found = self.clients.get(client_id)
        return found[0] if found is not None else None

    def save_pending(self, request_id: str, pending: PendingAuthorization) -> None:
        self.pending[secret_hash(request_id)] = pending

    def get_pending(self, request_id: str) -> PendingAuthorization | None:
        return self.pending.get(secret_hash(request_id))

    def take_pending(self, request_id: str) -> PendingAuthorization | None:
        return self.pending.pop(secret_hash(request_id), None)

    def save_code(self, code: str, stored: StoredCode) -> None:
        self.codes[secret_hash(code)] = stored

    def get_code(self, code: str) -> StoredCode | None:
        return self.codes.get(secret_hash(code))

    def take_code(self, code: str) -> StoredCode | None:
        return self.codes.pop(secret_hash(code), None)

    def save_token(self, token: str, stored: StoredToken) -> None:
        self.tokens[secret_hash(token)] = stored

    def get_token(self, token: str) -> StoredToken | None:
        return self.tokens.get(secret_hash(token))

    def mark_rotated(self, token: str) -> bool:
        stored = self.tokens.get(secret_hash(token))
        if stored is None or stored.rotated:
            return False
        self.tokens[secret_hash(token)] = stored.model_copy(update={"rotated": True})
        return True

    def delete_grant(self, grant_id: str) -> None:
        self.tokens = {k: t for k, t in self.tokens.items() if t.grant_id != grant_id}


@dataclass
class FakeIdTokenVerifier:
    """Firebase ID tokens for the login callback: token string -> claims."""

    users: dict[str, Claims] = field(default_factory=dict[str, Claims])

    def add(
        self,
        id_token: str = "id-token-me",
        *,
        uid: str = ME,
        email_verified: bool = True,
        provider: str = "password",
    ) -> str:
        self.users[id_token] = {
            "uid": uid,
            "email": "cook@example.com",
            "email_verified": email_verified,
            "firebase": {"sign_in_provider": provider},
        }
        return id_token

    def verify_id_token(self, token: str) -> Claims:
        if token not in self.users:
            raise InvalidTokenError("invalid ID token")
        return self.users[token]

    def verify_app_check_token(self, token: str) -> Claims:
        raise AssertionError("the login callback must not need App Check")


@dataclass
class Clock:
    now: float = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


OAUTH_CONFIG = OAuthConfig(
    issuer_url=f"{BASE_URL}/",
    resource_url=RESOURCE,
    scopes=[RECIPES_READ_SCOPE],
    redirect_uri_allowlist=["https://claude.ai/api/mcp/auth_callback"],
)
CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"
