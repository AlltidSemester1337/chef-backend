"""Helpers for MCP tests: a fake token verifier and a JSON-RPC client over TestClient.

The server is stateless and answers with plain JSON, so each call is a single POST.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken

from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE
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
