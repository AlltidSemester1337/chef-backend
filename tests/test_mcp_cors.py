"""Spec for browser access to /mcp: CORS and the Origin check.

Clients like Claude.ai and Claude Code call /mcp from a server or a native process
and send no Origin header, so they are unaffected. A browser client (MCP Inspector
in Direct mode, a future Chef web client) sends one and needs two things:

- the Origin check (the SDK's DNS rebinding protection) to accept its origin, and
- CORS headers, also on the 401 that starts sign-in (WWW-Authenticate exposed).

Only origins in CHEF_MCP_ALLOWED_ORIGINS get either; none are allowed by default.
/mcp authenticates with a bearer token, never cookies, so no credentials are allowed.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_support import BASE_URL, MCP_HEADERS, FakeMcpTokenVerifier, FakeRecipeRepository

from chef_backend.config import Settings
from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService

INSPECTOR = "http://localhost:6274"
EVIL = "https://evil.example"
TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


def client(verifier: FakeMcpTokenVerifier, allowed_origins: list[str]) -> TestClient:
    app = create_app(Settings(public_base_url=BASE_URL, mcp_allowed_origins=allowed_origins))
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(FakeRecipeRepository())
    return TestClient(app, base_url=BASE_URL)


@pytest.fixture
def verifier() -> FakeMcpTokenVerifier:
    return FakeMcpTokenVerifier()


@pytest.fixture
def http(verifier: FakeMcpTokenVerifier) -> Iterator[TestClient]:
    """The Inspector's origin is allowed."""
    with client(verifier, [INSPECTOR]) as http:
        yield http


@pytest.fixture
def closed(verifier: FakeMcpTokenVerifier) -> Iterator[TestClient]:
    """The default: no browser origins allowed."""
    with client(verifier, []) as http:
        yield http


def preflight(http: TestClient, origin: str) -> Any:
    return http.options(
        "/mcp",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type, mcp-protocol-version",
        },
    )


def post(http: TestClient, origin: str | None, token: str | None) -> Any:
    headers = dict(MCP_HEADERS)
    if origin is not None:
        headers["Origin"] = origin
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return http.post("/mcp", json=TOOLS_LIST, headers=headers)


# --- An allowed origin -------------------------------------------------------------------


def test_the_preflight_allows_what_an_mcp_client_sends(http: TestClient) -> None:
    response = preflight(http, INSPECTOR)

    assert response.status_code == 200  # no token needed for a preflight
    assert response.headers["access-control-allow-origin"] == INSPECTOR
    assert "POST" in response.headers["access-control-allow-methods"]
    allowed = response.headers["access-control-allow-headers"].lower()
    for header in ["authorization", "content-type", "mcp-protocol-version"]:
        assert header in allowed, header
    # Bearer tokens only: the browser must never send cookies along.
    assert "access-control-allow-credentials" not in response.headers


def test_an_allowed_origin_reaches_the_server(
    http: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    response = post(http, INSPECTOR, verifier.issue())

    assert response.status_code == 200, response.text
    assert response.headers["access-control-allow-origin"] == INSPECTOR
    assert response.json()["result"]["tools"]


def test_the_sign_in_challenge_is_readable_from_the_browser(http: TestClient) -> None:
    response = post(http, INSPECTOR, token=None)

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == INSPECTOR
    # The client finds the authorization server through WWW-Authenticate.
    assert "www-authenticate" in response.headers["access-control-expose-headers"].lower()


def test_an_allowed_origin_still_needs_a_valid_token(http: TestClient) -> None:
    assert post(http, INSPECTOR, "forged").status_code == 401


# --- Other origins -------------------------------------------------------------------------


def test_an_unlisted_origin_gets_no_cors_headers(http: TestClient) -> None:
    response = preflight(http, EVIL)

    assert "access-control-allow-origin" not in response.headers


def test_an_unlisted_origin_is_refused_even_with_a_valid_token(
    http: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    response = post(http, EVIL, verifier.issue())

    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


def test_by_default_no_browser_origin_is_allowed(
    closed: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    assert "access-control-allow-origin" not in preflight(closed, INSPECTOR).headers
    assert post(closed, INSPECTOR, verifier.issue()).status_code == 403


@pytest.mark.parametrize("origin", [None, BASE_URL], ids=["no origin", "same origin"])
def test_clients_without_a_foreign_origin_are_unaffected(
    closed: TestClient, verifier: FakeMcpTokenVerifier, origin: str | None
) -> None:
    assert post(closed, origin, verifier.issue()).status_code == 200
