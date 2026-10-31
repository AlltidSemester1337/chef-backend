"""The MCP endpoint's wiring: OAuth resource-server behaviour, transport, mounting.

The tools have their own spec (test_mcp_tools.py); the token verifier here is a fake.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import ProviderTokenVerifier
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp_support import BASE_URL, RESOURCE, FakeMcpTokenVerifier, FakeRecipeRepository, McpClient

from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService

METADATA_PATH = "/.well-known/oauth-protected-resource/mcp"


@pytest.fixture
def verifier() -> FakeMcpTokenVerifier:
    return FakeMcpTokenVerifier()


@pytest.fixture
def http(verifier: FakeMcpTokenVerifier) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(FakeRecipeRepository())
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


def test_without_a_token_the_client_is_pointed_to_the_resource_metadata(
    http: TestClient,
) -> None:
    response = McpClient(http, token=None).post("tools/list")

    assert response.status_code == 401
    challenge = response.headers["www-authenticate"]
    assert challenge.startswith("Bearer ")
    assert f'resource_metadata="{BASE_URL}{METADATA_PATH}"' in challenge


def test_protected_resource_metadata(http: TestClient) -> None:
    metadata = http.get(METADATA_PATH).json()

    assert metadata["resource"] == RESOURCE
    assert metadata["authorization_servers"] == [f"{BASE_URL}/"]
    assert metadata["scopes_supported"] == ["recipes:read"]


def test_an_unknown_token_is_401(http: TestClient) -> None:
    assert McpClient(http, "forged").post("tools/list").status_code == 401


def test_a_token_for_another_resource_is_401(
    http: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    token = verifier.issue(resource="https://other.example.com/mcp")

    assert McpClient(http, token).post("tools/list").status_code == 401


def test_a_token_without_the_recipes_scope_is_403(
    http: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    token = verifier.issue(scopes=["something:else"])

    assert McpClient(http, token).post("tools/list").status_code == 403


def test_a_valid_token_reaches_the_server(http: TestClient, verifier: FakeMcpTokenVerifier) -> None:
    response = McpClient(http, verifier.issue()).post("tools/list")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_an_unexpected_host_header_is_rejected(
    http: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    # DNS rebinding protection: only the configured public host is served.
    response = http.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Host": "evil.example.com", "Authorization": f"Bearer {verifier.issue()}"}
        | {"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code == 421


def test_in_production_access_tokens_are_checked_by_our_authorization_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object()
    monkeypatch.setattr("chef_backend.mcp_server.oauth.get_oauth_provider", lambda: provider)
    get_mcp_token_verifier.cache_clear()
    try:
        verifier = get_mcp_token_verifier()
    finally:
        get_mcp_token_verifier.cache_clear()

    assert isinstance(verifier, ProviderTokenVerifier)
    assert verifier.provider is provider  # pyright: ignore[reportUnknownMemberType]


def test_the_http_api_is_unaffected_and_unknown_paths_are_404(http: TestClient) -> None:
    assert http.get("/health").json()["status"] == "ok"
    assert http.get("/openapi.json").status_code == 200
    assert http.get("/no-such-path").status_code == 404


# --- Tool errors ---------------------------------------------------------------


@pytest.fixture
def app_with_test_tools(verifier: FakeMcpTokenVerifier) -> Iterator[McpClient]:
    """The real app plus throwaway tools, independent of the recipe tools."""
    app = create_app()
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    mcp: FastMCP = app.state.mcp

    @mcp.tool()
    def crash() -> str:
        raise RuntimeError("projects/secret-project/databases/(default) detail")

    @mcp.tool()
    def refuse() -> str:
        raise ToolError("recipe_not_found")

    @mcp.tool()
    def echo(n: int) -> int:
        return n

    with TestClient(app, base_url=BASE_URL) as client:
        yield McpClient(client, verifier.issue())


def test_unexpected_tool_failures_are_internal_error_without_details(
    app_with_test_tools: McpClient, caplog: pytest.LogCaptureFixture
) -> None:
    result = app_with_test_tools.call("crash")

    assert result["isError"] is True
    assert result["content"][0]["text"] == "internal_error"
    assert "secret-project" in caplog.text  # logged for us, not sent to the client


def test_deliberate_tool_errors_and_validation_errors_reach_the_client(
    app_with_test_tools: McpClient,
) -> None:
    refused = app_with_test_tools.call("refuse")
    invalid = app_with_test_tools.call("echo", n="not a number")

    assert refused["isError"] is True and "recipe_not_found" in refused["content"][0]["text"]
    assert invalid["isError"] is True and "internal_error" not in invalid["content"][0]["text"]


# --- Stateless transport -------------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_no_event_stream_or_session_end_is_offered(
    http: TestClient, verifier: FakeMcpTokenVerifier, method: str
) -> None:
    # An open GET stream would keep the Cloud Run instance busy (and billed) forever.
    response = http.request(
        method,
        "/mcp",
        headers={"Authorization": f"Bearer {verifier.issue()}", "Accept": "text/event-stream"},
    )

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
