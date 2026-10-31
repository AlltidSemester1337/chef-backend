"""The MCP sign-in end to end over HTTP, as an MCP client such as Claude does it.

Discovery, dynamic registration, /authorize, the login page and callback, /token
with PKCE, a call to /mcp, refresh and revocation. Firestore and Firebase Auth are
replaced by an in-memory store and a fake ID token verifier.
"""

import base64
import hashlib
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.server.auth.provider import ProviderTokenVerifier
from mcp_support import (
    BASE_URL,
    CLAUDE_CALLBACK,
    OAUTH_CONFIG,
    RESOURCE,
    Clock,
    FakeIdTokenVerifier,
    FakeRecipeRepository,
    InMemoryOAuthStore,
    McpClient,
)

from chef_backend.auth import get_token_verifier
from chef_backend.config import Settings, get_settings
from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.mcp_server.oauth import ChefOAuthProvider, get_oauth_provider
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService

LOGIN_SETTINGS = Settings(
    gcp_project_id="demo-chef",
    firebase_web_api_key="test-web-api-key",
    firebase_web_app_id="1:123:web:abc",
)


@dataclass
class World:
    app: FastAPI
    http: TestClient
    store: InMemoryOAuthStore
    users: FakeIdTokenVerifier
    clock: Clock


@pytest.fixture
def world() -> Iterator[World]:
    store, users, clock = InMemoryOAuthStore(), FakeIdTokenVerifier(), Clock()
    provider = ChefOAuthProvider(store, OAUTH_CONFIG, clock)
    app = create_app()
    app.dependency_overrides.update(
        {
            get_oauth_provider: lambda: provider,
            get_mcp_token_verifier: lambda: ProviderTokenVerifier(provider),
            get_token_verifier: lambda: users,
            get_settings: lambda: LOGIN_SETTINGS,
            get_recipe_service: lambda: RecipeService(FakeRecipeRepository()),
        }
    )
    with TestClient(app, base_url=BASE_URL, follow_redirects=False) as http:
        yield World(app, http, store, users, clock)


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.decode().rstrip("=")


def register(http: TestClient, redirect_uri: str = CLAUDE_CALLBACK, **extra: Any) -> Any:
    body = {
        "redirect_uris": [redirect_uri],
        "client_name": "Claude",
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    } | extra
    return http.post("/register", json=body)


def start_login(http: TestClient, client_id: str, challenge: str) -> str:
    """GET /authorize; return the pending request id from the login page URL."""
    response = http.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "xyz",
            "scope": "recipes:read",
            "resource": RESOURCE,
        },
    )
    assert response.status_code == 302, response.text
    location = urlsplit(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == f"{BASE_URL}/oauth/login"
    return parse_qs(location.query)["request"][0]


def finish_login(http: TestClient, request_id: str, id_token: str) -> Any:
    return http.post("/oauth/callback", json={"request": request_id, "idToken": id_token})


def code_from(redirect_url: str) -> str:
    query = parse_qs(urlsplit(redirect_url).query)
    assert query["state"] == ["xyz"] and query["iss"] == [f"{BASE_URL}/"]
    return query["code"][0]


def exchange(http: TestClient, client_id: str, code: str, verifier: str) -> Any:
    return http.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": CLAUDE_CALLBACK,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": RESOURCE,
        },
    )


def sign_in(world: World) -> tuple[str, dict[str, Any]]:
    """The happy path up to tokens: (client_id, token response)."""
    client_id = register(world.http).json()["client_id"]
    verifier, challenge = pkce()
    request_id = start_login(world.http, client_id, challenge)
    redirect = finish_login(world.http, request_id, world.users.add()).json()["redirectUrl"]
    response = exchange(world.http, client_id, code_from(redirect), verifier)
    assert response.status_code == 200, response.text
    return client_id, response.json()


# --- Discovery ------------------------------------------------------------------------


def test_the_client_finds_the_authorization_server(world: World) -> None:
    resource = world.http.get("/.well-known/oauth-protected-resource/mcp").json()
    [issuer] = resource["authorization_servers"]
    metadata = world.http.get("/.well-known/oauth-authorization-server").json()

    assert metadata["issuer"] == issuer == f"{BASE_URL}/"
    assert metadata["authorization_endpoint"] == f"{BASE_URL}/authorize"
    assert metadata["token_endpoint"] == f"{BASE_URL}/token"
    assert metadata["registration_endpoint"] == f"{BASE_URL}/register"
    assert metadata["code_challenge_methods_supported"] == ["S256"]


# --- The whole flow ----------------------------------------------------------------------


def test_sign_in_then_call_a_tool(world: World) -> None:
    _, tokens = sign_in(world)

    assert tokens["token_type"] == "Bearer" and tokens["scope"] == "recipes:read"
    response = McpClient(world.http, tokens["access_token"]).post("tools/list")
    assert response.status_code == 200


def test_refresh_then_revoke(world: World) -> None:
    client_id, tokens = sign_in(world)

    refreshed = world.http.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": client_id,
            "resource": RESOURCE,
        },
    )
    assert refreshed.status_code == 200, refreshed.text
    new_access = refreshed.json()["access_token"]
    assert McpClient(world.http, new_access).post("tools/list").status_code == 200

    # The SDK's revocation form requires the client_secret field, empty for public clients.
    revoked = world.http.post(
        "/revoke", data={"token": new_access, "client_id": client_id, "client_secret": ""}
    )
    assert revoked.status_code == 200
    assert McpClient(world.http, new_access).post("tools/list").status_code == 401


# --- Registration ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "redirect_uri",
    ["https://evil.example/callback", "https://claude.ai.evil.example/api/mcp/auth_callback"],
)
def test_registration_outside_the_allowlist_is_refused(world: World, redirect_uri: str) -> None:
    response = register(world.http, redirect_uri)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_redirect_uri"
    assert world.store.clients == {}


def test_a_local_client_may_register_a_loopback_redirect(world: World) -> None:
    assert register(world.http, "http://127.0.0.1:43210/callback").status_code == 201


def test_registration_cannot_ask_for_unknown_scopes(world: World) -> None:
    assert register(world.http, scope="recipes:read recipes:write").status_code == 400


# --- The login page ---------------------------------------------------------------------------


def test_the_login_page_names_the_app_and_where_the_user_will_be_sent(world: World) -> None:
    client_id = register(world.http, client_name="Claude <b>").json()["client_id"]
    request_id = start_login(world.http, client_id, pkce()[1])

    page = world.http.get("/oauth/login", params={"request": request_id})

    assert page.status_code == 200
    assert "Connect Claude &lt;b&gt; to Chef" in page.text  # escaped, never raw HTML
    assert "<strong>claude.ai</strong>" in page.text
    assert "Read your recipes" in page.text
    assert 'data-api-key="test-web-api-key"' in page.text


def test_the_login_page_cannot_be_framed_or_run_foreign_scripts(world: World) -> None:
    client_id = register(world.http).json()["client_id"]
    page = world.http.get(
        "/oauth/login", params={"request": start_login(world.http, client_id, pkce()[1])}
    )

    csp = page.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "default-src 'none'" in csp
    assert "script-src 'self' https://www.gstatic.com;" in csp
    assert page.headers["x-frame-options"] == "DENY"
    assert page.headers["referrer-policy"] == "strict-origin"
    assert page.headers["cache-control"] == "no-store"


def test_an_unknown_or_used_request_shows_an_expired_page(world: World) -> None:
    page = world.http.get("/oauth/login", params={"request": "chef_req_unknown"})

    assert page.status_code == 400 and "expired" in page.text


def test_without_the_firebase_web_config_sign_in_is_unavailable(world: World) -> None:
    world.app.dependency_overrides[get_settings] = lambda: Settings()

    assert world.http.get("/oauth/login", params={"request": "x"}).status_code == 503


def test_the_login_script_and_styles_are_served(world: World) -> None:
    script = world.http.get("/oauth/static/login.js")
    styles = world.http.get("/oauth/static/login.css")

    assert script.status_code == 200 and script.headers["content-type"].startswith(
        "text/javascript"
    )
    assert "signInWithEmailAndPassword" in script.text and "inMemoryPersistence" in script.text
    assert styles.status_code == 200 and styles.headers["content-type"].startswith("text/css")


# --- The callback ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_token", "status_code", "detail"),
    [
        (lambda users: "not-a-firebase-token", 401, "invalid_id_token"),
        (lambda users: users.add("t-unverified", email_verified=False), 403, "email_not_verified"),
        (lambda users: users.add("t-anon", provider="anonymous"), 403, "anonymous_not_allowed"),
    ],
    ids=["invalid token", "unverified email", "anonymous"],
)
def test_the_callback_rejects_users_who_may_not_connect(
    world: World, make_token: Any, status_code: int, detail: str
) -> None:
    client_id = register(world.http).json()["client_id"]
    request_id = start_login(world.http, client_id, pkce()[1])

    response = finish_login(world.http, request_id, make_token(world.users))

    assert (response.status_code, response.json()) == (status_code, {"detail": detail})
    # The login is still open: the user can fix it (e.g. verify the email) and retry.
    assert finish_login(world.http, request_id, world.users.add()).status_code == 200


def test_a_login_completes_only_once(world: World) -> None:
    client_id = register(world.http).json()["client_id"]
    request_id = start_login(world.http, client_id, pkce()[1])
    token = world.users.add()

    assert finish_login(world.http, request_id, token).status_code == 200
    second = finish_login(world.http, request_id, token)
    assert (second.status_code, second.json()) == (400, {"detail": "login_expired"})


def test_cancel_returns_access_denied_to_the_client(world: World) -> None:
    client_id = register(world.http).json()["client_id"]
    request_id = start_login(world.http, client_id, pkce()[1])

    response = world.http.post("/oauth/deny", json={"request": request_id})

    query = parse_qs(urlsplit(response.json()["redirectUrl"]).query)
    assert query["error"] == ["access_denied"] and query["state"] == ["xyz"]


def test_the_callback_is_not_open_to_other_sites(world: World) -> None:
    response = world.http.options(
        "/oauth/callback",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in response.headers


# --- /token ----------------------------------------------------------------------------------


def test_a_wrong_code_verifier_gets_no_token(world: World) -> None:
    client_id = register(world.http).json()["client_id"]
    _, challenge = pkce()
    request_id = start_login(world.http, client_id, challenge)
    redirect = finish_login(world.http, request_id, world.users.add()).json()["redirectUrl"]

    response = exchange(world.http, client_id, code_from(redirect), "wrong-verifier-" * 4)

    assert response.status_code == 400 and response.json()["error"] == "invalid_grant"


def test_a_code_cannot_be_used_twice(world: World) -> None:
    client_id = register(world.http).json()["client_id"]
    verifier, challenge = pkce()
    request_id = start_login(world.http, client_id, challenge)
    code = code_from(finish_login(world.http, request_id, world.users.add()).json()["redirectUrl"])

    assert exchange(world.http, client_id, code, verifier).status_code == 200
    assert exchange(world.http, client_id, code, verifier).status_code == 400


def test_another_client_cannot_use_the_code(world: World) -> None:
    client_id = register(world.http).json()["client_id"]
    other_id = register(world.http).json()["client_id"]
    verifier, challenge = pkce()
    request_id = start_login(world.http, client_id, challenge)
    code = code_from(finish_login(world.http, request_id, world.users.add()).json()["redirectUrl"])

    assert exchange(world.http, other_id, code, verifier).status_code == 400
