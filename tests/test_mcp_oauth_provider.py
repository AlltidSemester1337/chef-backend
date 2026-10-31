"""ChefOAuthProvider: registration, login completion, codes, tokens, refresh, revoke.

Unit tests with an in-memory store and a fixed clock. The SDK's own checks (PKCE,
redirect URI match, code expiry at /token) are covered end to end in
test_mcp_oauth_flow.py.
"""

import pytest
from mcp.server.auth.provider import (
    AuthorizationParams,
    AuthorizeError,
    RegistrationError,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp_support import CLAUDE_CALLBACK, ME, OAUTH_CONFIG, RESOURCE, Clock, InMemoryOAuthStore
from pydantic import AnyUrl

from chef_backend.mcp_server.oauth import (
    ACCESS_TOKEN_TTL_SECONDS,
    CODE_TTL_SECONDS,
    PENDING_TTL_SECONDS,
    REFRESH_TOKEN_TTL_SECONDS,
    ChefOAuthProvider,
    is_allowed_redirect_uri,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def store() -> InMemoryOAuthStore:
    return InMemoryOAuthStore()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def provider(store: InMemoryOAuthStore, clock: Clock) -> ChefOAuthProvider:
    return ChefOAuthProvider(store, OAUTH_CONFIG, clock)


def client(
    client_id: str = "client-1", redirect: str = CLAUDE_CALLBACK
) -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id, client_name="Claude", redirect_uris=[AnyUrl(redirect)]
    )


def params(**overrides: object) -> AuthorizationParams:
    values: dict[str, object] = {
        "state": "state-1",
        "scopes": ["recipes:read"],
        "code_challenge": "challenge",
        "redirect_uri": AnyUrl(CLAUDE_CALLBACK),
        "redirect_uri_provided_explicitly": True,
        "resource": RESOURCE,
    }
    return AuthorizationParams.model_validate(values | overrides)


async def signed_in(provider: ChefOAuthProvider, the_client: OAuthClientInformationFull) -> str:
    """Run /authorize and the login callback; return the one-time code."""
    login_url = await provider.authorize(the_client, params())
    request_id = login_url.split("request=", 1)[1]
    redirect = provider.complete_login(request_id, uid=ME)
    assert redirect is not None
    return redirect.split("code=", 1)[1].split("&", 1)[0]


async def tokens_for(
    provider: ChefOAuthProvider, the_client: OAuthClientInformationFull
) -> OAuthToken:
    code = await signed_in(provider, the_client)
    loaded = await provider.load_authorization_code(the_client, code)
    assert loaded is not None
    return await provider.exchange_authorization_code(the_client, loaded)


# --- Redirect URI allowlist ----------------------------------------------------

ALLOWLIST = ["https://claude.ai/api/mcp/auth_callback"]


@pytest.mark.parametrize(
    "uri",
    [
        "https://claude.ai/api/mcp/auth_callback",
        "http://localhost:33418/callback",
        "http://127.0.0.1:5173/oauth/callback",
        "http://[::1]:8080/cb",
        "http://localhost/callback",
    ],
)
def test_allowed_redirect_uris(uri: str) -> None:
    assert is_allowed_redirect_uri(uri, ALLOWLIST)


@pytest.mark.parametrize(
    "uri",
    [
        "https://evil.example/cb",
        "https://claude.ai/api/mcp/auth_callback/../evil",
        "https://claude.ai.evil.example/api/mcp/auth_callback",
        "https://localhost:8080/callback",  # loopback is http only (RFC 8252)
        "http://localhost.evil.example/callback",
        "http://evil.example/localhost:80",
        "http://user@evil.example/",
        "javascript:alert(1)",
    ],
)
def test_other_redirect_uris_are_rejected(uri: str) -> None:
    assert not is_allowed_redirect_uri(uri, ALLOWLIST)


async def test_registration_with_a_disallowed_redirect_uri_fails(
    provider: ChefOAuthProvider, store: InMemoryOAuthStore
) -> None:
    with pytest.raises(RegistrationError) as error:
        await provider.register_client(client(redirect="https://evil.example/cb"))

    assert error.value.error == "invalid_redirect_uri"
    assert store.clients == {}


async def test_registration_stores_the_client_with_an_expiry(
    provider: ChefOAuthProvider, store: InMemoryOAuthStore, clock: Clock
) -> None:
    await provider.register_client(client())

    assert await provider.get_client("client-1") == client()
    assert store.clients["client-1"][1] > clock.now


async def test_an_overlong_client_name_is_rejected(provider: ChefOAuthProvider) -> None:
    long_name = client().model_copy(update={"client_name": "x" * 101})

    with pytest.raises(RegistrationError):
        await provider.register_client(long_name)


# --- /authorize and the login ---------------------------------------------------


async def test_authorize_sends_the_user_to_the_login_page(provider: ChefOAuthProvider) -> None:
    url = await provider.authorize(client(), params())

    assert url.startswith(f"{OAUTH_CONFIG.issuer_url.rstrip('/')}/oauth/login?request=chef_req_")


async def test_authorize_for_another_resource_is_refused(provider: ChefOAuthProvider) -> None:
    with pytest.raises(AuthorizeError):
        await provider.authorize(client(), params(resource="https://other.example/mcp"))


async def test_authorize_without_resource_binds_to_this_server(
    provider: ChefOAuthProvider, store: InMemoryOAuthStore
) -> None:
    await provider.authorize(client(), params(resource=None))

    [pending] = store.pending.values()
    assert pending.resource == RESOURCE


async def test_completing_the_login_redirects_with_code_state_and_issuer(
    provider: ChefOAuthProvider,
) -> None:
    login_url = await provider.authorize(client(), params())
    request_id = login_url.split("request=", 1)[1]

    redirect = provider.complete_login(request_id, uid=ME)

    assert redirect is not None and redirect.startswith(f"{CLAUDE_CALLBACK}?code=chef_code_")
    assert "state=state-1" in redirect
    assert "iss=http%3A%2F%2Flocalhost%3A8080%2F" in redirect


async def test_a_login_completes_only_once(provider: ChefOAuthProvider) -> None:
    request_id = (await provider.authorize(client(), params())).split("request=", 1)[1]

    assert provider.complete_login(request_id, uid=ME) is not None
    assert provider.complete_login(request_id, uid=ME) is None


async def test_an_expired_login_does_not_complete(
    provider: ChefOAuthProvider, clock: Clock
) -> None:
    request_id = (await provider.authorize(client(), params())).split("request=", 1)[1]
    clock.now += PENDING_TTL_SECONDS + 1

    assert provider.pending_login(request_id) is None
    assert provider.complete_login(request_id, uid=ME) is None


async def test_denying_redirects_with_access_denied_and_ends_the_login(
    provider: ChefOAuthProvider,
) -> None:
    request_id = (await provider.authorize(client(), params())).split("request=", 1)[1]

    redirect = provider.deny_login(request_id)

    assert (
        redirect is not None and "error=access_denied" in redirect and "state=state-1" in redirect
    )
    assert provider.complete_login(request_id, uid=ME) is None


# --- Codes ------------------------------------------------------------------------


async def test_a_code_belongs_to_its_client(provider: ChefOAuthProvider) -> None:
    code = await signed_in(provider, client())

    assert await provider.load_authorization_code(client("someone-else"), code) is None
    loaded = await provider.load_authorization_code(client(), code)
    assert loaded is not None and loaded.subject == ME and loaded.resource == RESOURCE


async def test_a_code_is_exchanged_only_once(provider: ChefOAuthProvider) -> None:
    code = await signed_in(provider, client())
    loaded = await provider.load_authorization_code(client(), code)
    assert loaded is not None

    await provider.exchange_authorization_code(client(), loaded)
    with pytest.raises(TokenError):
        await provider.exchange_authorization_code(client(), loaded)
    assert await provider.load_authorization_code(client(), code) is None


async def test_codes_live_five_minutes(provider: ChefOAuthProvider, clock: Clock) -> None:
    loaded = await provider.load_authorization_code(client(), await signed_in(provider, client()))

    assert loaded is not None and loaded.expires_at == clock.now + CODE_TTL_SECONDS


# --- Access tokens -------------------------------------------------------------------


async def test_the_exchange_issues_access_and_refresh_tokens(
    provider: ChefOAuthProvider, store: InMemoryOAuthStore
) -> None:
    tokens = await tokens_for(provider, client())

    assert tokens.access_token.startswith("chef_at_")
    assert tokens.refresh_token is not None and tokens.refresh_token.startswith("chef_rt_")
    assert (tokens.expires_in, tokens.scope) == (ACCESS_TOKEN_TTL_SECONDS, "recipes:read")
    # Only hashes are stored.
    assert tokens.access_token not in str(store.tokens)


async def test_an_access_token_identifies_the_user(provider: ChefOAuthProvider) -> None:
    tokens = await tokens_for(provider, client())

    access = await provider.load_access_token(tokens.access_token)

    assert access is not None
    assert (access.subject, access.client_id, access.scopes) == (ME, "client-1", ["recipes:read"])
    assert access.resource == RESOURCE


async def test_access_tokens_expire_after_an_hour(
    provider: ChefOAuthProvider, clock: Clock
) -> None:
    tokens = await tokens_for(provider, client())
    clock.now += ACCESS_TOKEN_TTL_SECONDS

    assert await provider.load_access_token(tokens.access_token) is None


async def test_a_refresh_token_is_not_an_access_token(provider: ChefOAuthProvider) -> None:
    tokens = await tokens_for(provider, client())
    assert tokens.refresh_token is not None

    assert await provider.load_access_token(tokens.refresh_token) is None
    assert await provider.load_access_token("chef_at_forged") is None


# --- Refresh ------------------------------------------------------------------------


async def test_refresh_rotates_both_tokens(provider: ChefOAuthProvider) -> None:
    first = await tokens_for(provider, client())
    assert first.refresh_token is not None
    refresh = await provider.load_refresh_token(client(), first.refresh_token)
    assert refresh is not None and refresh.subject == ME

    second = await provider.exchange_refresh_token(client(), refresh, ["recipes:read"])

    assert second.access_token != first.access_token
    assert second.refresh_token not in (None, first.refresh_token)
    assert await provider.load_access_token(second.access_token) is not None


async def test_reusing_a_rotated_refresh_token_revokes_the_whole_grant(
    provider: ChefOAuthProvider,
) -> None:
    first = await tokens_for(provider, client())
    assert first.refresh_token is not None
    refresh = await provider.load_refresh_token(client(), first.refresh_token)
    assert refresh is not None
    second = await provider.exchange_refresh_token(client(), refresh, ["recipes:read"])

    # The old refresh token comes back: someone has a copy.
    assert await provider.load_refresh_token(client(), first.refresh_token) is None
    assert await provider.load_access_token(second.access_token) is None
    assert second.refresh_token is not None
    assert await provider.load_refresh_token(client(), second.refresh_token) is None


async def test_a_refresh_token_belongs_to_its_client(provider: ChefOAuthProvider) -> None:
    tokens = await tokens_for(provider, client())
    assert tokens.refresh_token is not None

    assert await provider.load_refresh_token(client("someone-else"), tokens.refresh_token) is None
    assert await provider.load_refresh_token(client(), tokens.access_token) is None


async def test_refresh_tokens_expire_after_thirty_days(
    provider: ChefOAuthProvider, clock: Clock
) -> None:
    tokens = await tokens_for(provider, client())
    assert tokens.refresh_token is not None
    clock.now += REFRESH_TOKEN_TTL_SECONDS

    assert await provider.load_refresh_token(client(), tokens.refresh_token) is None


async def test_two_concurrent_refreshes_cannot_both_win(provider: ChefOAuthProvider) -> None:
    tokens = await tokens_for(provider, client())
    assert tokens.refresh_token is not None
    refresh = await provider.load_refresh_token(client(), tokens.refresh_token)
    assert refresh is not None

    await provider.exchange_refresh_token(client(), refresh, ["recipes:read"])
    with pytest.raises(TokenError):
        await provider.exchange_refresh_token(client(), refresh, ["recipes:read"])


# --- Revocation ---------------------------------------------------------------------


async def test_revoking_either_token_ends_the_grant(provider: ChefOAuthProvider) -> None:
    tokens = await tokens_for(provider, client())
    access = await provider.load_access_token(tokens.access_token)
    assert access is not None and tokens.refresh_token is not None

    await provider.revoke_token(access)

    assert await provider.load_access_token(tokens.access_token) is None
    assert await provider.load_refresh_token(client(), tokens.refresh_token) is None


async def test_other_grants_survive_a_revocation(provider: ChefOAuthProvider) -> None:
    mine = await tokens_for(provider, client())
    other = await tokens_for(provider, client())
    access = await provider.load_access_token(mine.access_token)
    assert access is not None

    await provider.revoke_token(access)

    assert await provider.load_access_token(other.access_token) is not None
