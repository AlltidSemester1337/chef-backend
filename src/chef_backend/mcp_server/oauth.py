"""The OAuth 2.1 authorization server for /mcp, on top of Firebase sign-in.

The MCP SDK runs the protocol endpoints (/authorize, /token, /register, /revoke and
the metadata) and checks PKCE, redirect URIs and expiry; this provider decides who
may register, sends the user to the login page, and issues and checks our tokens.
Firebase ID tokens never reach /mcp: the spec requires tokens issued for this
server, so the login page trades the ID token for a one-time code (see login.py).

Store calls block on Firestore, so they run in a worker thread.
"""

import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar
from urllib.parse import urlencode, urlsplit

import anyio.to_thread
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, AnyUrl

from chef_backend.ai.dependencies import get_firestore_client
from chef_backend.config import get_settings
from chef_backend.mcp_server.store import (
    FirestoreOAuthStore,
    OAuthStore,
    PendingAuthorization,
    StoredCode,
    StoredToken,
    TokenKind,
)

LOGIN_PATH = "/oauth/login"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]"}

PENDING_TTL_SECONDS = 10 * 60
CODE_TTL_SECONDS = 5 * 60
ACCESS_TOKEN_TTL_SECONDS = 60 * 60
REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
# Unused registrations disappear; each sign-in and refresh extends a client's life.
CLIENT_TTL_SECONDS = 90 * 24 * 60 * 60

T = TypeVar("T")


def is_allowed_redirect_uri(uri: str, allowlist: Iterable[str]) -> bool:
    """Exact matches from the allowlist, plus http loopback on any port (RFC 8252).

    A loopback redirect only reaches a program on the user's own machine (Claude
    Code), so it cannot hand a code to someone else.
    """
    if uri in allowlist:
        return True
    parts = urlsplit(uri)
    return parts.scheme == "http" and parts.netloc.rsplit(":", 1)[0] in LOOPBACK_HOSTS


def new_secret(prefix: str) -> str:
    """256 random bits; the prefix makes a leaked token recognisable to secret scanners."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


@dataclass(frozen=True)
class OAuthConfig:
    # Exactly as in the server metadata (with the trailing slash pydantic gives a bare
    # origin): clients compare the `iss` of the redirect to it as a plain string.
    issuer_url: str
    resource_url: str
    scopes: list[str]
    redirect_uri_allowlist: list[str]


class ChefOAuthProvider:
    """OAuthAuthorizationServerProvider for the MCP SDK."""

    def __init__(
        self, store: OAuthStore, config: OAuthConfig, clock: Callable[[], float] = time.time
    ) -> None:
        self._store = store
        self._config = config
        self._clock = clock

    async def _run(self, fn: Callable[[], T]) -> T:
        return await anyio.to_thread.run_sync(fn)

    # --- clients -------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return await self._run(lambda: self._store.get_client(client_id))

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        for uri in client_info.redirect_uris or []:
            if not is_allowed_redirect_uri(str(uri), self._config.redirect_uri_allowlist):
                raise RegistrationError(
                    "invalid_redirect_uri", "redirect_uri is not allowed for this server"
                )
        if client_info.client_name and len(client_info.client_name) > 100:
            raise RegistrationError("invalid_client_metadata", "client_name is too long")
        expires_at = self._clock() + CLIENT_TTL_SECONDS
        await self._run(lambda: self._store.save_client(client_info, expires_at))

    async def _touch_client(self, client: OAuthClientInformationFull) -> None:
        expires_at = self._clock() + CLIENT_TTL_SECONDS
        await self._run(lambda: self._store.save_client(client, expires_at))

    # --- authorization -------------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        # RFC 8707: a token is only ever issued for this server. Clients must send the
        # resource; older ones that do not still get a token bound to it.
        resource = params.resource or self._config.resource_url
        if resource.rstrip("/") != self._config.resource_url:
            raise AuthorizeError("invalid_request", "unknown resource")
        if client.client_id is None:
            raise AuthorizeError("unauthorized_client", "client has no id")
        request_id = new_secret("chef_req")
        pending = PendingAuthorization(
            client_id=client.client_id,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            state=params.state,
            scopes=params.scopes or list(self._config.scopes),
            code_challenge=params.code_challenge,
            resource=self._config.resource_url,
            expires_at=self._clock() + PENDING_TTL_SECONDS,
        )
        await self._run(lambda: self._store.save_pending(request_id, pending))
        login_url = self._config.issuer_url.rstrip("/") + LOGIN_PATH
        return f"{login_url}?{urlencode({'request': request_id})}"

    def pending_login(
        self, request_id: str
    ) -> tuple[PendingAuthorization, OAuthClientInformationFull] | None:
        """The pending login and its client, for the login page (sync)."""
        pending = self._store.get_pending(request_id)
        if pending is None or pending.expires_at <= self._clock():
            return None
        client = self._store.get_client(pending.client_id)
        return (pending, client) if client is not None else None

    def complete_login(self, request_id: str, uid: str) -> str | None:
        """Turn a pending login into a one-time code; the client's redirect URL (sync).

        None if the login is unknown, expired or already completed. Called by the
        login callback after it has verified the user.
        """
        pending = self._store.take_pending(request_id)
        if pending is None or pending.expires_at <= self._clock():
            return None
        code = new_secret("chef_code")
        self._store.save_code(
            code,
            StoredCode(
                client_id=pending.client_id,
                uid=uid,
                scopes=pending.scopes,
                code_challenge=pending.code_challenge,
                redirect_uri=pending.redirect_uri,
                redirect_uri_provided_explicitly=pending.redirect_uri_provided_explicitly,
                resource=pending.resource,
                expires_at=self._clock() + CODE_TTL_SECONDS,
            ),
        )
        return _redirect(
            pending.redirect_uri, code=code, state=pending.state, iss=self._config.issuer_url
        )

    def deny_login(self, request_id: str) -> str | None:
        """The user declined: end the pending login, redirect with access_denied (sync)."""
        pending = self._store.take_pending(request_id)
        if pending is None or pending.expires_at <= self._clock():
            return None
        return _redirect(
            pending.redirect_uri,
            error="access_denied",
            state=pending.state,
            iss=self._config.issuer_url,
        )

    # --- authorization codes -------------------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        stored = await self._run(lambda: self._store.get_code(authorization_code))
        if stored is None or stored.client_id != client.client_id:
            return None
        return AuthorizationCode(
            code=authorization_code,
            scopes=stored.scopes,
            expires_at=stored.expires_at,
            client_id=stored.client_id,
            code_challenge=stored.code_challenge,
            redirect_uri=stored.redirect_uri,
            redirect_uri_provided_explicitly=stored.redirect_uri_provided_explicitly,
            resource=stored.resource,
            subject=stored.uid,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Taken, not read: of two concurrent exchanges of one code, only one wins.
        stored = await self._run(lambda: self._store.take_code(authorization_code.code))
        if stored is None or stored.client_id != client.client_id:
            raise TokenError("invalid_grant", "authorization code is invalid")
        await self._touch_client(client)
        return await self._issue(
            client_id=stored.client_id,
            uid=stored.uid,
            scopes=stored.scopes,
            grant_id=new_secret("grant"),
        )

    # --- refresh tokens ------------------------------------------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        stored = await self._run(lambda: self._store.get_token(refresh_token))
        if stored is None or stored.kind != "refresh" or stored.client_id != client.client_id:
            return None
        if stored.rotated:
            # Reuse of an exchanged refresh token: assume theft and end the whole grant.
            await self._run(lambda: self._store.delete_grant(stored.grant_id))
            return None
        if stored.expires_at <= self._clock():
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=stored.client_id,
            scopes=stored.scopes,
            expires_at=stored.expires_at,
            resource=stored.resource,
            subject=stored.uid,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        stored = await self._run(lambda: self._store.get_token(refresh_token.token))
        if stored is None or not await self._run(
            lambda: self._store.mark_rotated(refresh_token.token)
        ):
            # Lost a race with a concurrent exchange of the same token.
            raise TokenError("invalid_grant", "refresh token is invalid")
        await self._touch_client(client)
        return await self._issue(
            client_id=stored.client_id, uid=stored.uid, scopes=scopes, grant_id=stored.grant_id
        )

    # --- access tokens -------------------------------------------------------

    async def load_access_token(self, token: str) -> AccessToken | None:
        stored = await self._run(lambda: self._store.get_token(token))
        if stored is None or stored.kind != "access" or stored.expires_at <= self._clock():
            return None
        return AccessToken(
            token=token,
            client_id=stored.client_id,
            scopes=stored.scopes,
            expires_at=stored.expires_at,
            resource=stored.resource,
            subject=stored.uid,
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        stored = await self._run(lambda: self._store.get_token(token.token))
        if stored is not None:
            await self._run(lambda: self._store.delete_grant(stored.grant_id))

    # --- helpers -------------------------------------------------------------

    async def _issue(
        self, *, client_id: str, uid: str, scopes: list[str], grant_id: str
    ) -> OAuthToken:
        now = int(self._clock())
        access, refresh = new_secret("chef_at"), new_secret("chef_rt")

        def stored(kind: TokenKind, ttl: int) -> StoredToken:
            return StoredToken(
                kind=kind,
                client_id=client_id,
                uid=uid,
                scopes=scopes,
                resource=self._config.resource_url,
                grant_id=grant_id,
                expires_at=now + ttl,
            )

        def save() -> None:
            self._store.save_token(access, stored("access", ACCESS_TOKEN_TTL_SECONDS))
            self._store.save_token(refresh, stored("refresh", REFRESH_TOKEN_TTL_SECONDS))

        await self._run(save)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",  # noqa: S106 - the token type name, not a secret
            expires_in=ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=refresh,
            scope=" ".join(scopes),
        )


def _redirect(redirect_uri: AnyUrl, **params: str | None) -> str:
    from mcp.server.auth.provider import construct_redirect_uri

    return construct_redirect_uri(str(redirect_uri), **params)


@lru_cache
def get_oauth_provider() -> ChefOAuthProvider:
    from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE
    from chef_backend.mcp_server.server import MCP_PATH

    settings = get_settings()
    base_url = settings.public_base_url.rstrip("/")
    return ChefOAuthProvider(
        FirestoreOAuthStore(get_firestore_client()),
        OAuthConfig(
            issuer_url=str(AnyHttpUrl(base_url)),
            resource_url=base_url + MCP_PATH,
            scopes=[RECIPES_READ_SCOPE],
            redirect_uri_allowlist=list(settings.mcp_redirect_uri_allowlist),
        ),
    )
