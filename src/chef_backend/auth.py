"""Request authentication: Firebase ID token + Firebase App Check token.

Auth is a FastAPI dependency rather than middleware, so each route opts in
(``Depends(get_current_user)``) and open routes such as ``/health`` stay open.

Token verification sits behind the ``TokenVerifier`` protocol. Production uses
``FirebaseTokenVerifier``; tests override ``get_token_verifier`` with a fake via
``app.dependency_overrides``.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Protocol, cast

# firebase_admin ships without type stubs; its untyped calls are confined to FirebaseTokenVerifier.
import firebase_admin  # pyright: ignore[reportMissingTypeStubs]
from fastapi import Depends, Header, HTTPException, status
from firebase_admin import app_check, auth  # pyright: ignore[reportMissingTypeStubs]

from chef_backend.config import get_settings

Claims = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    uid: str
    email: str | None
    email_verified: bool


class InvalidTokenError(Exception):
    """The client sent a token that failed verification.

    This is a plain Python exception: FastAPI does not know about it, so
    get_current_user must translate it into an HTTP 401. Infrastructure failures
    (e.g. fetching Google's public keys) must not be raised as this error; they
    propagate and become a 5xx.
    """


class TokenVerifier(Protocol):
    def verify_id_token(self, token: str) -> Claims:
        """Return the decoded claims of a Firebase ID token, or raise InvalidTokenError."""
        ...

    def verify_app_check_token(self, token: str) -> Claims:
        """Return the decoded claims of an App Check token, or raise InvalidTokenError."""
        ...


class FirebaseTokenVerifier:
    """TokenVerifier backed by the Firebase Admin SDK.

    No credential is passed, so the SDK falls back to Application Default
    Credentials (the Cloud Run service account; locally ``gcloud auth
    application-default login``). Credentials are loaded lazily, and verifying
    ID tokens does not need them at all: only Google's public keys.
    """

    def __init__(self, project_id: str, app_name: str = "chef-backend") -> None:
        # The project ID is what both verifiers check the token audience against.
        # Passing it explicitly avoids depending on how ADC resolves it.
        self._app = firebase_admin.initialize_app(  # pyright: ignore[reportUnknownMemberType]
            options={"projectId": project_id}, name=app_name
        )

    def verify_id_token(self, token: str) -> Claims:
        try:
            claims = auth.verify_id_token(token, app=self._app)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        except (auth.InvalidIdTokenError, ValueError) as error:
            # InvalidIdTokenError covers expired and malformed tokens; ValueError an empty token.
            # CertificateFetchError is deliberately not caught: it is our failure, not the client's.
            raise InvalidTokenError("invalid ID token") from error
        return cast(Claims, claims)

    def verify_app_check_token(self, token: str) -> Claims:
        try:
            claims = app_check.verify_token(token, app=self._app)  # pyright: ignore[reportUnknownMemberType]
        except ValueError as error:
            # The SDK reports every invalid-token case as ValueError. A missing project ID
            # would also be a ValueError, which is why __init__ requires one.
            raise InvalidTokenError("invalid App Check token") from error
        return claims

    def close(self) -> None:
        firebase_admin.delete_app(self._app)  # pyright: ignore[reportUnknownMemberType]


@lru_cache
def get_token_verifier() -> TokenVerifier:
    """Create the production verifier on first use, so /health never touches Firebase."""
    project_id = get_settings().gcp_project_id
    if not project_id:
        # Fail closed with a 500 rather than letting every request through or 401.
        raise RuntimeError("CHEF_GCP_PROJECT_ID must be set to verify Firebase tokens")
    return FirebaseTokenVerifier(project_id)


def get_current_user(
    verifier: Annotated[TokenVerifier, Depends(get_token_verifier)],
    authorization: Annotated[str | None, Header()] = None,
    app_check_token: Annotated[str | None, Header(alias="X-Firebase-AppCheck")] = None,
) -> AuthenticatedUser:
    """FastAPI dependency: return the caller, or raise HTTPException(401).

    Every rejection gets the same response, so callers cannot probe which check
    failed. Keep this a plain ``def`` (not ``async def``): verification can block
    on a network call, and FastAPI runs sync dependencies in a thread pool.
    """
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"}
    )

    scheme, _, id_token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not id_token or not app_check_token:
        raise unauthorized

    try:
        # App Check first: it rejects traffic that does not come from our apps
        # before we spend anything on the user's token.
        verifier.verify_app_check_token(app_check_token)
        claims = verifier.verify_id_token(id_token)
    except InvalidTokenError:
        raise unauthorized from None

    # A verified ID token always has "uid"; email claims are absent for e.g. anonymous users.
    return AuthenticatedUser(
        uid=claims["uid"],
        email=claims.get("email"),
        email_verified=claims.get("email_verified", False),
    )
