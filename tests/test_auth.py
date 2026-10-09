"""Spec for the get_current_user dependency, using a fake TokenVerifier."""

from collections.abc import Iterator

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from chef_backend.auth import (
    AuthenticatedUser,
    Claims,
    InvalidTokenError,
    get_current_user,
    get_token_verifier,
)
from chef_backend.main import create_app

# Real Firebase ID tokens are JWTs and always start with "eyJ".
VALID_ID_TOKEN = "eyJ-valid-id-token"
VALID_APP_CHECK_TOKEN = "valid-app-check-token"
SENSITIVE_ERROR_TEXT = "signature mismatch for kid abc123"


class FakeVerifier:
    """Accepts exactly one ID token and one App Check token; records every call."""

    def __init__(self, id_claims: Claims | None = None) -> None:
        self.id_claims: Claims = id_claims or {
            "uid": "user-123",
            "email": "cook@example.com",
            "email_verified": True,
        }
        self.id_tokens_seen: list[str] = []
        self.app_check_tokens_seen: list[str] = []
        self.unexpected_error: Exception | None = None

    def verify_id_token(self, token: str) -> Claims:
        self.id_tokens_seen.append(token)
        if self.unexpected_error is not None:
            raise self.unexpected_error
        if token != VALID_ID_TOKEN:
            raise InvalidTokenError(SENSITIVE_ERROR_TEXT)
        return self.id_claims

    def verify_app_check_token(self, token: str) -> Claims:
        self.app_check_tokens_seen.append(token)
        if token != VALID_APP_CHECK_TOKEN:
            raise InvalidTokenError(SENSITIVE_ERROR_TEXT)
        return {"app_id": "1:123:android:abc"}


def build_app(verifier: FakeVerifier) -> FastAPI:
    app = create_app()

    @app.get("/protected")
    def protected(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, object]:  # noqa: B008
        return {"uid": user.uid, "email": user.email, "email_verified": user.email_verified}

    app.dependency_overrides[get_token_verifier] = lambda: verifier
    return app


@pytest.fixture
def verifier() -> FakeVerifier:
    return FakeVerifier()


@pytest.fixture
def client(verifier: FakeVerifier) -> Iterator[TestClient]:
    with TestClient(build_app(verifier), raise_server_exceptions=False) as test_client:
        yield test_client


def headers(
    authorization: str | None = f"Bearer {VALID_ID_TOKEN}",
    app_check: str | None = VALID_APP_CHECK_TOKEN,
) -> dict[str, str]:
    result: dict[str, str] = {}
    if authorization is not None:
        result["Authorization"] = authorization
    if app_check is not None:
        result["X-Firebase-AppCheck"] = app_check
    return result


# --- Happy path ---------------------------------------------------------------


def test_valid_tokens_return_the_user(client: TestClient) -> None:
    response = client.get("/protected", headers=headers())

    assert response.status_code == 200
    assert response.json() == {
        "uid": "user-123",
        "email": "cook@example.com",
        "email_verified": True,
    }


def test_verifier_receives_the_raw_tokens(client: TestClient, verifier: FakeVerifier) -> None:
    client.get("/protected", headers=headers())

    assert verifier.id_tokens_seen == [VALID_ID_TOKEN]
    assert verifier.app_check_tokens_seen == [VALID_APP_CHECK_TOKEN]


def test_bearer_scheme_is_case_insensitive(client: TestClient) -> None:
    # RFC 7235: the auth scheme name is case-insensitive.
    response = client.get("/protected", headers=headers(authorization=f"bearer {VALID_ID_TOKEN}"))

    assert response.status_code == 200


def test_user_without_email_gets_none_and_unverified() -> None:
    # Anonymous and phone sign-ins have no email claims at all.
    verifier = FakeVerifier(id_claims={"uid": "anon-1"})
    with TestClient(build_app(verifier)) as client:
        response = client.get("/protected", headers=headers())

    assert response.status_code == 200
    assert response.json() == {"uid": "anon-1", "email": None, "email_verified": False}


# --- Rejections: all 401, all indistinguishable --------------------------------

REJECTED_HEADERS = {
    "missing authorization": headers(authorization=None),
    "wrong scheme": headers(authorization=f"Basic {VALID_ID_TOKEN}"),
    "bearer without token": headers(authorization="Bearer "),
    "token without scheme": headers(authorization=VALID_ID_TOKEN),
    "invalid id token": headers(authorization="Bearer forged-token"),
    "missing app check": headers(app_check=None),
    "empty app check": headers(app_check=""),
    "invalid app check": headers(app_check="forged-app-check"),
    "nothing at all": {},
}


@pytest.mark.parametrize("request_headers", REJECTED_HEADERS.values(), ids=REJECTED_HEADERS.keys())
def test_rejected_requests_get_401(client: TestClient, request_headers: dict[str, str]) -> None:
    response = client.get("/protected", headers=request_headers)

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.parametrize("request_headers", REJECTED_HEADERS.values(), ids=REJECTED_HEADERS.keys())
def test_rejections_do_not_reveal_why(client: TestClient, request_headers: dict[str, str]) -> None:
    # Same body for every failure, so a caller cannot probe which check failed,
    # and verifier error messages never reach the client.
    baseline = client.get("/protected", headers={}).json()

    response = client.get("/protected", headers=request_headers)

    assert response.json() == baseline
    assert SENSITIVE_ERROR_TEXT not in response.text


def test_invalid_app_check_is_rejected_even_with_valid_id_token(
    client: TestClient, verifier: FakeVerifier
) -> None:
    response = client.get("/protected", headers=headers(app_check="forged-app-check"))

    assert response.status_code == 401
    assert verifier.app_check_tokens_seen == ["forged-app-check"]


def test_unexpected_verifier_failure_is_500_not_401(
    client: TestClient, verifier: FakeVerifier
) -> None:
    # E.g. Google's public keys could not be fetched. That is our outage, not a bad
    # token: a 401 would make clients sign the user out for nothing.
    verifier.unexpected_error = RuntimeError("certificate fetch failed")

    response = client.get("/protected", headers=headers())

    assert response.status_code == 500


# --- Open routes stay open -----------------------------------------------------


def test_healthz_needs_no_tokens_and_never_calls_the_verifier(
    client: TestClient, verifier: FakeVerifier
) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert verifier.id_tokens_seen == []
    assert verifier.app_check_tokens_seen == []
