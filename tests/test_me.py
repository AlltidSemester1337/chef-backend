from typing import Any

import pytest
from fastapi.testclient import TestClient

from chef_backend.auth import AuthenticatedUser, Claims, get_current_user, get_token_verifier
from chef_backend.main import create_app

AUTH_HEADERS = {"Authorization": "Bearer eyJ-id-token", "X-Firebase-AppCheck": "app-check"}


class UnusedVerifier:
    """Fails the test if called: requests without tokens must be rejected before verification."""

    def verify_id_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")

    def verify_app_check_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")


class ClaimsVerifier:
    """Accepts any token and returns the given ID token claims."""

    def __init__(self, id_claims: Claims) -> None:
        self.id_claims = id_claims

    def verify_id_token(self, token: str) -> Claims:
        return self.id_claims

    def verify_app_check_token(self, token: str) -> Claims:
        return {"app_id": "1:123:android:abc"}


def me_with_claims(claims: Claims) -> dict[str, Any]:
    app = create_app()
    app.dependency_overrides[get_token_verifier] = lambda: ClaimsVerifier(claims)
    response = TestClient(app).get("/v1/me", headers=AUTH_HEADERS)
    assert response.status_code == 200
    return response.json()


def test_me_returns_the_authenticated_user() -> None:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        uid="user-123", email="cook@example.com", email_verified=True
    )

    response = TestClient(app).get("/v1/me")

    assert response.status_code == 200
    assert response.json() == {
        "uid": "user-123",
        "email": "cook@example.com",
        "email_verified": True,
        "is_anonymous": False,
        "is_admin": False,
    }


def test_me_requires_authentication() -> None:
    app = create_app()
    app.dependency_overrides[get_token_verifier] = UnusedVerifier

    response = TestClient(app).get("/v1/me")

    assert response.status_code == 401


def test_anonymous_sign_in_is_recognised() -> None:
    me = me_with_claims({"uid": "anon", "firebase": {"sign_in_provider": "anonymous"}})

    assert me["is_anonymous"] is True


def test_regular_sign_in_is_not_anonymous() -> None:
    me = me_with_claims({"uid": "u1", "firebase": {"sign_in_provider": "google.com"}})

    assert me["is_anonymous"] is False


@pytest.mark.parametrize(
    ("admin_claim", "expected"),
    [(True, True), (False, False), ("true", False), (1, False), (None, False)],
)
def test_only_a_boolean_true_admin_claim_grants_admin(admin_claim: object, expected: bool) -> None:
    claims: Claims = {"uid": "u1"}
    if admin_claim is not None:
        claims["admin"] = admin_claim

    assert me_with_claims(claims)["is_admin"] is expected
