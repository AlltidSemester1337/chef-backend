from fastapi.testclient import TestClient

from chef_backend.auth import AuthenticatedUser, Claims, get_current_user, get_token_verifier
from chef_backend.main import create_app


class UnusedVerifier:
    """Fails the test if called: requests without tokens must be rejected before verification."""

    def verify_id_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")

    def verify_app_check_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")


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
    }


def test_me_requires_authentication() -> None:
    app = create_app()
    app.dependency_overrides[get_token_verifier] = UnusedVerifier

    response = TestClient(app).get("/v1/me")

    assert response.status_code == 401
