"""FirebaseTokenVerifier: maps Firebase SDK outcomes onto the TokenVerifier contract.

The SDK functions are monkeypatched, so no network or credentials are needed.
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from firebase_admin import app_check, auth

from chef_backend import auth as chef_auth
from chef_backend.auth import FirebaseTokenVerifier, InvalidTokenError
from chef_backend.config import Settings


@pytest.fixture
def verifier() -> Iterator[FirebaseTokenVerifier]:
    # A unique app name per test, since firebase_admin keeps a global app registry.
    firebase_verifier = FirebaseTokenVerifier("demo-project", app_name=f"test-{uuid.uuid4()}")
    yield firebase_verifier
    firebase_verifier.close()


def raise_(error: Exception) -> Any:
    def fail(*_args: object, **_kwargs: object) -> Any:
        raise error

    return fail


def test_id_token_claims_are_returned(
    verifier: FirebaseTokenVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "verify_id_token", lambda token, app: {"uid": "u1", "token": token})

    assert verifier.verify_id_token("t") == {"uid": "u1", "token": "t"}


@pytest.mark.parametrize(
    "error",
    [
        auth.InvalidIdTokenError("bad"),
        auth.ExpiredIdTokenError("expired", cause=None),
        ValueError("empty token"),
    ],
    ids=["invalid", "expired", "empty"],
)
def test_bad_id_tokens_become_invalid_token_error(
    verifier: FirebaseTokenVerifier, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(auth, "verify_id_token", raise_(error))

    with pytest.raises(InvalidTokenError):
        verifier.verify_id_token("t")


def test_certificate_fetch_failure_is_not_a_client_error(
    verifier: FirebaseTokenVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = auth.CertificateFetchError("network down", cause=None)
    monkeypatch.setattr(auth, "verify_id_token", raise_(error))

    with pytest.raises(auth.CertificateFetchError):
        verifier.verify_id_token("t")


def test_app_check_claims_are_returned(
    verifier: FirebaseTokenVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_check, "verify_token", lambda token, app: {"app_id": "a1"})

    assert verifier.verify_app_check_token("t") == {"app_id": "a1"}


def test_bad_app_check_token_becomes_invalid_token_error(
    verifier: FirebaseTokenVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_check, "verify_token", raise_(ValueError("expired")))

    with pytest.raises(InvalidTokenError):
        verifier.verify_app_check_token("t")


def test_production_verifier_requires_a_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chef_auth, "get_settings", lambda: Settings(gcp_project_id=None))
    chef_auth.get_token_verifier.cache_clear()

    with pytest.raises(RuntimeError, match="CHEF_GCP_PROJECT_ID"):
        chef_auth.get_token_verifier()

    chef_auth.get_token_verifier.cache_clear()
