"""FirestoreOAuthStore against the Firestore emulator.

Skipped unless FIRESTORE_EMULATOR_HOST is set (see tests/test_quota_firestore.py).
Each test uses fresh random keys, so tests can share the emulator.
"""

import os
import secrets
import time
from datetime import UTC, datetime

import pytest
from google.cloud import firestore
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from chef_backend.mcp_server.store import (
    CLIENTS_COLLECTION,
    TOKENS_COLLECTION,
    FirestoreOAuthStore,
    PendingAuthorization,
    StoredCode,
    StoredToken,
    secret_hash,
)

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ, reason="needs the Firestore emulator"
)

CALLBACK = AnyUrl("https://claude.ai/api/mcp/auth_callback")
RESOURCE = "http://localhost:8080/mcp"


@pytest.fixture
def client() -> firestore.Client:
    return firestore.Client(project="demo-chef-backend")


@pytest.fixture
def store(client: firestore.Client) -> FirestoreOAuthStore:
    return FirestoreOAuthStore(client)


def key(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(16)}"


def later(seconds: float = 600) -> float:
    return time.time() + seconds


def token(grant_id: str, kind: str = "access") -> StoredToken:
    return StoredToken.model_validate(
        {
            "kind": kind,
            "client_id": "c1",
            "uid": "u1",
            "scopes": ["recipes:read"],
            "resource": RESOURCE,
            "grant_id": grant_id,
            "expires_at": int(later()),
        }
    )


def test_clients_round_trip_and_expire(
    client: firestore.Client, store: FirestoreOAuthStore
) -> None:
    info = OAuthClientInformationFull(
        client_id=key("client"), client_name="Claude", redirect_uris=[CALLBACK]
    )
    assert info.client_id is not None

    store.save_client(info, later())
    assert store.get_client(info.client_id) == info

    store.save_client(info, time.time() - 1)
    assert store.get_client(info.client_id) is None
    expire_at = client.collection(CLIENTS_COLLECTION).document(info.client_id).get().get("expireAt")
    assert expire_at < datetime.now(UTC)


def test_pending_logins_are_taken_once(store: FirestoreOAuthStore) -> None:
    request_id = key("chef_req")
    pending = PendingAuthorization(
        client_id="c1",
        redirect_uri=CALLBACK,
        redirect_uri_provided_explicitly=True,
        state="s",
        scopes=["recipes:read"],
        code_challenge="ch",
        resource=RESOURCE,
        expires_at=later(),
    )
    store.save_pending(request_id, pending)

    assert store.get_pending(request_id) == pending
    assert store.take_pending(request_id) == pending
    assert store.take_pending(request_id) is None
    assert store.get_pending(request_id) is None


def test_codes_are_taken_once(store: FirestoreOAuthStore) -> None:
    code = key("chef_code")
    stored = StoredCode(
        client_id="c1",
        uid="u1",
        scopes=["recipes:read"],
        code_challenge="ch",
        redirect_uri=CALLBACK,
        redirect_uri_provided_explicitly=True,
        resource=RESOURCE,
        expires_at=later(),
    )
    store.save_code(code, stored)

    assert store.get_code(code) == stored
    assert store.take_code(code) == stored
    assert store.take_code(code) is None


def test_tokens_are_stored_by_hash_only(
    client: firestore.Client, store: FirestoreOAuthStore
) -> None:
    secret = key("chef_at")
    store.save_token(secret, token("g1"))

    assert store.get_token(secret) == token("g1")
    assert client.collection(TOKENS_COLLECTION).document(secret).get().exists is False
    document = client.collection(TOKENS_COLLECTION).document(secret_hash(secret)).get()
    assert document.exists and secret not in str(document.to_dict())
    assert isinstance(document.get("expireAt"), datetime)


def test_a_refresh_token_is_marked_rotated_once(store: FirestoreOAuthStore) -> None:
    secret = key("chef_rt")
    store.save_token(secret, token("g2", kind="refresh"))

    assert store.mark_rotated(secret) is True
    assert store.mark_rotated(secret) is False
    rotated = store.get_token(secret)
    assert rotated is not None and rotated.rotated is True
    assert store.mark_rotated(key("chef_rt")) is False


def test_deleting_a_grant_removes_only_its_tokens(store: FirestoreOAuthStore) -> None:
    grant, other = key("grant"), key("grant")
    access, refresh, unrelated = key("chef_at"), key("chef_rt"), key("chef_at")
    store.save_token(access, token(grant))
    store.save_token(refresh, token(grant, kind="refresh"))
    store.save_token(unrelated, token(other))

    store.delete_grant(grant)

    assert store.get_token(access) is None and store.get_token(refresh) is None
    assert store.get_token(unrelated) is not None
