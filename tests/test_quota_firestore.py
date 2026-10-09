"""FirestoreQuotaStore against the Firestore emulator.

Skipped unless FIRESTORE_EMULATOR_HOST is set, e.g.:

    java -jar ~/.cache/firebase/emulators/cloud-firestore-emulator-*.jar \
        --host 127.0.0.1 --port 8085 &
    FIRESTORE_EMULATOR_HOST=127.0.0.1:8085 uv run pytest tests/test_quota_firestore.py
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from google.api_core.exceptions import Aborted
from google.cloud import firestore

from chef_backend.quota import INTERACTION_COUNT_FIELD, USERS_COLLECTION, FirestoreQuotaStore

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ, reason="needs the Firestore emulator"
)


@pytest.fixture
def client() -> firestore.Client:
    return firestore.Client(project="demo-chef-backend")


def test_first_interaction_creates_the_counter(client: firestore.Client) -> None:
    uid = f"user-{uuid.uuid4()}"

    assert FirestoreQuotaStore(client).increment_and_get(uid) == 1


def test_increment_keeps_other_user_fields(client: firestore.Client) -> None:
    uid = f"user-{uuid.uuid4()}"
    user_ref = client.collection(USERS_COLLECTION).document(uid)
    user_ref.set({"preferences": {"summary": "vegetarian"}, INTERACTION_COUNT_FIELD: 4})

    assert FirestoreQuotaStore(client).increment_and_get(uid) == 5

    data = user_ref.get().to_dict() or {}
    assert data["preferences"] == {"summary": "vegetarian"}
    assert data[INTERACTION_COUNT_FIELD] == 5


def test_concurrent_increments_are_never_lost_or_duplicated(client: firestore.Client) -> None:
    # Under heavy contention some transactions may give up (the caller then gets a 500
    # and no AI call is made). What must never happen is a lost or double-counted call.
    uid = f"user-{uuid.uuid4()}"
    store = FirestoreQuotaStore(client)

    def attempt(_: int) -> int | None:
        try:
            return store.increment_and_get(uid)
        except (ValueError, Aborted):
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [r for r in pool.map(attempt, range(10)) if r is not None]

    assert results, "at least one increment should succeed"
    assert sorted(results) == list(range(1, len(results) + 1))
    stored = client.collection(USERS_COLLECTION).document(uid).get().to_dict() or {}
    assert stored[INTERACTION_COUNT_FIELD] == len(results)
