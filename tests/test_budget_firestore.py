"""FirestoreBudgetStore against the Firestore emulator (see test_quota_firestore.py)."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pytest
from google.api_core.exceptions import Aborted
from google.cloud import firestore

from chef_backend.budget import (
    STOCKHOLM,
    USAGE_COLLECTION,
    FirestoreBudgetStore,
    UsageKind,
    usage_doc_id,
)

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ, reason="needs the Firestore emulator"
)

DAY = date(2026, 10, 18)


@pytest.fixture
def client() -> firestore.Client:
    return firestore.Client(project="demo-chef-backend")


def test_usage_accumulates_per_kind_in_one_document(client: firestore.Client) -> None:
    uid = f"user-{uuid.uuid4()}"
    store = FirestoreBudgetStore(client)

    assert store.add_and_get(uid, DAY, UsageKind.TTS_CHARS, 120) == 120
    assert store.add_and_get(uid, DAY, UsageKind.TTS_CHARS, 30) == 150
    assert store.add_and_get(uid, DAY, UsageKind.AI_UNCOUNTED_CALLS, 1) == 1

    doc = client.collection(USAGE_COLLECTION).document(usage_doc_id(uid, DAY)).get()
    data = doc.to_dict() or {}
    assert data["uid"] == uid
    assert data["day"] == "2026-10-18"
    assert (data["tts_chars"], data["ai_uncounted_calls"]) == (150, 1)


def test_document_expires_thirty_days_after_its_day(client: firestore.Client) -> None:
    uid = f"user-{uuid.uuid4()}"
    FirestoreBudgetStore(client).add_and_get(uid, DAY, UsageKind.AI_UNCOUNTED_CALLS, 1)

    data = client.collection(USAGE_COLLECTION).document(usage_doc_id(uid, DAY)).get().to_dict()
    expire_at: datetime = (data or {})["expireAt"]
    assert expire_at == datetime(2026, 10, 18, tzinfo=STOCKHOLM) + timedelta(days=30)


def test_each_day_has_its_own_document(client: firestore.Client) -> None:
    uid = f"user-{uuid.uuid4()}"
    store = FirestoreBudgetStore(client)
    store.add_and_get(uid, DAY, UsageKind.AI_UNCOUNTED_CALLS, 5)

    assert store.add_and_get(uid, DAY + timedelta(days=1), UsageKind.AI_UNCOUNTED_CALLS, 1) == 1


def test_concurrent_additions_are_never_lost_or_duplicated(client: firestore.Client) -> None:
    uid = f"user-{uuid.uuid4()}"
    store = FirestoreBudgetStore(client)

    def attempt(_: int) -> int | None:
        try:
            return store.add_and_get(uid, DAY, UsageKind.TTS_CHARS, 10)
        except (ValueError, Aborted):
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [r for r in pool.map(attempt, range(10)) if r is not None]

    assert results, "at least one addition should succeed"
    assert sorted(results) == [10 * n for n in range(1, len(results) + 1)]
    data = client.collection(USAGE_COLLECTION).document(usage_doc_id(uid, DAY)).get().to_dict()
    assert (data or {})["tts_chars"] == 10 * len(results)
