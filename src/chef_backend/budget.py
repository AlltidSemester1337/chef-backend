"""Daily per-user usage budget for calls the beta quota does not count.

Background AI purposes and text-to-speech need an eligible user (see quota.py)
but do not consume the lifetime quota. This budget bounds them per user and
Europe/Stockholm day. Usage lives in its own collection, usage_daily, which
clients cannot read or write (no Firestore rule matches it).
"""

from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from google.cloud import firestore

from chef_backend.auth import AuthenticatedUser

USAGE_COLLECTION = "usage_daily"
STOCKHOLM = ZoneInfo("Europe/Stockholm")
# Firestore's TTL policy on expireAt deletes old usage documents.
RETENTION = timedelta(days=30)


class UsageKind(StrEnum):
    AI_UNCOUNTED_CALLS = "ai_uncounted_calls"
    TTS_CHARS = "tts_chars"
    IMAGES = "images"


class BudgetDecision(StrEnum):
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"


class BudgetStore(Protocol):
    def add_and_get(self, uid: str, day: date, kind: UsageKind, amount: int) -> int:
        """Atomically add amount to the user's usage for the day; return the new total."""
        ...


def stockholm_today() -> date:
    return datetime.now(STOCKHOLM).date()


def check_budget(
    user: AuthenticatedUser,
    store: BudgetStore,
    *,
    kind: UsageKind,
    amount: int,
    limit: int,
    day: date,
) -> BudgetDecision:
    """Record the usage and decide whether it fits in today's budget.

    Call after check_quota has accepted the user. Like the quota, usage is recorded
    before the upstream call, so failed (or deliberately failing) calls count too.
    """
    if user.is_admin:
        return BudgetDecision.ALLOWED
    total = store.add_and_get(user.uid, day, kind, amount)
    return BudgetDecision.RATE_LIMITED if total > limit else BudgetDecision.ALLOWED


def usage_doc_id(uid: str, day: date) -> str:
    return f"{uid}_{day.isoformat()}"


class FirestoreBudgetStore:
    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def add_and_get(self, uid: str, day: date, kind: UsageKind, amount: int) -> int:
        doc_ref = self._client.collection(USAGE_COLLECTION).document(usage_doc_id(uid, day))
        return _add(self._client.transaction(), doc_ref, uid, day, kind, amount)


@firestore.transactional  # pyright: ignore[reportUnknownMemberType]
def _add(
    transaction: firestore.Transaction,
    doc_ref: Any,
    uid: str,
    day: date,
    kind: UsageKind,
    amount: int,
) -> int:
    snapshot = doc_ref.get(transaction=transaction)
    data: dict[str, Any] = snapshot.to_dict() or {}
    total = int(data.get(kind.value, 0)) + amount
    expire_at = datetime.combine(day, time(), tzinfo=STOCKHOLM) + RETENTION
    transaction.set(  # pyright: ignore[reportUnknownMemberType]
        doc_ref,
        {
            "uid": uid,
            "day": day.isoformat(),
            kind.value: total,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "expireAt": expire_at,
        },
        merge=True,
    )
    return total
