"""Open-beta interaction quota, enforced on the server.

Same rules as the Android app's BetaQuotaPolicy: anonymous users are blocked,
admins are unlimited, unverified emails must verify first, everyone else gets
a fixed number of counted interactions.
"""

from enum import StrEnum
from typing import Any, Protocol

from google.cloud import firestore

from chef_backend.auth import AuthenticatedUser

USERS_COLLECTION = "users"
INTERACTION_COUNT_FIELD = "betaInteractionCount"


class QuotaDecision(StrEnum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    EMAIL_VERIFICATION_REQUIRED = "email_verification_required"


class QuotaStore(Protocol):
    def increment_and_get(self, uid: str) -> int:
        """Atomically add one to the user's interaction count and return the new value."""
        ...


def check_quota(
    user: AuthenticatedUser, store: QuotaStore, *, counted: bool, limit: int
) -> QuotaDecision:
    """Decide whether the user may make an AI call, recording it if it counts.

    Uncounted calls still require an eligible user, so background purposes cannot
    be used to bypass the rules; they just don't consume the quota.
    """
    if user.is_anonymous:
        return QuotaDecision.BLOCKED
    if user.is_admin:
        return QuotaDecision.ALLOWED
    if not user.email_verified:
        return QuotaDecision.EMAIL_VERIFICATION_REQUIRED
    if not counted:
        return QuotaDecision.ALLOWED
    # Like the app: the count goes up before the check, so call number limit + 1 is blocked.
    count = store.increment_and_get(user.uid)
    return QuotaDecision.BLOCKED if count > limit else QuotaDecision.ALLOWED


class FirestoreQuotaStore:
    """Stores the count on users/{uid}, the same field the Android app uses."""

    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def increment_and_get(self, uid: str) -> int:
        user_ref = self._client.collection(USERS_COLLECTION).document(uid)
        return _increment(self._client.transaction(), user_ref)


@firestore.transactional  # pyright: ignore[reportUnknownMemberType]
def _increment(transaction: firestore.Transaction, user_ref: Any) -> int:
    # A transaction (not firestore.Increment) so the new value is read atomically.
    # merge=True keeps the user's other fields (preferences, cooking resources).
    snapshot = user_ref.get(transaction=transaction)
    data: dict[str, Any] = snapshot.to_dict() or {}
    new_count = int(data.get(INTERACTION_COUNT_FIELD, 0)) + 1
    transaction.set(user_ref, {INTERACTION_COUNT_FIELD: new_count}, merge=True)  # pyright: ignore[reportUnknownMemberType]
    return new_count
