from dataclasses import replace

import pytest

from chef_backend.auth import AuthenticatedUser
from chef_backend.quota import QuotaDecision, check_quota

LIMIT = 3
VERIFIED = AuthenticatedUser(uid="u1", email="cook@example.com", email_verified=True)


class FakeQuotaStore:
    def __init__(self, count: int = 0) -> None:
        self.count = count
        self.increments = 0

    def increment_and_get(self, uid: str) -> int:
        self.increments += 1
        self.count += 1
        return self.count


def test_verified_user_within_limit_is_allowed_and_counted() -> None:
    store = FakeQuotaStore()

    decision = check_quota(VERIFIED, store, counted=True, limit=LIMIT)

    assert decision == QuotaDecision.ALLOWED
    assert store.increments == 1


def test_the_call_after_the_limit_is_blocked() -> None:
    store = FakeQuotaStore()

    decisions = [check_quota(VERIFIED, store, counted=True, limit=LIMIT) for _ in range(LIMIT + 1)]

    assert decisions == [QuotaDecision.ALLOWED] * LIMIT + [QuotaDecision.BLOCKED]


def test_uncounted_calls_do_not_consume_quota() -> None:
    store = FakeQuotaStore()

    decision = check_quota(VERIFIED, store, counted=False, limit=LIMIT)

    assert decision == QuotaDecision.ALLOWED
    assert store.increments == 0


CASES = {
    "anonymous": (replace(VERIFIED, email=None, email_verified=False, is_anonymous=True),
                  QuotaDecision.BLOCKED),
    "unverified email": (replace(VERIFIED, email_verified=False),
                         QuotaDecision.EMAIL_VERIFICATION_REQUIRED),
    "admin": (replace(VERIFIED, is_admin=True), QuotaDecision.ALLOWED),
    "unverified admin": (replace(VERIFIED, email_verified=False, is_admin=True),
                         QuotaDecision.ALLOWED),
}  # fmt: skip


@pytest.mark.parametrize("counted", [True, False], ids=["counted", "uncounted"])
@pytest.mark.parametrize(("user", "expected"), CASES.values(), ids=CASES.keys())
def test_rules_before_counting(
    user: AuthenticatedUser, expected: QuotaDecision, counted: bool
) -> None:
    # Over the limit already: proves these users are decided without the counter.
    store = FakeQuotaStore(count=LIMIT * 10)

    assert check_quota(user, store, counted=counted, limit=LIMIT) == expected
    assert store.increments == 0
