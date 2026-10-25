from dataclasses import dataclass, field, replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from chef_backend import budget
from chef_backend.auth import AuthenticatedUser
from chef_backend.budget import BudgetDecision, UsageKind, check_budget, stockholm_today

DAY = date(2026, 10, 18)
VERIFIED = AuthenticatedUser(uid="u1", email="cook@example.com", email_verified=True)


@dataclass
class FakeBudgetStore:
    usage: dict[tuple[str, date, UsageKind], int] = field(
        default_factory=dict[tuple[str, date, UsageKind], int]
    )
    calls: int = 0

    def add_and_get(self, uid: str, day: date, kind: UsageKind, amount: int) -> int:
        self.calls += 1
        key = (uid, day, kind)
        self.usage[key] = self.usage.get(key, 0) + amount
        return self.usage[key]


def check(
    store: FakeBudgetStore,
    *,
    amount: int = 1,
    limit: int = 3,
    day: date = DAY,
    kind: UsageKind = UsageKind.AI_UNCOUNTED_CALLS,
    user: AuthenticatedUser = VERIFIED,
) -> BudgetDecision:
    return check_budget(user, store, kind=kind, amount=amount, limit=limit, day=day)


def test_calls_up_to_the_limit_are_allowed_then_rate_limited() -> None:
    store = FakeBudgetStore()

    decisions = [check(store) for _ in range(4)]

    assert decisions == [BudgetDecision.ALLOWED] * 3 + [BudgetDecision.RATE_LIMITED]


def test_amount_is_added_so_characters_can_be_budgeted() -> None:
    store = FakeBudgetStore()
    kind = UsageKind.TTS_CHARS

    assert check(store, kind=kind, amount=60, limit=100) == BudgetDecision.ALLOWED
    assert check(store, kind=kind, amount=40, limit=100) == BudgetDecision.ALLOWED
    assert check(store, kind=kind, amount=1, limit=100) == BudgetDecision.RATE_LIMITED


def test_kinds_have_separate_budgets() -> None:
    store = FakeBudgetStore()
    for _ in range(3):
        check(store)

    assert check(store, kind=UsageKind.TTS_CHARS, amount=1) == BudgetDecision.ALLOWED


def test_a_new_day_starts_a_new_budget() -> None:
    store = FakeBudgetStore()
    for _ in range(4):
        check(store)

    assert check(store, day=date(2026, 10, 19)) == BudgetDecision.ALLOWED


def test_users_have_separate_budgets() -> None:
    store = FakeBudgetStore()
    for _ in range(4):
        check(store)

    assert check(store, user=replace(VERIFIED, uid="u2")) == BudgetDecision.ALLOWED


def test_admins_are_exempt_and_not_recorded() -> None:
    store = FakeBudgetStore()

    decisions = [check(store, user=replace(VERIFIED, is_admin=True)) for _ in range(10)]

    assert set(decisions) == {BudgetDecision.ALLOWED}
    assert store.calls == 0


@pytest.mark.parametrize(
    ("utc_time", "expected"),
    [
        # 22:30 UTC is 00:30 in Stockholm (CEST, UTC+2): already the next day there.
        (datetime(2026, 10, 18, 22, 30, tzinfo=ZoneInfo("UTC")), date(2026, 10, 19)),
        (datetime(2026, 10, 18, 21, 30, tzinfo=ZoneInfo("UTC")), date(2026, 10, 18)),
    ],
)
def test_the_budget_day_follows_stockholm_time(
    monkeypatch: pytest.MonkeyPatch, utc_time: datetime, expected: date
) -> None:
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> "FrozenDatetime":  # type: ignore[override]
            converted = utc_time.astimezone(budget.STOCKHOLM)
            return cls.fromtimestamp(converted.timestamp(), tz=budget.STOCKHOLM)

    monkeypatch.setattr(budget, "datetime", FrozenDatetime)

    assert stockholm_today() == expected
