"""HTTP mapping of quota and budget decisions, shared by the AI and TTS routes."""

from fastapi import HTTPException, status

from chef_backend.budget import BudgetDecision
from chef_backend.quota import QuotaDecision


def require_quota(decision: QuotaDecision) -> None:
    # Exhaustive: pyright (strict) flags these matches if an enum gains a member.
    match decision:
        case QuotaDecision.BLOCKED:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "quota_exceeded")
        case QuotaDecision.EMAIL_VERIFICATION_REQUIRED:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "email_verification_required")
        case QuotaDecision.ALLOWED:
            pass


def require_budget(decision: BudgetDecision) -> None:
    match decision:
        case BudgetDecision.RATE_LIMITED:
            # Distinct from quota_exceeded in logs; clients treat every 429 the same.
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate_limited")
        case BudgetDecision.ALLOWED:
            pass
