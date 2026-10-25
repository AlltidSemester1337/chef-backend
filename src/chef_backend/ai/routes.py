"""HTTP routes for the AI proxy."""

import logging
from collections.abc import Mapping
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from chef_backend.access import require_budget, require_quota
from chef_backend.ai.client import ChatCompletionClient, CompletionCall, UpstreamError
from chef_backend.ai.dependencies import (
    get_beta_interaction_limit,
    get_budget_store,
    get_chat_client,
    get_daily_ai_uncounted_call_limit,
    get_purposes,
    get_quota_store,
    get_today,
)
from chef_backend.ai.models import CompletionRequest, CompletionResponse
from chef_backend.ai.purposes import Purpose, PurposeConfig
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.budget import BudgetStore, UsageKind, check_budget
from chef_backend.quota import QuotaStore, check_quota

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ai", tags=["ai"])


@router.post("/complete")
def complete(
    request: CompletionRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    chat_client: Annotated[ChatCompletionClient, Depends(get_chat_client)],
    purposes: Annotated[Mapping[Purpose, PurposeConfig], Depends(get_purposes)],
    quota_store: Annotated[QuotaStore, Depends(get_quota_store)],
    limit: Annotated[int, Depends(get_beta_interaction_limit)],
    budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
    daily_limit: Annotated[int, Depends(get_daily_ai_uncounted_call_limit)],
    today: Annotated[date, Depends(get_today)],
) -> CompletionResponse:
    """Run one AI call for a named purpose; prompt and model are chosen server-side."""
    config = purposes[request.purpose]

    counted = config.counts_toward_quota
    require_quota(check_quota(user, quota_store, counted=counted, limit=limit))
    if not counted:
        # The lifetime quota bounds counted purposes; the daily budget bounds the rest.
        require_budget(
            check_budget(
                user,
                budget_store,
                kind=UsageKind.AI_UNCOUNTED_CALLS,
                amount=1,
                limit=daily_limit,
                day=today,
            )
        )

    try:
        completion = chat_client.complete(
            CompletionCall(
                purpose=request.purpose,
                config=config,
                messages=request.messages,
                user_id=user.uid,
            )
        )
    except UpstreamError:
        # Log the purpose only; message content is user data and stays out of logs.
        logger.exception("AI provider call failed (purpose=%s)", request.purpose.value)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "upstream_error") from None

    return CompletionResponse(content=completion.content)
