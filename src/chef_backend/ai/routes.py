"""HTTP routes for the AI proxy."""

import logging
from collections.abc import Mapping
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from chef_backend.ai.client import ChatCompletionClient, CompletionCall, UpstreamError
from chef_backend.ai.dependencies import (
    get_beta_interaction_limit,
    get_chat_client,
    get_purposes,
    get_quota_store,
)
from chef_backend.ai.models import CompletionRequest, CompletionResponse
from chef_backend.ai.purposes import Purpose, PurposeConfig
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.quota import QuotaDecision, QuotaStore, check_quota

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
) -> CompletionResponse:
    """Run one AI call for a named purpose; prompt and model are chosen server-side."""
    config = purposes[request.purpose]

    decision = check_quota(user, quota_store, counted=config.counts_toward_quota, limit=limit)
    # Exhaustive: pyright (strict) flags this match if QuotaDecision gains a member.
    match decision:
        case QuotaDecision.BLOCKED:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "quota_exceeded")
        case QuotaDecision.EMAIL_VERIFICATION_REQUIRED:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "email_verification_required")
        case QuotaDecision.ALLOWED:
            pass

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
