"""Personalised cooking resources (websites, channels, blogs) for the home screen.

The client sends only its locale. The backend reads the user's preference summary
itself, asks the model, and returns a validated list: the model output is never
passed through as is. Only https links with a known type survive.
"""

import json
import logging
from collections.abc import Mapping
from datetime import date
from functools import lru_cache
from typing import Annotated, Any, Literal, Protocol, cast

from fastapi import APIRouter, Depends, HTTPException, status
from google.cloud import firestore
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator

from chef_backend.access import require_budget, require_quota
from chef_backend.ai.client import ChatCompletionClient, CompletionCall, UpstreamError
from chef_backend.ai.dependencies import (
    get_beta_interaction_limit,
    get_budget_store,
    get_chat_client,
    get_daily_ai_uncounted_call_limit,
    get_firestore_client,
    get_purposes,
    get_quota_store,
    get_today,
)
from chef_backend.ai.models import ChatMessage
from chef_backend.ai.purposes import Purpose, PurposeConfig
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.budget import BudgetStore, UsageKind, check_budget
from chef_backend.quota import USERS_COLLECTION, QuotaStore, check_quota

logger = logging.getLogger(__name__)

MAX_RESOURCES = 8
# BCP 47-ish: "sv", "sv-SE", "zh-Hant-TW". Kept strict because it goes into the prompt.
LOCALE_PATTERN = r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,3}$"


class CookingResourcesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locale: Annotated[str, Field(pattern=LOCALE_PATTERN, max_length=35)]


class CookingResource(BaseModel):
    """One validated item. Field names match the apps' existing model."""

    title: Annotated[str, Field(min_length=1, max_length=120)]
    url: Annotated[HttpUrl, Field(max_length=500)]
    type: Literal["website", "youtube", "blog", "instagram"]
    description: Annotated[str, Field(min_length=1, max_length=300)]

    @field_validator("url")
    @classmethod
    def https_only(cls, url: HttpUrl) -> HttpUrl:
        if url.scheme != "https":
            raise ValueError("only https links are allowed")
        return url


class CookingResourcesResponse(BaseModel):
    resources: list[CookingResource]


class PreferencesReader(Protocol):
    def load_summary(self, uid: str) -> str | None:
        """The user's preference summary, or None if there is none."""
        ...


class FirestorePreferencesReader:
    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def load_summary(self, uid: str) -> str | None:
        snapshot = self._client.collection(USERS_COLLECTION).document(uid).get()  # pyright: ignore[reportUnknownMemberType]
        data: dict[str, Any] = snapshot.to_dict() or {}
        preferences = data.get("preferences")
        if not isinstance(preferences, dict):
            return None
        summary = cast(dict[str, Any], preferences).get("summary")
        return summary if isinstance(summary, str) and summary.strip() else None


@lru_cache
def get_preferences_reader() -> PreferencesReader:
    return FirestorePreferencesReader(get_firestore_client())


def build_prompt(locale: str, preferences_summary: str | None) -> str:
    lines = [f"User locale: {locale}"]
    if preferences_summary:
        lines.append(f"User cooking preferences: {preferences_summary}")
    lines.append("Recommend 5-8 personalized cooking resources for this user.")
    return "\n".join(lines)


def parse_resources(content: str) -> list[CookingResource]:
    """Keep the valid items of the model's answer; raise UpstreamError if it is unusable."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise UpstreamError("cooking resources answer is not JSON") from error
    # The prompt asks for {"resources": [...]}; tolerate a bare list as well.
    if isinstance(payload, dict):
        payload = cast(dict[str, Any], payload).get("resources")
    if not isinstance(payload, list):
        raise UpstreamError("cooking resources answer has no list")
    items = cast(list[Any], payload)

    resources: list[CookingResource] = []
    for item in items:
        try:
            resources.append(CookingResource.model_validate(item))
        except ValidationError:
            continue
    if dropped := len(items) - len(resources):
        logger.warning("Dropped %d invalid cooking resources", dropped)
    return resources[:MAX_RESOURCES]


router = APIRouter(prefix="/v1", tags=["home"])


@router.post("/cooking-resources")
def cooking_resources(
    request: CookingResourcesRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    chat_client: Annotated[ChatCompletionClient, Depends(get_chat_client)],
    purposes: Annotated[Mapping[Purpose, PurposeConfig], Depends(get_purposes)],
    preferences: Annotated[PreferencesReader, Depends(get_preferences_reader)],
    quota_store: Annotated[QuotaStore, Depends(get_quota_store)],
    limit: Annotated[int, Depends(get_beta_interaction_limit)],
    budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
    daily_limit: Annotated[int, Depends(get_daily_ai_uncounted_call_limit)],
    today: Annotated[date, Depends(get_today)],
) -> CookingResourcesResponse:
    """Recommend cooking resources for the caller. Uses the daily budget for AI calls."""
    require_quota(check_quota(user, quota_store, counted=False, limit=limit))
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

    prompt = build_prompt(request.locale, preferences.load_summary(user.uid))
    try:
        completion = chat_client.complete(
            CompletionCall(
                purpose=Purpose.COOKING_RESOURCES,
                config=purposes[Purpose.COOKING_RESOURCES],
                messages=[ChatMessage(role="user", content=prompt)],
                user_id=user.uid,
            )
        )
        resources = parse_resources(completion.content)
    except UpstreamError:
        logger.exception("Cooking resources generation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "upstream_error") from None

    return CookingResourcesResponse(resources=resources)
