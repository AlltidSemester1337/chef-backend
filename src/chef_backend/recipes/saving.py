# The Firestore client is only partially typed; results are annotated where they land.
# pyright: reportUnknownMemberType=false
"""Saving a new recipe for a user, the way the app does, plus an image.

The Admin SDK bypasses the security rules, so the draft is validated here with the
same limits as the rules (firestore.rules, recipeFieldsValid), and the document has
exactly the shape the apps write. As in the app, a saved recipe is a favourite:
it shows up among the user's recipes and is shared with the community.

The image comes after the save, with the same model, storage and daily budget as
the app's image endpoint. If it fails, the recipe stays saved without an image,
which the apps already handle.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, Protocol, cast

from google.api_core.exceptions import GoogleAPIError
from google.cloud import firestore
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chef_backend.ai.client import UpstreamError
from chef_backend.ai.dependencies import get_budget_store, get_firestore_client
from chef_backend.auth import AuthenticatedUser
from chef_backend.budget import BudgetDecision, BudgetStore, UsageKind, check_budget
from chef_backend.config import get_settings
from chef_backend.images import (
    IMAGE_PROMPT_TEMPLATE,
    MAX_RECIPE_CHARS,
    ImageGenerator,
    ImageStore,
    get_image_generator,
    get_image_store,
    to_jpeg,
)
from chef_backend.recipes.models import Difficulty, Recipe
from chef_backend.recipes.repository import RECIPES_COLLECTION

logger = logging.getLogger(__name__)

ShortText = Annotated[str, Field(max_length=100)]
ItemText = Annotated[str, Field(max_length=200)]


class IngredientDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=200)]
    quantity: ItemText = ""
    unit: ItemText = ""


class RecipeDraft(BaseModel):
    """A new recipe as a client describes it. Limits follow the Firestore rules."""

    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, Field(min_length=1, max_length=200)]
    summary: Annotated[str, Field(max_length=2000)] = ""
    servings: ShortText = ""
    prep_time: ShortText = ""
    cooking_time: ShortText = ""
    difficulty: Difficulty | None = None
    tags: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=50)]], Field(max_length=30)
    ] = []
    ingredients: Annotated[list[IngredientDraft], Field(min_length=1, max_length=100)]
    instructions: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=2000)]],
        Field(min_length=1, max_length=100),
    ]
    tips_and_tricks: Annotated[str, Field(max_length=5000)] = ""
    nutrients_per_serving: Annotated[list[IngredientDraft], Field(max_length=50)] = []

    @field_validator("tags")
    @classmethod
    def lowercase_unique_tags(cls, tags: list[str]) -> list[str]:
        # The apps store tags in lowercase (firestore-schema.md).
        return list(dict.fromkeys(tag.strip().lower() for tag in tags if tag.strip()))


class ImageOutcome(StrEnum):
    ADDED = "added"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SavedRecipe:
    recipe: Recipe
    image: ImageOutcome


class DailySaveLimitReached(Exception):
    """The user has saved as many recipes today as the budget allows."""


class RecipeStore(Protocol):
    def create(self, uid: str, draft: RecipeDraft, now: datetime) -> Recipe:
        """Write a new recipe document for `uid`; return it as stored."""
        ...

    def set_image_url(self, recipe_id: str, image_url: str) -> None: ...


def recipe_document(recipe_id: str, uid: str, draft: RecipeDraft, now: datetime) -> dict[str, Any]:
    """The Firestore document, with exactly the fields and names the apps write."""

    def items(entries: list[IngredientDraft]) -> list[dict[str, str]]:
        return [entry.model_dump() for entry in entries]

    return {
        "id": recipe_id,
        "uid": uid,
        "title": draft.title,
        "summary": draft.summary,
        "servings": draft.servings,
        "prepTime": draft.prep_time,
        "cookingTime": draft.cooking_time,
        "difficulty": draft.difficulty,
        "tags": draft.tags,
        "ingredients": items(draft.ingredients),
        "instructions": draft.instructions,
        "tipsAndTricks": draft.tips_and_tricks,
        "nutrientsPerServing": items(draft.nutrients_per_serving),
        # As in the app: saved means favourite, which also shares it with the community.
        "isFavourite": True,
        "updatedAt": now,
    }


class FirestoreRecipeStore:
    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def create(self, uid: str, draft: RecipeDraft, now: datetime) -> Recipe:
        ref: Any = self._client.collection(RECIPES_COLLECTION).document()  # a new auto-id
        recipe_id = cast(str, ref.id)
        document = recipe_document(recipe_id, uid, draft, now)
        ref.create(document)  # fails rather than overwrite, should an id ever repeat
        return Recipe.model_validate(document)

    def set_image_url(self, recipe_id: str, image_url: str) -> None:
        self._client.collection(RECIPES_COLLECTION).document(recipe_id).update(
            {"imageUrl": image_url}
        )


def image_prompt(draft: RecipeDraft) -> str:
    ingredients = ", ".join(i.name for i in draft.ingredients)
    text = f"{draft.title}. {draft.summary} Ingredients: {ingredients}"
    return IMAGE_PROMPT_TEMPLATE + text[:MAX_RECIPE_CHARS]


@dataclass(frozen=True)
class SaveLimits:
    daily_saves: int
    daily_images: int


class RecipeSaver:
    def __init__(
        self,
        store: RecipeStore,
        budget: BudgetStore,
        image_generator: Callable[[], ImageGenerator],
        image_store: Callable[[], ImageStore],
        limits: SaveLimits,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._budget = budget
        self._image_generator = image_generator
        self._image_store = image_store
        self._limits = limits
        self._now = now

    def save(self, uid: str, draft: RecipeDraft, today: date) -> SavedRecipe:
        """Save the draft for `uid`, then try to add an image.

        Raises DailySaveLimitReached before writing anything when today's budget is
        used up. Sign-in already required a verified, non-anonymous account.
        """
        user = AuthenticatedUser(uid=uid, email=None, email_verified=True)
        saves = check_budget(
            user,
            self._budget,
            kind=UsageKind.RECIPE_SAVES,
            amount=1,
            limit=self._limits.daily_saves,
            day=today,
        )
        if saves is BudgetDecision.RATE_LIMITED:
            raise DailySaveLimitReached

        recipe = self._store.create(uid, draft, self._now())
        image, image_url = self._add_image(user, recipe, draft, today)
        if image_url is not None:
            recipe = recipe.model_copy(update={"image_url": image_url})
        return SavedRecipe(recipe=recipe, image=image)

    def _add_image(
        self, user: AuthenticatedUser, recipe: Recipe, draft: RecipeDraft, today: date
    ) -> tuple[ImageOutcome, str | None]:
        images = check_budget(
            user,
            self._budget,
            kind=UsageKind.IMAGES,
            amount=1,
            limit=self._limits.daily_images,
            day=today,
        )
        if images is BudgetDecision.RATE_LIMITED:
            return ImageOutcome.BUDGET_EXCEEDED, None
        try:
            jpeg = to_jpeg(self._image_generator().generate(image_prompt(draft)))
            image_url = self._image_store().upload_jpeg(jpeg)
            self._store.set_image_url(recipe.id, image_url)
        except (UpstreamError, GoogleAPIError):
            # The recipe text is user data and stays out of the log.
            logger.exception("Image for saved recipe %s failed", recipe.id)
            return ImageOutcome.FAILED, None
        return ImageOutcome.ADDED, image_url


@lru_cache
def get_recipe_saver() -> RecipeSaver:
    settings = get_settings()
    return RecipeSaver(
        FirestoreRecipeStore(get_firestore_client()),
        get_budget_store(),
        get_image_generator,
        get_image_store,
        SaveLimits(
            daily_saves=settings.daily_recipe_save_limit,
            daily_images=settings.daily_image_limit,
        ),
    )
