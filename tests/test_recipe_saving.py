"""RecipeSaver and RecipeDraft: validation, the saved document, budgets and the image."""

import io
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pytest
from mcp_support import ME, FakeRecipeRepository
from PIL import Image
from pydantic import ValidationError

from chef_backend.ai.client import UpstreamError
from chef_backend.budget import UsageKind
from chef_backend.recipes.models import Recipe, RecipeVariant
from chef_backend.recipes.saving import (
    DailySaveLimitReached,
    ImageOutcome,
    NotRecipeOwner,
    RecipeDraft,
    RecipeNotFound,
    RecipeSaver,
    SaveLimits,
    VariantDraft,
    image_prompt,
    recipe_document,
    variant_document,
)
from chef_backend.recipes.service import RecipeService

UID = "cook-1"
TODAY = date(2026, 11, 1)
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def draft(**overrides: Any) -> RecipeDraft:
    values: dict[str, Any] = {
        "title": "Chana masala",
        "summary": "Spicy chickpea stew.",
        "servings": "4",
        "prep_time": "10 minutes",
        "cooking_time": "20 minutes",
        "difficulty": "EASY",
        "tags": ["Vegetarian", "indian", "vegetarian"],
        "ingredients": [{"name": "chickpeas", "quantity": "800", "unit": "g"}, {"name": "salt"}],
        "instructions": ["Fry the onion.", "Simmer."],
    }
    return RecipeDraft.model_validate(values | overrides)


def png() -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (4, 4), (200, 100, 50, 255)).save(output, format="PNG")
    return output.getvalue()


@dataclass
class FakeStore:
    created: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    images: dict[str, str] = field(default_factory=dict[str, str])
    variants: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list[tuple[str, dict[str, Any]]]
    )

    def create(self, uid: str, draft: RecipeDraft, now: datetime) -> Recipe:
        document = recipe_document(f"new-{len(self.created) + 1}", uid, draft, now)
        self.created.append(document)
        return Recipe.model_validate(document)

    def set_image_url(self, recipe_id: str, image_url: str) -> None:
        self.images[recipe_id] = image_url

    def create_variant(self, recipe_id: str, draft: VariantDraft, now: datetime) -> RecipeVariant:
        document = variant_document(f"variant-{len(self.variants) + 1}", draft, now)
        self.variants.append((recipe_id, document))
        return RecipeVariant.model_validate(document)


@dataclass
class FakeBudget:
    usage: dict[tuple[str, date, UsageKind], int] = field(
        default_factory=dict[tuple[str, date, UsageKind], int]
    )

    def add_and_get(self, uid: str, day: date, kind: UsageKind, amount: int) -> int:
        key = (uid, day, kind)
        self.usage[key] = self.usage.get(key, 0) + amount
        return self.usage[key]


@dataclass
class FakeGenerator:
    error: Exception | None = None
    prompts: list[str] = field(default_factory=list[str])

    def generate(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return png()


@dataclass
class FakeImageStore:
    uploads: list[bytes] = field(default_factory=list[bytes])

    def upload_jpeg(self, data: bytes) -> str:
        self.uploads.append(data)
        return f"https://images.example/recipes/{len(self.uploads)}.jpg"


@dataclass
class World:
    store: FakeStore = field(default_factory=FakeStore)
    repository: FakeRecipeRepository = field(default_factory=FakeRecipeRepository)
    budget: FakeBudget = field(default_factory=FakeBudget)
    generator: FakeGenerator = field(default_factory=FakeGenerator)
    images: FakeImageStore = field(default_factory=FakeImageStore)
    limits: SaveLimits = field(default_factory=lambda: SaveLimits(daily_saves=2, daily_images=1))

    def saver(self) -> RecipeSaver:
        return RecipeSaver(
            self.store,
            RecipeService(self.repository),
            self.budget,
            lambda: self.generator,
            lambda: self.images,
            self.limits,
            now=lambda: NOW,
        )


@pytest.fixture
def world() -> World:
    return World()


# --- The draft ---------------------------------------------------------------------


def test_tags_are_stored_lowercase_and_once() -> None:
    assert draft().tags == ["vegetarian", "indian"]


INVALID: dict[str, dict[str, Any]] = {
    "empty title": {"title": ""},
    "title too long": {"title": "x" * 201},
    "no ingredients": {"ingredients": []},
    "no instructions": {"instructions": []},
    "unknown difficulty": {"difficulty": "IMPOSSIBLE"},
    "too many tags": {"tags": [f"t{n}" for n in range(31)]},
    "ingredient without name": {"ingredients": [{"quantity": "1"}]},
    "unknown field": {"uid": "someone-else"},
    "summary too long": {"summary": "x" * 2001},
}


@pytest.mark.parametrize("overrides", INVALID.values(), ids=INVALID.keys())
def test_invalid_drafts_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        draft(**overrides)


# --- The saved document -------------------------------------------------------------


def test_the_document_has_the_apps_shape_and_is_a_shared_favourite(world: World) -> None:
    world.saver().save(UID, draft(), TODAY)

    [document] = world.store.created
    assert document == {
        "id": "new-1",
        "uid": UID,
        "title": "Chana masala",
        "summary": "Spicy chickpea stew.",
        "servings": "4",
        "prepTime": "10 minutes",
        "cookingTime": "20 minutes",
        "difficulty": "EASY",
        "tags": ["vegetarian", "indian"],
        "ingredients": [
            {"name": "chickpeas", "quantity": "800", "unit": "g"},
            {"name": "salt", "quantity": "", "unit": ""},
        ],
        "instructions": ["Fry the onion.", "Simmer."],
        "tipsAndTricks": "",
        "nutrientsPerServing": [],
        "isFavourite": True,
        "updatedAt": NOW,
    }


def test_the_saved_recipe_gets_an_image(world: World) -> None:
    saved = world.saver().save(UID, draft(), TODAY)

    assert saved.image is ImageOutcome.ADDED
    assert saved.recipe.image_url == "https://images.example/recipes/1.jpg"
    assert world.store.images == {"new-1": "https://images.example/recipes/1.jpg"}
    assert world.images.uploads[0][:2] == b"\xff\xd8"  # re-encoded as JPEG
    assert "Chana masala" in world.generator.prompts[0]


def test_the_image_prompt_names_the_dish_and_its_ingredients() -> None:
    prompt = image_prompt(draft())

    assert "Chana masala" in prompt and "chickpeas, salt" in prompt


# --- Budgets -------------------------------------------------------------------------


def test_saves_beyond_the_daily_limit_write_nothing(world: World) -> None:
    saver = world.saver()
    saver.save(UID, draft(), TODAY)
    saver.save(UID, draft(), TODAY)

    with pytest.raises(DailySaveLimitReached):
        saver.save(UID, draft(), TODAY)
    assert len(world.store.created) == 2


def test_without_image_budget_the_recipe_is_saved_without_an_image(world: World) -> None:
    saver = world.saver()
    saver.save(UID, draft(), TODAY)

    second = saver.save(UID, draft(), TODAY)

    assert second.image is ImageOutcome.BUDGET_EXCEEDED
    assert second.recipe.image_url is None
    assert len(world.generator.prompts) == 1
    assert world.budget.usage[(UID, TODAY, UsageKind.IMAGES)] == 2


def test_saves_and_images_share_the_apps_budget_document(world: World) -> None:
    world.saver().save(UID, draft(), TODAY)

    assert world.budget.usage == {
        (UID, TODAY, UsageKind.RECIPE_SAVES): 1,
        (UID, TODAY, UsageKind.IMAGES): 1,
    }


# --- A failing image ------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [UpstreamError("image request failed: HTTP 500")], ids=["provider error"]
)
def test_a_failing_image_leaves_the_recipe_saved_without_one(
    world: World, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    world.generator.error = error

    saved = world.saver().save(UID, draft(summary="PRIVATE notes"), TODAY)

    assert saved.image is ImageOutcome.FAILED
    assert saved.recipe.image_url is None and world.store.images == {}
    assert len(world.store.created) == 1
    assert "new-1" in caplog.text and "PRIVATE" not in caplog.text


# --- Variants ----------------------------------------------------------------------


def variant(**overrides: Any) -> VariantDraft:
    values: dict[str, Any] = {
        "label": "For two",
        "title": "Chana masala for two",
        "servings": "2",
        "ingredients": [{"name": "chickpeas", "quantity": "400", "unit": "g"}],
        "instructions": ["Fry the onion.", "Simmer."],
    }
    return VariantDraft.model_validate(values | overrides)


def test_a_variant_is_added_to_the_users_own_recipe(world: World) -> None:
    saved = world.saver().save_variant(ME, "mine", variant(), TODAY)

    [(recipe_id, document)] = world.store.variants
    assert recipe_id == "mine"
    assert document == {
        "id": saved.id,
        "label": "For two",
        "createdAt": NOW,
        "isPinned": False,
        "title": "Chana masala for two",
        "summary": "",
        "servings": "2",
        "prepTime": "",
        "cookingTime": "",
        "difficulty": None,
        "ingredients": [{"name": "chickpeas", "quantity": "400", "unit": "g"}],
        "instructions": ["Fry the onion.", "Simmer."],
        "tipsAndTricks": "",
        "nutrientsPerServing": [],
    }
    assert (saved.label, saved.is_pinned) == ("For two", False)


@pytest.mark.parametrize(
    ("recipe_id", "error"),
    [
        ("theirs-shared", NotRecipeOwner),
        ("theirs-private", RecipeNotFound),
        ("missing", RecipeNotFound),
        ("mine/variants/x", RecipeNotFound),
    ],
    ids=["someone else's shared", "someone else's private", "missing", "a path"],
)
def test_only_the_owner_can_add_a_variant(
    world: World, recipe_id: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        world.saver().save_variant(ME, recipe_id, variant(), TODAY)

    assert world.store.variants == []
    assert world.budget.usage == {}


def test_variants_and_recipes_share_the_daily_save_budget(world: World) -> None:
    saver = world.saver()
    saver.save(ME, draft(), TODAY)
    saver.save_variant(ME, "mine", variant(), TODAY)

    with pytest.raises(DailySaveLimitReached):
        saver.save_variant(ME, "mine", variant(), TODAY)
    assert len(world.store.variants) == 1


@pytest.mark.parametrize(
    "overrides",
    [{"label": ""}, {"label": "x" * 101}, {"title": ""}, {"ingredients": []}, {"tags": ["x"]}],
    ids=["empty label", "long label", "empty title", "no ingredients", "tags"],
)
def test_invalid_variant_drafts_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        variant(**overrides)
