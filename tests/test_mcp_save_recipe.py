"""Spec for the MCP tool save_recipe (chef_backend.mcp_server.write_tools).

save_recipe(title, ingredients, instructions, summary="", servings="", prep_time="",
cooking_time="", difficulty=None, tags=[], tips_and_tricks="",
nutrients_per_serving=[]) saves a new recipe for the user in the access token, via
RecipeSaver, and returns {"recipe": <the full recipe view>, "image": "added" |
"budget_exceeded" | "failed"}.

- ingredients and nutrients_per_serving are lists of {name, quantity, unit}.
- It needs the recipes:write scope (require_scope); without it, a tool error
  containing "insufficient_scope" and nothing is saved.
- Today's save budget used up: tool error "daily_save_limit_reached".
- Invalid input (empty title, no ingredients, ...): a tool error, nothing saved.
- Annotations: not read-only, not destructive, not idempotent (each call saves a
  new recipe), open world (it publishes to the community). The description says
  the recipe is shared with the community, so the model can tell the user first.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_support import BASE_URL, ME, OTHER, FakeMcpTokenVerifier, FakeRecipeRepository, McpClient
from test_recipe_saving import FakeBudget, FakeGenerator, FakeImageStore

from chef_backend.ai.dependencies import get_today
from chef_backend.main import create_app
from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE, get_mcp_token_verifier
from chef_backend.recipes.models import Recipe, RecipeVariant
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.saving import (
    RecipeDraft,
    RecipeSaver,
    SaveLimits,
    VariantDraft,
    get_recipe_saver,
    recipe_document,
    variant_document,
)
from chef_backend.recipes.service import RecipeService

TODAY = date(2026, 11, 1)

CHANA: dict[str, Any] = {
    "title": "Chana masala",
    "summary": "Spicy chickpea stew.",
    "servings": "4",
    "prep_time": "10 minutes",
    "cooking_time": "20 minutes",
    "difficulty": "EASY",
    "tags": ["vegetarian", "indian"],
    "ingredients": [
        {"name": "chickpeas", "quantity": "800", "unit": "g"},
        {"name": "salt"},
    ],
    "instructions": ["Fry the onion.", "Add the spices.", "Simmer."],
    "tips_and_tricks": "Toast the spices first.",
}


@dataclass
class RepositoryBackedStore:
    """Saves into the fake repository, so the read tools see what was saved."""

    repository: FakeRecipeRepository
    images: dict[str, str] = field(default_factory=dict[str, str])

    def create(self, uid: str, draft: RecipeDraft, now: datetime) -> Recipe:
        recipe_id = f"saved-{len(self.repository.recipes)}"
        recipe = Recipe.model_validate(recipe_document(recipe_id, uid, draft, now))
        self.repository.recipes.append(recipe)
        return recipe

    def set_image_url(self, recipe_id: str, image_url: str) -> None:
        self.images[recipe_id] = image_url

    def create_variant(self, recipe_id: str, draft: VariantDraft, now: datetime) -> RecipeVariant:
        saved = self.repository.saved_variants.setdefault(recipe_id, [])
        document = variant_document(f"variant-{len(saved) + 1}", draft, now)
        saved.append(RecipeVariant.model_validate(document))
        return saved[-1]


@dataclass
class World:
    http: TestClient
    verifier: FakeMcpTokenVerifier
    repository: FakeRecipeRepository
    store: RepositoryBackedStore
    budget: FakeBudget

    def client(self, **token: Any) -> McpClient:
        return McpClient(self.http, self.verifier.issue(f"t-{len(self.verifier.tokens)}", **token))

    def writer(self) -> McpClient:
        return self.client(scopes=[RECIPES_READ_SCOPE, "recipes:write"])


@pytest.fixture
def world() -> Iterator[World]:
    verifier = FakeMcpTokenVerifier()
    repository = FakeRecipeRepository()
    store = RepositoryBackedStore(repository)
    budget = FakeBudget()
    saver = RecipeSaver(
        store,
        RecipeService(repository),
        budget,
        FakeGenerator,
        FakeImageStore,
        SaveLimits(daily_saves=2, daily_images=5),
    )
    app = create_app()
    app.dependency_overrides.update(
        {
            get_mcp_token_verifier: lambda: verifier,
            get_recipe_service: lambda: RecipeService(repository),
            get_recipe_saver: lambda: saver,
            get_today: lambda: TODAY,
        }
    )
    with TestClient(app, base_url=BASE_URL) as http:
        yield World(http, verifier, repository, store, budget)


def save_tool(mcp: McpClient) -> dict[str, Any]:
    [tool] = [t for t in mcp.list_tools() if t["name"] == "save_recipe"]
    return tool


def error_text(result: dict[str, Any]) -> str:
    assert result["isError"] is True, result
    return " ".join(block["text"] for block in result["content"])


# --- What the model sees ----------------------------------------------------------------


def test_the_tool_is_an_open_world_non_idempotent_write(world: World) -> None:
    annotations = save_tool(world.writer()).get("annotations") or {}

    assert annotations.get("readOnlyHint") is False
    assert annotations.get("destructiveHint") is False
    assert annotations.get("idempotentHint") is False
    # It publishes to the community, i.e. to other users.
    assert annotations.get("openWorldHint") is True


def test_the_description_says_saved_recipes_are_shared(world: World) -> None:
    description = save_tool(world.writer()).get("description") or ""

    assert len(description) >= 80
    assert "shared" in description.lower() and "community" in description.lower()


def test_title_ingredients_and_instructions_are_required(world: World) -> None:
    schema = save_tool(world.writer())["inputSchema"]

    assert set(schema["required"]) == {"title", "ingredients", "instructions"}
    for name, prop in schema["properties"].items():
        assert prop.get("description"), f"{name} has no description"


# --- Saving ------------------------------------------------------------------------------


def test_saving_returns_the_full_recipe_with_its_image(world: World) -> None:
    result = world.writer().call("save_recipe", **CHANA)

    assert result["isError"] is False, result["content"]
    saved = result["structuredContent"]
    assert saved["image"] == "added"
    recipe = saved["recipe"]
    assert (recipe["title"], recipe["isMine"], recipe["isFavourite"]) == (
        "Chana masala",
        True,
        True,
    )
    assert "uid" not in recipe and ME not in str(result)
    assert recipe["imageUrl"] == "https://images.example/recipes/1.jpg"
    assert recipe["ingredients"][1] == {"name": "salt", "quantity": "", "unit": ""}
    assert recipe["tipsAndTricks"] == "Toast the spices first."


def test_the_saved_recipe_belongs_to_the_user_and_can_be_found(world: World) -> None:
    writer = world.writer()
    saved_id = writer.call("save_recipe", **CHANA)["structuredContent"]["recipe"]["id"]

    [stored] = [r for r in world.repository.recipes if r.id == saved_id]
    assert stored.uid == ME and ME not in str(writer.call("get_recipe", recipe_id=saved_id))
    found = writer.call("search_recipes", query="chana", scope="mine")["structuredContent"]
    assert [r["id"] for r in found["recipes"]] == [saved_id]


def test_a_long_real_world_title_and_summary_can_be_saved(world: World) -> None:
    long = CHANA | {
        "title": "Spicy Gochujang-Glazed Baked Chicken Thighs with Sesame Roasted Broccoli",
        "summary": "A weeknight favourite. " * 20,
        "tips_and_tricks": "Marinate overnight for more flavour. " * 20,
    }

    assert world.writer().call("save_recipe", **long)["isError"] is False


def test_missing_optional_facts_are_left_empty_not_invented(world: World) -> None:
    minimal = {k: CHANA[k] for k in ("title", "ingredients", "instructions")}

    recipe = world.writer().call("save_recipe", **minimal)["structuredContent"]["recipe"]

    assert (recipe["servings"], recipe["difficulty"], recipe["prepTime"]) == ("", None, "")


def test_an_invalid_recipe_says_what_is_wrong(world: World) -> None:
    text = error_text(world.writer().call("save_recipe", **(CHANA | {"title": ""})))

    assert "invalid_recipe" in text and "title" in text


def test_only_the_required_fields_are_enough(world: World) -> None:
    minimal = {k: CHANA[k] for k in ("title", "ingredients", "instructions")}

    assert world.writer().call("save_recipe", **minimal)["isError"] is False


# --- Refusals ----------------------------------------------------------------------------


def test_without_the_write_scope_nothing_is_saved(world: World) -> None:
    reader = world.client(scopes=[RECIPES_READ_SCOPE])

    assert "insufficient_scope" in error_text(reader.call("save_recipe", **CHANA))
    assert len(world.repository.recipes) == 3  # only the fixture recipes


def test_beyond_the_daily_limit_nothing_more_is_saved(world: World) -> None:
    writer = world.writer()
    writer.call("save_recipe", **CHANA)
    writer.call("save_recipe", **CHANA)

    assert "daily_save_limit_reached" in error_text(writer.call("save_recipe", **CHANA))
    assert len(world.repository.recipes) == 5


@pytest.mark.parametrize(
    "overrides",
    [{"title": ""}, {"ingredients": []}, {"instructions": []}, {"difficulty": "IMPOSSIBLE"}],
    ids=["empty title", "no ingredients", "no instructions", "unknown difficulty"],
)
def test_invalid_recipes_are_tool_errors_and_save_nothing(
    world: World, overrides: dict[str, Any]
) -> None:
    error_text(world.writer().call("save_recipe", **(CHANA | overrides)))

    assert len(world.repository.recipes) == 3


def test_another_users_token_saves_for_that_user(world: World) -> None:
    other = world.client(uid=OTHER, scopes=[RECIPES_READ_SCOPE, "recipes:write"])

    saved_id = other.call("save_recipe", **CHANA)["structuredContent"]["recipe"]["id"]

    assert [r.uid for r in world.repository.recipes if r.id == saved_id] == [OTHER]
