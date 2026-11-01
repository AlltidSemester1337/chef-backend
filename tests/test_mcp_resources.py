"""Spec for the MCP recipe resources (chef_backend.mcp_server.resources).

Task 1, reading: chef://recipes/{recipe_id} is a resource template. Reading it gives
the recipe as Markdown (mimeType text/markdown) for the user in the access token:

    # Title

    Summary (when there is one)

    - Servings: 4 · Prep: 10 minutes · Cook: 20 minutes · Difficulty: EASY
      (only the parts that have a value; the line is left out if none has)
    - Tags: vegetarian, quick  (left out without tags)

    ## Ingredients
    - 1 pc onion        (quantity, unit and name; empty parts left out)

    ## Instructions
    1. Chop.

    ## Tips
    The tips text.

A section without content is left out. Invisible and missing recipes are the same
error, "recipe_not_found". The owner's uid never appears.

Task 2, listing: resources/list includes the user's own recipes, newest first and at
most MAX_LISTED_RECIPES, so they show up in the client's attachment menu. Each one
is uri chef://recipes/{id}, name = the title, mimeType text/markdown and, when the
recipe has a summary, a short description (views.short_summary). Other users'
recipes are never listed, shared or not.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_support import (
    BASE_URL,
    ME,
    OTHER,
    FakeMcpTokenVerifier,
    FakeRecipeRepository,
    McpClient,
)

from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.mcp_server.resources import MAX_LISTED_RECIPES
from chef_backend.recipes.models import Recipe
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService
from chef_backend.recipes.views import short_summary

FULL = Recipe.model_validate(
    {
        "id": "full",
        "uid": ME,
        "title": "Chana masala",
        "summary": "Spicy chickpea stew.",
        "difficulty": "EASY",
        "servings": "4",
        "prepTime": "10 minutes",
        "cookingTime": "20 minutes",
        "tags": ["vegetarian", "indian"],
        "ingredients": [
            {"name": "chickpeas", "quantity": "800", "unit": "g"},
            {"name": "garam masala", "quantity": "2", "unit": "tsp"},
            {"name": "salt"},
        ],
        "instructions": ["Fry the onion.", "Add the spices.", "Simmer for 20 minutes."],
        "tipsAndTricks": "Toast the spices first.",
    }
)
BARE = Recipe.model_validate({"id": "bare", "uid": ME, "title": "Toast"})


@pytest.fixture
def verifier() -> FakeMcpTokenVerifier:
    return FakeMcpTokenVerifier()


@pytest.fixture
def repository() -> FakeRecipeRepository:
    repository = FakeRecipeRepository()
    repository.recipes += [FULL, BARE]
    return repository


@pytest.fixture
def http(verifier: FakeMcpTokenVerifier, repository: FakeRecipeRepository) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(repository)
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


@pytest.fixture
def mcp(http: TestClient, verifier: FakeMcpTokenVerifier) -> McpClient:
    return McpClient(http, verifier.issue())


def markdown(mcp: McpClient, recipe_id: str) -> str:
    response = mcp.read(f"chef://recipes/{recipe_id}")
    assert "result" in response, response
    [content] = response["result"]["contents"]
    assert content["uri"] == f"chef://recipes/{recipe_id}"
    assert content["mimeType"] == "text/markdown"
    return content["text"]


# --- The template ---------------------------------------------------------------------


def test_recipes_are_offered_as_a_described_markdown_template(mcp: McpClient) -> None:
    [template] = [
        t for t in mcp.list_resource_templates() if t["uriTemplate"] == "chef://recipes/{recipe_id}"
    ]

    assert template["name"]
    assert len(template.get("description") or "") >= 20
    assert template["mimeType"] == "text/markdown"


# --- The Markdown ------------------------------------------------------------------------


def test_a_full_recipe_reads_as_markdown(mcp: McpClient) -> None:
    assert markdown(mcp, "full") == (
        "# Chana masala\n"
        "\n"
        "Spicy chickpea stew.\n"
        "\n"
        "- Servings: 4 · Prep: 10 minutes · Cook: 20 minutes · Difficulty: EASY\n"
        "- Tags: vegetarian, indian\n"
        "\n"
        "## Ingredients\n"
        "- 800 g chickpeas\n"
        "- 2 tsp garam masala\n"
        "- salt\n"
        "\n"
        "## Instructions\n"
        "1. Fry the onion.\n"
        "2. Add the spices.\n"
        "3. Simmer for 20 minutes.\n"
        "\n"
        "## Tips\n"
        "Toast the spices first.\n"
    )


def test_empty_parts_and_sections_are_left_out(mcp: McpClient) -> None:
    assert markdown(mcp, "bare") == "# Toast\n"


def test_another_users_shared_recipe_can_be_read_without_its_owner(mcp: McpClient) -> None:
    text = markdown(mcp, "theirs-shared")

    assert text.startswith("# Title theirs-shared\n")
    assert "- 1 pc onion\n" in text
    assert OTHER not in text and ME not in text


# --- Errors -------------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_id", ["theirs-private", "does-not-exist", "bad.id"])
def test_invisible_and_missing_recipes_are_the_same_error(mcp: McpClient, recipe_id: str) -> None:
    response: dict[str, Any] = mcp.read(f"chef://recipes/{recipe_id}")

    assert response["error"]["message"] == "recipe_not_found"


def test_a_path_is_never_taken_for_a_recipe_id(mcp: McpClient) -> None:
    # The template matches one path segment, so this never reaches the service.
    assert mcp.read("chef://recipes/mine/variants/v1")["error"]["message"] == "unknown_resource"


# --- Task 2: the list ------------------------------------------------------------------


def recipe_uris(mcp: McpClient) -> list[str]:
    return [r["uri"] for r in mcp.list_resources() if r["uri"].startswith("chef://recipes/")]


def test_the_users_own_recipes_are_listed_newest_first(mcp: McpClient) -> None:
    # The fake repository returns the user's recipes newest first: bare, full, mine.
    assert recipe_uris(mcp) == ["chef://recipes/bare", "chef://recipes/full", "chef://recipes/mine"]


def test_each_listed_recipe_is_named_and_described(mcp: McpClient) -> None:
    listed = {r["uri"]: r for r in mcp.list_resources()}

    full = listed["chef://recipes/full"]
    assert (full["name"], full["mimeType"]) == ("Chana masala", "text/markdown")
    assert full["description"] == short_summary(FULL.summary)
    assert listed["chef://recipes/bare"]["name"] == "Toast"
    assert not listed["chef://recipes/bare"].get("description")


def test_other_users_recipes_are_never_listed(mcp: McpClient) -> None:
    uris = recipe_uris(mcp)

    assert "chef://recipes/theirs-shared" not in uris
    assert "chef://recipes/theirs-private" not in uris


def test_the_list_is_for_the_user_in_the_token(
    http: TestClient, verifier: FakeMcpTokenVerifier
) -> None:
    other = McpClient(http, verifier.issue("other-token", uid=OTHER))

    assert set(recipe_uris(other)) == {
        "chef://recipes/theirs-shared",
        "chef://recipes/theirs-private",
    }


def test_at_most_the_newest_recipes_are_listed(
    mcp: McpClient, repository: FakeRecipeRepository
) -> None:
    repository.recipes += [
        Recipe.model_validate({"id": f"extra-{n}", "uid": ME, "title": f"Extra {n}"})
        for n in range(MAX_LISTED_RECIPES + 5)
    ]

    uris = recipe_uris(mcp)

    assert len(uris) == MAX_LISTED_RECIPES
    assert uris[0] == f"chef://recipes/extra-{MAX_LISTED_RECIPES + 4}"


def test_every_listed_recipe_can_be_read(mcp: McpClient) -> None:
    for uri in recipe_uris(mcp):
        assert "result" in mcp.read(uri), uri
