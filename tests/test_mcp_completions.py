"""Spec for argument completion (chef_backend.mcp_server.completions).

One completion handler answers for:
- prompt shopping_list, argument recipe_ids: the last fragment the user is typing
  (after the last comma or space) is matched like search_recipes does (title, tags,
  ingredient names; scope all, newest first). Each suggestion is the whole new
  argument value: the ids already typed, joined with ", ", then the suggested id.
  Ids already typed are not suggested again. An empty fragment suggests everything.
- resource template chef://recipes/{recipe_id}, argument recipe_id: the same match,
  each suggestion just the id.
- prompt plan_week, argument days: "1" to "7", those starting with what is typed.
- anything else: no suggestions.
At most MAX_SUGGESTIONS values; when there are more, hasMore is true and total is
the number of matches. Only recipes the user may see are suggested.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_support import (
    BASE_URL,
    ME,
    FakeMcpTokenVerifier,
    FakeRecipeRepository,
    McpClient,
    recipe,
)

from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.mcp_server.completions import MAX_SUGGESTIONS
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService

SHOPPING = {"type": "ref/prompt", "name": "shopping_list"}
PLAN = {"type": "ref/prompt", "name": "plan_week"}
RECIPE = {"type": "ref/resource", "uri": "chef://recipes/{recipe_id}"}


@pytest.fixture
def repository() -> FakeRecipeRepository:
    return FakeRecipeRepository()


@pytest.fixture
def mcp(repository: FakeRecipeRepository) -> Iterator[McpClient]:
    verifier = FakeMcpTokenVerifier()
    app = create_app()
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(repository)
    with TestClient(app, base_url=BASE_URL) as http:
        yield McpClient(http, verifier.issue())


def completion(response: dict[str, Any]) -> dict[str, Any]:
    assert "result" in response, response
    return response["result"]["completion"]


def values(response: dict[str, Any]) -> list[str]:
    return completion(response)["values"]


def test_the_server_says_it_offers_completions(mcp: McpClient) -> None:
    response = mcp.post(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    ).json()

    assert "completions" in response["result"]["capabilities"]


# --- shopping_list: recipe_ids -----------------------------------------------------------


def test_an_empty_value_suggests_every_visible_recipe_newest_first(mcp: McpClient) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", "")) == ["theirs-shared", "mine"]


def test_the_fragment_is_matched_like_the_search(mcp: McpClient) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", "mi")) == ["mine"]
    assert values(mcp.complete(SHOPPING, "recipe_ids", "ONION")) == ["theirs-shared", "mine"]


def test_recipes_the_user_may_not_see_are_never_suggested(mcp: McpClient) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", "theirs")) == ["theirs-shared"]


def test_only_the_last_fragment_is_completed_and_typed_ids_are_kept(mcp: McpClient) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", "mine, theirs")) == ["mine, theirs-shared"]


@pytest.mark.parametrize("typed", ["mine,", "mine ", "mine, "])
def test_after_a_separator_the_next_recipe_is_suggested_without_repeats(
    mcp: McpClient, typed: str
) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", typed)) == ["mine, theirs-shared"]


def test_no_match_gives_no_suggestions(mcp: McpClient) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", "sushi")) == []


def test_many_matches_are_capped_and_marked(
    mcp: McpClient, repository: FakeRecipeRepository
) -> None:
    repository.recipes += [recipe(f"extra-{n}", ME, day=10 + n) for n in range(MAX_SUGGESTIONS + 5)]

    found = completion(mcp.complete(SHOPPING, "recipe_ids", ""))

    assert len(found["values"]) == MAX_SUGGESTIONS
    assert found["values"][0] == f"extra-{MAX_SUGGESTIONS + 4}"
    assert (found.get("hasMore"), found.get("total")) == (True, MAX_SUGGESTIONS + 7)


def test_exactly_the_cap_is_not_marked_as_more(
    mcp: McpClient, repository: FakeRecipeRepository
) -> None:
    repository.recipes += [recipe(f"extra-{n}", ME, day=10 + n) for n in range(MAX_SUGGESTIONS - 2)]

    for found in (
        completion(mcp.complete(SHOPPING, "recipe_ids", "")),
        completion(mcp.complete(RECIPE, "recipe_id", "")),
    ):
        assert len(found["values"]) == MAX_SUGGESTIONS
        assert found.get("hasMore") is not True
        assert found.get("total") is None


def test_typed_ids_do_not_count_towards_the_cap(
    mcp: McpClient, repository: FakeRecipeRepository
) -> None:
    repository.recipes += [recipe(f"extra-{n}", ME, day=10 + n) for n in range(MAX_SUGGESTIONS - 1)]

    found = completion(mcp.complete(SHOPPING, "recipe_ids", "mine, "))

    assert len(found["values"]) == MAX_SUGGESTIONS
    assert "mine, mine" not in found["values"]
    assert found.get("hasMore") is not True


def test_leading_separators_are_ignored(mcp: McpClient) -> None:
    assert values(mcp.complete(SHOPPING, "recipe_ids", " mine,")) == ["mine, theirs-shared"]


# --- The recipe resource template --------------------------------------------------------


def test_the_recipe_uri_completes_to_bare_ids(mcp: McpClient) -> None:
    assert values(mcp.complete(RECIPE, "recipe_id", "theirs")) == ["theirs-shared"]
    assert values(mcp.complete(RECIPE, "recipe_id", "")) == ["theirs-shared", "mine"]


# --- plan_week: days ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("", ["1", "2", "3", "4", "5", "6", "7"]), ("5", ["5"]), ("9", []), ("x", [])],
)
def test_days_complete_to_one_through_seven(
    mcp: McpClient, typed: str, expected: list[str]
) -> None:
    assert values(mcp.complete(PLAN, "days", typed)) == expected


# --- Everything else -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "argument"),
    [(PLAN, "wishes"), (SHOPPING, "servings"), ({"type": "ref/prompt", "name": "nope"}, "x")],
    ids=["free text", "servings", "unknown prompt"],
)
def test_other_arguments_get_no_suggestions(
    mcp: McpClient, ref: dict[str, str], argument: str
) -> None:
    assert values(mcp.complete(ref, argument, "")) == []
