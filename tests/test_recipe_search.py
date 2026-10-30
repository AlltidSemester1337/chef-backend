"""Spec for search_recipes: which candidates match a free-text query.

Matching: every word of the query must occur, case-insensitively, as a substring of
the title, a tag or an ingredient name (different words may match different fields).
Summary and instructions are not searched. Order is kept, and at most `limit`
recipes are returned.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from chef_backend.recipes.models import Ingredient, Recipe
from chef_backend.recipes.search import search_recipes


def recipe(recipe_id: str, title: str, **fields: Any) -> Recipe:
    return Recipe(id=recipe_id, title=title, **fields)


def ingredients(*names: str) -> list[Ingredient]:
    return [Ingredient(name=name) for name in names]


CHANA = recipe(
    "chana",
    "Chana masala",
    tags=["vegetarian", "indian"],
    ingredients=ingredients("kikärtor", "garam masala", "lök"),
    summary="Kryddig gryta med kokosmjölk.",
    instructions=["Rosta kryddorna i en het panna."],
)
PANCAKES = recipe(
    "pancakes",
    "Pannkakor",
    tags=["vegetarian", "breakfast"],
    ingredients=ingredients("Ägg", "mjölk", "vetemjöl"),
)
TACOS = recipe(
    "tacos",
    "Fredagstacos",
    tags=["mexican"],
    ingredients=ingredients("nötfärs", "tortillas", "lök"),
)
ALL = [CHANA, PANCAKES, TACOS]


def ids(recipes: list[Recipe]) -> list[str]:
    return [r.id for r in recipes]


# --- What matches --------------------------------------------------------------


def test_matches_a_substring_of_the_title() -> None:
    assert ids(search_recipes(ALL, "pannkak", limit=10)) == ["pancakes"]


def test_matches_a_tag() -> None:
    assert ids(search_recipes(ALL, "mexican", limit=10)) == ["tacos"]


def test_matches_an_ingredient_name() -> None:
    assert ids(search_recipes(ALL, "kikärt", limit=10)) == ["chana"]


@pytest.mark.parametrize("query", ["CHANA", "Chana", "ägg", "ÄGG"])
def test_matching_ignores_case_including_swedish_letters(query: str) -> None:
    assert len(search_recipes(ALL, query, limit=10)) == 1


def test_matching_ignores_case_in_tags_too() -> None:
    # Tags are free text from the apps and the model, so their case varies.
    soup = recipe("soup", "Soppa", tags=["Vegan", "Lunch"])

    assert ids(search_recipes([soup], "vegan", limit=10)) == ["soup"]


def test_every_word_must_match() -> None:
    assert ids(search_recipes(ALL, "vegetarian mjölk", limit=10)) == ["pancakes"]
    assert search_recipes(ALL, "vegetarian nötfärs", limit=10) == []


def test_words_may_match_different_fields() -> None:
    # "chana" in the title, "indian" in the tags, "garam" in an ingredient.
    assert ids(search_recipes(ALL, "chana indian garam", limit=10)) == ["chana"]


def test_extra_whitespace_between_words_is_ignored() -> None:
    assert ids(search_recipes(ALL, "  vegetarian   mjölk ", limit=10)) == ["pancakes"]


def test_summary_and_instructions_are_not_searched() -> None:
    assert search_recipes(ALL, "kokosmjölk", limit=10) == []
    assert search_recipes(ALL, "panna", limit=10) == []


def test_no_match_is_an_empty_list() -> None:
    assert search_recipes(ALL, "sushi", limit=10) == []


# --- Order and limit -----------------------------------------------------------


def test_matches_keep_the_given_order() -> None:
    # "lök" is in chana and tacos; the candidates come newest first.
    assert ids(search_recipes([TACOS, PANCAKES, CHANA], "lök", limit=10)) == ["tacos", "chana"]


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_matches_everything(query: str) -> None:
    assert ids(search_recipes(ALL, query, limit=10)) == ["chana", "pancakes", "tacos"]


def test_at_most_limit_recipes_are_returned() -> None:
    assert ids(search_recipes(ALL, "", limit=2)) == ["chana", "pancakes"]
    assert ids(search_recipes(ALL, "vegetarian", limit=1)) == ["chana"]


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_returns_nothing(limit: int) -> None:
    assert search_recipes(ALL, "", limit=limit) == []


def test_works_on_any_iterable_without_consuming_more_than_needed() -> None:
    seen: list[str] = []

    def candidates() -> Iterator[Recipe]:
        for r in ALL:
            seen.append(r.id)
            yield r

    assert ids(search_recipes(candidates(), "vegetarian", limit=1)) == ["chana"]
    assert seen == ["chana"]
