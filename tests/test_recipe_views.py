"""Views: search hits get a short, single-line summary; full recipes keep it whole."""

import pytest

from chef_backend.recipes.models import Recipe
from chef_backend.recipes.views import (
    MAX_SUMMARY_CHARS,
    RecipeSummaryView,
    RecipeView,
    short_summary,
)

LONG_SUMMARY = (
    "\n\nThis vibrant paella combines the flavors of Spain with a touch of spice.\n\n"
    "**Yields:** 4 servings\n**Prep time:** 20 minutes\n\n"
    "**Nutritional Information per serving (approximate):**\n\n"
    "* Calories: 520 kcal\n* Protein: 35g\n* Carbohydrates: 55g\n* Fat: 18g\n" * 3
)


def recipe(summary: str) -> Recipe:
    return Recipe.model_validate({"id": "r1", "uid": "me", "title": "Paella", "summary": summary})


def test_a_short_summary_is_kept_but_put_on_one_line() -> None:
    assert short_summary("  Quick and\n\nflavorful fajitas. ") == "Quick and flavorful fajitas."


def test_a_long_summary_is_cut_at_a_word_with_an_ellipsis() -> None:
    result = short_summary(LONG_SUMMARY)

    assert len(result) <= MAX_SUMMARY_CHARS
    assert result.startswith("This vibrant paella combines")
    assert result.endswith("…")
    assert "\n" not in result
    # Cut between words, not inside one.
    assert result[:-1].split(" ")[-1] in " ".join(LONG_SUMMARY.split()).split(" ")


@pytest.mark.parametrize("length", [MAX_SUMMARY_CHARS - 1, MAX_SUMMARY_CHARS])
def test_a_summary_at_the_limit_is_not_cut(length: int) -> None:
    text = "x" * length

    assert short_summary(text) == text


def test_one_overlong_word_is_still_cut() -> None:
    result = short_summary("x" * 500)

    assert len(result) == MAX_SUMMARY_CHARS and result.endswith("…")


def test_search_hits_get_the_short_summary_and_full_recipes_the_whole_text() -> None:
    hit = RecipeSummaryView.of(recipe(LONG_SUMMARY), "me")
    full = RecipeView.of(recipe(LONG_SUMMARY), "me")

    assert len(hit.summary) <= MAX_SUMMARY_CHARS
    assert full.summary == LONG_SUMMARY
