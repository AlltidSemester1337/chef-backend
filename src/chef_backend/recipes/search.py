"""Free-text search over a list of recipes, in memory.

The candidates are already filtered by access and ordered newest first (see
RecipeService); this module only decides which of them match and how many to return.
A beta user has tens of recipes, so a linear scan is fine; a real index can replace
this later without changing the callers.
"""

from collections.abc import Iterable
from itertools import islice

from chef_backend.recipes.models import Recipe


def search_recipes(recipes: Iterable[Recipe], query: str, limit: int) -> list[Recipe]:
    """The recipes matching `query`, in their given order, at most `limit` of them."""
    words = query.casefold().split()
    matches = (recipe for recipe in recipes if _matches(recipe, words))
    # islice stops reading the candidates once `limit` matches are found.
    return list(islice(matches, max(limit, 0)))


def _matches(recipe: Recipe, words: list[str]) -> bool:
    """Every word occurs in at least one searchable field. No words match everything."""
    fields = [recipe.title, *recipe.tags, *(ingredient.name for ingredient in recipe.ingredients)]
    searchable = [field.casefold() for field in fields]
    return all(any(word in field for field in searchable) for word in words)
