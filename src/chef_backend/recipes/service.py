"""Recipe reads for one signed-in user: the layer the HTTP routes and MCP tools call.

Every method takes the caller's uid and applies the access policy, so a caller can
never reach a recipe by guessing its id. Invisible and missing recipes look the same
(None), which the routes turn into 404.
"""

import re
from datetime import UTC, datetime
from enum import StrEnum

from chef_backend.recipes.access import is_visible_to
from chef_backend.recipes.models import Recipe, RecipeVariant
from chef_backend.recipes.repository import RecipeRepository
from chef_backend.recipes.search import search_recipes

_OLDEST = datetime.min.replace(tzinfo=UTC)
# Firestore auto-ids and the migrated RTDB push ids. Anything else, in particular a
# "/", would make document() address a different path than a recipe.
_RECIPE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


class Scope(StrEnum):
    MINE = "mine"
    COMMUNITY = "community"
    ALL = "all"


class RecipeService:
    def __init__(self, repository: RecipeRepository) -> None:
        self._repository = repository

    def search(self, uid: str, query: str, scope: Scope, limit: int) -> list[Recipe]:
        return search_recipes(self._candidates(uid, scope), query, limit)

    def get(self, uid: str, recipe_id: str) -> Recipe | None:
        if not _RECIPE_ID.fullmatch(recipe_id):
            return None
        recipe = self._repository.get(recipe_id)
        return recipe if recipe is not None and is_visible_to(recipe, uid) else None

    def list_variants(self, uid: str, recipe_id: str) -> list[RecipeVariant] | None:
        """The variants of a visible recipe, or None if the recipe is not visible."""
        if self.get(uid, recipe_id) is None:
            return None
        return self._repository.list_variants(recipe_id)

    def _candidates(self, uid: str, scope: Scope) -> list[Recipe]:
        match scope:
            case Scope.MINE:
                return self._repository.list_by_owner(uid)
            case Scope.COMMUNITY:
                return self._repository.list_community()
            case Scope.ALL:
                # The user's own favourites are in both lists: keep one copy of each.
                mine = self._repository.list_by_owner(uid)
                own_ids = {recipe.id for recipe in mine}
                community = [r for r in self._repository.list_community() if r.id not in own_ids]
                merged = mine + community
                merged.sort(key=lambda r: r.updated_at or _OLDEST, reverse=True)
                return merged
