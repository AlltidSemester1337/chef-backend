"""HTTP routes for reading recipes: search, one recipe, its variants.

Read-only, no AI and no quota: any signed-in user (with App Check) may read what the
access policy lets them see. Invisible and missing recipes are both 404.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from chef_backend.ai.dependencies import get_firestore_client
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.recipes.repository import FirestoreRecipeRepository
from chef_backend.recipes.service import RecipeService, Scope
from chef_backend.recipes.views import RecipeSummaryView, RecipeView, VariantView

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_QUERY_CHARS = 200


@lru_cache
def get_recipe_service() -> RecipeService:
    return RecipeService(FirestoreRecipeRepository(get_firestore_client()))


class RecipeSearchResponse(BaseModel):
    recipes: list[RecipeSummaryView]


class VariantsResponse(BaseModel):
    variants: list[VariantView]


router = APIRouter(prefix="/v1/recipes", tags=["recipes"])

CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]
Service = Annotated[RecipeService, Depends(get_recipe_service)]


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "recipe_not_found")


@router.get("")
def search_recipes(
    user: CurrentUser,
    service: Service,
    q: Annotated[str, Query(max_length=MAX_QUERY_CHARS)] = "",
    scope: Scope = Scope.ALL,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> RecipeSearchResponse:
    """Search the caller's own and the community's recipes, newest first."""
    recipes = service.search(user.uid, q, scope, limit)
    return RecipeSearchResponse(recipes=[RecipeSummaryView.of(r, user.uid) for r in recipes])


@router.get("/{recipe_id}")
def get_recipe(recipe_id: str, user: CurrentUser, service: Service) -> RecipeView:
    """One recipe the caller may see."""
    recipe = service.get(user.uid, recipe_id)
    if recipe is None:
        raise _not_found()
    return RecipeView.of(recipe, user.uid)


@router.get("/{recipe_id}/variants")
def list_variants(recipe_id: str, user: CurrentUser, service: Service) -> VariantsResponse:
    """The variants of a recipe the caller may see, oldest first."""
    variants = service.list_variants(user.uid, recipe_id)
    if variants is None:
        raise _not_found()
    return VariantsResponse(variants=[VariantView.of(v) for v in variants])
