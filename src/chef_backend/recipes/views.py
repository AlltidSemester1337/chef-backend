"""What callers see of a recipe: shared by the HTTP API and the MCP tools.

The stored models carry the owner's uid; these views replace it with `isMine`, so
a caller never learns another user's uid. Fields are listed explicitly (no copying
of the stored model), so a field added to the documents is not exposed by accident.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from chef_backend.recipes.models import Difficulty, Ingredient, Recipe, RecipeVariant


class _View(BaseModel):
    # camelCase on the wire, like the documents the apps already read.
    model_config = ConfigDict(
        alias_generator=to_camel, serialize_by_alias=True, validate_by_name=True, frozen=True
    )


class IngredientView(_View):
    name: str
    quantity: str
    unit: str

    @classmethod
    def of(cls, ingredient: Ingredient) -> "IngredientView":
        return cls(name=ingredient.name, quantity=ingredient.quantity, unit=ingredient.unit)


class RecipeSummaryView(_View):
    """A search hit: enough to pick a recipe, without the full text."""

    id: str
    title: str
    summary: str
    difficulty: Difficulty | None
    prep_time: str
    cooking_time: str
    servings: str
    tags: list[str]
    image_url: str | None
    is_mine: bool
    is_favourite: bool
    updated_at: datetime | None

    @classmethod
    def of(cls, recipe: Recipe, viewer_uid: str) -> "RecipeSummaryView":
        return cls.model_validate(_summary_fields(recipe, viewer_uid))


class RecipeView(RecipeSummaryView):
    ingredients: list[IngredientView]
    instructions: list[str]
    tips_and_tricks: str
    nutrients_per_serving: list[IngredientView]
    video_url: str | None

    @classmethod
    def of(cls, recipe: Recipe, viewer_uid: str) -> "RecipeView":
        return cls.model_validate(
            _summary_fields(recipe, viewer_uid)
            | {
                "ingredients": [IngredientView.of(i) for i in recipe.ingredients],
                "instructions": list(recipe.instructions),
                "tips_and_tricks": recipe.tips_and_tricks,
                "nutrients_per_serving": [
                    IngredientView.of(n) for n in recipe.nutrients_per_serving
                ],
                "video_url": recipe.video_url,
            }
        )


class VariantView(_View):
    id: str
    label: str
    created_at: datetime | None
    is_pinned: bool
    title: str
    summary: str
    difficulty: Difficulty | None
    prep_time: str
    cooking_time: str
    servings: str
    ingredients: list[IngredientView]
    instructions: list[str]
    tips_and_tricks: str
    nutrients_per_serving: list[IngredientView]

    @classmethod
    def of(cls, variant: RecipeVariant) -> "VariantView":
        return cls(
            id=variant.id,
            label=variant.label,
            created_at=variant.created_at,
            is_pinned=variant.is_pinned,
            title=variant.title,
            summary=variant.summary,
            difficulty=variant.difficulty,
            prep_time=variant.prep_time,
            cooking_time=variant.cooking_time,
            servings=variant.servings,
            ingredients=[IngredientView.of(i) for i in variant.ingredients],
            instructions=list(variant.instructions),
            tips_and_tricks=variant.tips_and_tricks,
            nutrients_per_serving=[IngredientView.of(n) for n in variant.nutrients_per_serving],
        )


def _summary_fields(recipe: Recipe, viewer_uid: str) -> dict[str, object]:
    return {
        "id": recipe.id,
        "title": recipe.title,
        "summary": recipe.summary,
        "difficulty": recipe.difficulty,
        "prep_time": recipe.prep_time,
        "cooking_time": recipe.cooking_time,
        "servings": recipe.servings,
        "tags": list(recipe.tags),
        "image_url": recipe.image_url,
        "is_mine": recipe.uid == viewer_uid,
        "is_favourite": recipe.is_favourite,
        "updated_at": recipe.updated_at,
    }
