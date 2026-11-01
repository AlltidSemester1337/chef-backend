"""MCP tools that change the user's data. Each one checks its own scope first.

Reads only need recipes:read, which every /mcp request has; a tool that writes
calls require_scope(...) so that a connection made without that permission gets
an error the model can explain ("reconnect Chef and allow it").
"""

from collections.abc import Callable
from datetime import date
from typing import Annotated

from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from chef_backend.mcp_server.auth import RECIPES_WRITE_SCOPE, current_uid, require_scope
from chef_backend.mcp_server.base import ChefMCP
from chef_backend.recipes.models import Difficulty
from chef_backend.recipes.saving import (
    DailySaveLimitReached,
    ImageOutcome,
    IngredientDraft,
    RecipeDraft,
    RecipeSaver,
)
from chef_backend.recipes.views import RecipeView


class SaveRecipeResult(BaseModel):
    """What the model gets back: the saved recipe as the user sees it (no uid)."""

    recipe: RecipeView
    image: ImageOutcome


def register_write_tools(
    mcp: ChefMCP, recipe_saver: Callable[[], RecipeSaver], today: Callable[[], date]
) -> None:
    """Add the write tools to `mcp`. Call `recipe_saver()` and `today()` inside, per call."""

    @mcp.tool(
        description=(
            "Save a new recipe to the user's Chef collection, for example one you created "
            "or adapted together in this conversation. Like saving in the Chef app, a saved "
            "recipe is also shared with the Chef community, so confirm with the user before "
            "calling this. Write the recipe in English (Chef stores recipes in English) and "
            "give every ingredient its quantity and unit. Each call saves a new recipe, so "
            "call it once per recipe. Returns the saved recipe and whether an image was "
            "generated for it."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
        ),
    )
    def save_recipe(
        title: Annotated[str, Field(description="The recipe's name, e.g. 'Chana masala'.")],
        ingredients: Annotated[
            list[IngredientDraft],
            Field(
                description="Every ingredient: name, quantity and unit, e.g. "
                "{name: 'chickpeas', quantity: '800', unit: 'g'}. Leave quantity and unit "
                "empty for things like 'salt'."
            ),
        ],
        instructions: Annotated[
            list[str],
            Field(
                description="The steps in order, one per item, without numbers "
                "(e.g. 'Fry the onion until soft.'); Chef numbers them."
            ),
        ],
        summary: Annotated[
            str, Field(description="One or two sentences describing the dish.")
        ] = "",
        servings: Annotated[str, Field(description="How many it serves, e.g. '4'.")] = "",
        prep_time: Annotated[str, Field(description="Preparation time, e.g. '15 minutes'.")] = "",
        cooking_time: Annotated[
            str, Field(description="Cooking time, e.g. '30 minutes' or '1 hour'.")
        ] = "",
        difficulty: Annotated[
            Difficulty | None, Field(description="EASY, MEDIUM or HARD, if you can judge it.")
        ] = None,
        tags: Annotated[
            list[str],
            Field(
                description="Lowercase tags used for search: cuisine ('indian'), main "
                "ingredient ('chicken'), diet ('vegetarian'), time ('under 30 minutes'), "
                "occasion ('weeknight')."
            ),
        ] = [],  # noqa: B006 - never mutated; part of the schema the model sees
        tips_and_tricks: Annotated[
            str,
            Field(description="Optional tips: substitutions, leftovers, what to watch for."),
        ] = "",
        nutrients_per_serving: Annotated[
            list[IngredientDraft],
            Field(
                description="Optional nutrition per serving, e.g. "
                "{name: 'protein', quantity: '14', unit: 'g'}."
            ),
        ] = [],  # noqa: B006 - never mutated; part of the schema the model sees
    ) -> SaveRecipeResult:
        require_scope(RECIPES_WRITE_SCOPE)
        uid = current_uid()
        try:
            draft = RecipeDraft(
                title=title,
                summary=summary,
                servings=servings,
                prep_time=prep_time,
                cooking_time=cooking_time,
                difficulty=difficulty,
                tags=tags,
                ingredients=ingredients,
                instructions=instructions,
                tips_and_tricks=tips_and_tricks,
                nutrients_per_serving=nutrients_per_serving,
            )
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in error.errors()
            )
            raise ToolError(f"invalid_recipe: {problems}") from None
        try:
            saved = recipe_saver().save(uid=uid, draft=draft, today=today())
        except DailySaveLimitReached:
            raise ToolError("daily_save_limit_reached") from None
        return SaveRecipeResult(recipe=RecipeView.of(saved.recipe, uid), image=saved.image)
