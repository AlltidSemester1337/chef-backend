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
    NotRecipeOwner,
    RecipeDraft,
    RecipeNotFound,
    RecipeSaver,
    VariantDraft,
)
from chef_backend.recipes.views import RecipeView, VariantView

# Parameters a recipe and a variant share, described once for both tools.
Ingredients = Annotated[
    list[IngredientDraft],
    Field(
        description="Every ingredient: name, quantity and unit, e.g. "
        "{name: 'chickpeas', quantity: '800', unit: 'g'}. Leave quantity and unit "
        "empty for things like 'salt'."
    ),
]
Instructions = Annotated[
    list[str],
    Field(
        description="The steps in order, one per item, without numbers "
        "(e.g. 'Fry the onion until soft.'); Chef numbers them."
    ),
]
Summary = Annotated[str, Field(description="One or two sentences describing the dish.")]
Servings = Annotated[str, Field(description="How many it serves, e.g. '4'.")]
PrepTime = Annotated[str, Field(description="Preparation time, e.g. '15 minutes'.")]
CookingTime = Annotated[str, Field(description="Cooking time, e.g. '30 minutes' or '1 hour'.")]
DifficultyLevel = Annotated[
    Difficulty | None, Field(description="EASY, MEDIUM or HARD, if you can judge it.")
]
TipsAndTricks = Annotated[
    str, Field(description="Optional tips: substitutions, leftovers, what to watch for.")
]
Nutrients = Annotated[
    list[IngredientDraft],
    Field(
        description="Optional nutrition per serving, e.g. "
        "{name: 'protein', quantity: '14', unit: 'g'}."
    ),
]

# Open world: what these tools save can be seen by other Chef users, so clients that
# weigh annotations should treat a call with the care of publishing.
SHARED_WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
)


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
            "generated for it. To adapt one of the user's own recipes while keeping the "
            "original, use save_recipe_variant instead."
        ),
        annotations=SHARED_WRITE,
    )
    def save_recipe(
        title: Annotated[str, Field(description="The recipe's name, e.g. 'Chana masala'.")],
        ingredients: Ingredients,
        instructions: Instructions,
        summary: Summary = "",
        servings: Servings = "",
        prep_time: PrepTime = "",
        cooking_time: CookingTime = "",
        difficulty: DifficultyLevel = None,
        tags: Annotated[
            list[str],
            Field(
                description="Lowercase tags used for search: cuisine ('indian'), main "
                "ingredient ('chicken'), diet ('vegetarian'), time ('under 30 minutes'), "
                "occasion ('weeknight')."
            ),
        ] = [],  # noqa: B006 - never mutated; part of the schema the model sees
        tips_and_tricks: TipsAndTricks = "",
        nutrients_per_serving: Nutrients = [],  # noqa: B006 - as tags
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
            raise ToolError(f"invalid_recipe: {problems(error)}") from None
        try:
            saved = recipe_saver().save(uid=uid, draft=draft, today=today())
        except DailySaveLimitReached:
            raise ToolError("daily_save_limit_reached") from None
        return SaveRecipeResult(recipe=RecipeView.of(saved.recipe, uid), image=saved.image)

    @mcp.tool(
        description=(
            "Save an adapted version of one of the user's own recipes as a variant of it, "
            "e.g. a vegetarian version or one scaled for two. The original stays unchanged; "
            "the Chef app shows the variant next to it. Only for recipes that are the "
            "user's own (isMine is true): to adapt someone else's recipe, save the adapted "
            "recipe as the user's own with save_recipe instead. Give the whole adapted "
            "recipe, not only what changed, in English and with a quantity and unit for "
            "every ingredient. Variants of a recipe shared with the community are visible "
            "to other Chef users too, so confirm with the user before calling this. Each "
            "call adds a new variant. Returns the saved variant."
        ),
        annotations=SHARED_WRITE,
    )
    def save_recipe_variant(
        recipe_id: Annotated[
            str,
            Field(
                description="The id of the user's own recipe to add the variant to, "
                "from search_recipes or get_recipe."
            ),
        ],
        label: Annotated[
            str,
            Field(
                description="A short name for what is different, at most 100 characters, "
                "e.g. 'Vegetarian', 'For two' or 'No onion'."
            ),
        ],
        title: Annotated[
            str,
            Field(
                description="The variant's title as it should be shown, e.g. 'Vegetarian "
                "tikka masala'; may be the original title."
            ),
        ],
        ingredients: Ingredients,
        instructions: Instructions,
        summary: Summary = "",
        servings: Servings = "",
        prep_time: PrepTime = "",
        cooking_time: CookingTime = "",
        difficulty: DifficultyLevel = None,
        tips_and_tricks: TipsAndTricks = "",
        nutrients_per_serving: Nutrients = [],  # noqa: B006 - never mutated; schema only
    ) -> VariantView:
        require_scope(RECIPES_WRITE_SCOPE)
        try:
            draft = VariantDraft(
                label=label,
                title=title,
                summary=summary,
                servings=servings,
                prep_time=prep_time,
                cooking_time=cooking_time,
                difficulty=difficulty,
                ingredients=ingredients,
                instructions=instructions,
                tips_and_tricks=tips_and_tricks,
                nutrients_per_serving=nutrients_per_serving,
            )
        except ValidationError as error:
            raise ToolError(f"invalid_variant: {problems(error)}") from None
        try:
            saved = recipe_saver().save_variant(
                uid=current_uid(), recipe_id=recipe_id, draft=draft, today=today()
            )
        except RecipeNotFound:
            raise ToolError("recipe_not_found") from None
        except NotRecipeOwner:
            raise ToolError(
                "not_your_recipe: variants can only be added to the user's own recipes. "
                "Save the adapted recipe as the user's own with save_recipe instead."
            ) from None
        except DailySaveLimitReached:
            raise ToolError("daily_save_limit_reached") from None
        return VariantView.of(variant=saved)


def problems(error: ValidationError) -> str:
    """What is wrong with the input, per field, for the model to fix and retry."""
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in error.errors()
    )
