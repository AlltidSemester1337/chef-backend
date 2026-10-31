"""The MCP tools: what an AI assistant can do with the user's recipes.

Tools are thin: they take the caller's uid from the access token (current_uid),
call RecipeService and return the same views as the HTTP API. The model reads the
names, descriptions and parameter schemas to decide when and how to call a tool,
so those are part of the interface, not decoration.
"""

from collections.abc import Callable
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from chef_backend.mcp_server.auth import current_uid
from chef_backend.recipes.service import MAX_QUERY_CHARS, MAX_SEARCH_LIMIT, RecipeService, Scope
from chef_backend.recipes.views import (
    RecipeSearchResponse,
    RecipeSummaryView,
    RecipeView,
    VariantsResponse,
    VariantView,
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

RecipeId = Annotated[
    str,
    Field(
        description="The id of a recipe, as returned by search_recipes.",
        min_length=1,
    ),
]


def register_tools(mcp: FastMCP, recipe_service: Callable[[], RecipeService]) -> None:
    """Add the recipe tools to `mcp`. Call `recipe_service()` inside a tool, per call."""

    @mcp.tool(
        description="Get one recipe in full: ingredients with quantity and unit, "
        "step-by-step instructions, tips, servings and nutrients per serving. "
        "Use the id from search_recipes. 'recipe_not_found' means the id does not exist "
        "or is not visible to the user; do not retry it.",
        annotations=READ_ONLY,
    )
    def get_recipe(recipe_id: RecipeId) -> RecipeView:
        uid = current_uid()
        recipe = recipe_service().get(uid=uid, recipe_id=recipe_id)
        if recipe is None:
            raise ToolError("recipe_not_found")
        return RecipeView.of(recipe=recipe, viewer_uid=uid)

    @mcp.tool(
        description="Find recipes by keywords. Returns short summaries (title, summary, "
        "tags, times, isMine, isFavourite = shared with the community), newest first. "
        "Every word must occur in the title, a tag or an ingredient name; partial words "
        "match ('chick' finds 'chickpeas'). Use 1-3 specific keywords such as an "
        "ingredient, dish or tag ('chicken curry', 'vegetarian'), not a sentence. If "
        "nothing matches, retry with fewer or different words before concluding there "
        "is no such recipe. Use get_recipe for ingredients and instructions.",
        annotations=READ_ONLY,
    )
    def search_recipes(
        query: Annotated[
            str,
            Field(
                description="1-3 English keywords. Empty lists every recipe in the scope.",
                max_length=MAX_QUERY_CHARS,
            ),
        ] = "",
        scope: Annotated[
            Literal["mine", "others", "community", "all"],
            Field(
                description="Which recipes to search. "
                "mine = the user's own recipes. "
                "others = recipes shared by other users; "
                "use this when the user asks for the "
                "community's recipes or for inspiration from others. "
                "community = every shared recipe, "
                "the user's own shared ones included; use only "
                "when they explicitly want their own shared recipes too. "
                "all = mine + community (default)."
            ),
        ] = "all",
        limit: Annotated[
            int,
            Field(
                description="Maximum number of recipes returned. Keep it small: every hit "
                "takes up context.",
                ge=1,
                le=MAX_SEARCH_LIMIT,
            ),
        ] = 10,
    ) -> RecipeSearchResponse:
        uid = current_uid()
        recipes = recipe_service().search(uid=uid, query=query, scope=Scope(scope), limit=limit)
        recipes_view = [RecipeSummaryView.of(recipe=r, viewer_uid=uid) for r in recipes]
        return RecipeSearchResponse(recipes=recipes_view)

    @mcp.tool(
        description="List the user's saved variants of a recipe, oldest first: adapted "
        "versions such as milder, vegan or a half batch, each with its own ingredients "
        "and instructions. Often empty.",
        annotations=READ_ONLY,
    )
    def list_recipe_variants(recipe_id: RecipeId) -> VariantsResponse:
        variants = recipe_service().list_variants(uid=current_uid(), recipe_id=recipe_id)
        if variants is None:
            raise ToolError("recipe_not_found")
        variants_view = [VariantView.of(variant=v) for v in variants]
        return VariantsResponse(variants=variants_view)
