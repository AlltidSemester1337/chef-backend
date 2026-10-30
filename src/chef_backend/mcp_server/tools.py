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
        description="recipe id value to be fetched, returned from search_recipes() tool.",
        min_length=1,
    ),
]


def register_tools(mcp: FastMCP, recipe_service: Callable[[], RecipeService]) -> None:
    """Add the recipe tools to `mcp`. Call `recipe_service()` inside a tool, per call."""

    @mcp.tool(
        description="Get recipe details by passing reference by id. "
        "Use when details for a recipe is required.",
        annotations=READ_ONLY,
    )
    def get_recipe(recipe_id: RecipeId) -> RecipeView:
        uid = current_uid()
        recipe = recipe_service().get(uid=uid, recipe_id=recipe_id)
        if recipe is None:
            raise ToolError("recipe_not_found")
        return RecipeView.of(recipe=recipe, viewer_uid=uid)

    @mcp.tool(
        description="Search recipes by given search query to return list "
        "of matching recipe summaries. "
        "Use whenever searching recipes is required. "
        "For recipe details use get_recipe() tool.",
        annotations=READ_ONLY,
    )
    def search_recipes(
        query: Annotated[
            str,
            Field(
                description="search query value in english, "
                "each word must exist in recipe tags, ingredients or title. "
                "An empty query lists every recipe in the scope.",
                max_length=MAX_QUERY_CHARS,
            ),
        ] = "",
        scope: Annotated[
            Literal["mine", "community", "all"],
            Field(
                description="scope for search, mine = the user's own recipes, "
                "community = recipes shared with the community, "
                "including the user's own shared ones, "
                "all = mine + community"
            ),
        ] = "all",
        limit: Annotated[
            int, Field(description="maximum number of recipes returned", ge=1, le=MAX_SEARCH_LIMIT)
        ] = 10,
    ) -> RecipeSearchResponse:
        uid = current_uid()
        recipes = recipe_service().search(uid=uid, query=query, scope=Scope(scope), limit=limit)
        recipes_view = [RecipeSummaryView.of(recipe=r, viewer_uid=uid) for r in recipes]
        return RecipeSearchResponse(recipes=recipes_view)

    @mcp.tool(
        description="List variants for a given recipe by reference id. "
        "Variant means alternate version of original recipe for example vegan. "
        "Use whenever fetching recipe variant details is required.",
        annotations=READ_ONLY,
    )
    def list_recipe_variants(recipe_id: RecipeId) -> VariantsResponse:
        variants = recipe_service().list_variants(uid=current_uid(), recipe_id=recipe_id)
        if variants is None:
            raise ToolError("recipe_not_found")
        variants_view = [VariantView.of(variant=v) for v in variants]
        return VariantsResponse(variants=variants_view)
