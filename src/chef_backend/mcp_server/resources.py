"""The MCP resources: recipes a user can attach to a conversation.

A resource is content the client (or the user, from the attachment menu) chooses to
put into the conversation, unlike a tool, which the model decides to call. Each
recipe is a resource at chef://recipes/{recipe_id}, read as Markdown, and the
user's own recipes are listed so they show up in the attachment menu.
"""

from collections.abc import Callable

from mcp.server.fastmcp.exceptions import ResourceError
from mcp.types import Resource
from pydantic import AnyUrl

from chef_backend.mcp_server.auth import current_uid
from chef_backend.mcp_server.base import ChefMCP
from chef_backend.recipes.service import RecipeService, Scope
from chef_backend.recipes.views import RecipeView, short_summary

RECIPE_URI_TEMPLATE = "chef://recipes/{recipe_id}"
# The attachment menu lists the user's newest recipes; older ones stay reachable by id.
MAX_LISTED_RECIPES = 50


def register_resources(mcp: ChefMCP, recipe_service: Callable[[], RecipeService]) -> None:
    """Add the recipe resources to `mcp`. Call `recipe_service()` inside, per call."""

    @mcp.resource(
        uri=RECIPE_URI_TEMPLATE,
        name="recipe",
        description="One Chef recipe in full: ingredients, steps and tips, as Markdown.",
        mime_type="text/markdown",
    )
    def recipe_resource(recipe_id: str) -> str:
        uid = current_uid()
        recipe_details = recipe_service().get(uid=uid, recipe_id=recipe_id)
        if recipe_details is None:
            raise ResourceError("recipe_not_found")
        return recipe_markdown(RecipeView.of(recipe=recipe_details, viewer_uid=uid))

    @mcp.resource_lister()
    def recipes() -> list[Resource]:
        return [
            Resource(
                uri=AnyUrl(RECIPE_URI_TEMPLATE.format(recipe_id=recipe.id)),
                name=recipe.title,
                mimeType="text/markdown",
                description=short_summary(recipe.summary) or None,
            )
            for recipe in recipe_service().search(current_uid(), "", Scope.MINE, MAX_LISTED_RECIPES)
        ]


def recipe_markdown(view: RecipeView) -> str:
    """The recipe as Markdown; empty parts and sections are left out."""
    blocks = [f"# {view.title}"]
    if view.summary:
        blocks.append(view.summary)

    facts = [
        ("Servings", view.servings),
        ("Prep", view.prep_time),
        ("Cook", view.cooking_time),
        ("Difficulty", view.difficulty),
    ]
    meta: list[str] = []
    if fact_line := " · ".join(f"{label}: {value}" for label, value in facts if value):
        meta.append(f"- {fact_line}")
    if view.tags:
        meta.append(f"- Tags: {', '.join(view.tags)}")
    if meta:
        blocks.append("\n".join(meta))

    if view.ingredients:
        lines = (
            "- " + " ".join(part for part in (i.quantity, i.unit, i.name) if part)
            for i in view.ingredients
        )
        blocks.append("## Ingredients\n" + "\n".join(lines))
    if view.instructions:
        steps = (f"{n}. {step}" for n, step in enumerate(view.instructions, start=1))
        blocks.append("## Instructions\n" + "\n".join(steps))
    if view.tips_and_tricks:
        blocks.append(f"## Tips\n{view.tips_and_tricks}")

    return "\n\n".join(blocks) + "\n"
