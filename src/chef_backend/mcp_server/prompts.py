"""The MCP prompts: ready-made requests the user picks from the client's prompt menu.

A prompt is a template the user chooses (often as a slash command), filled in with
a few arguments; the server turns it into messages for the model. Unlike a tool,
the model does not decide to use it, and unlike a resource, it is an instruction
rather than content. Prompt arguments always arrive as strings.
"""

from collections.abc import Callable
from typing import Annotated

from mcp.server.fastmcp.prompts.base import Message, UserMessage
from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl, Field

from chef_backend.mcp_server.auth import current_uid
from chef_backend.mcp_server.base import ChefMCP, PromptError
from chef_backend.mcp_server.resources import RECIPE_URI_TEMPLATE, recipe_markdown
from chef_backend.recipes.service import RecipeService
from chef_backend.recipes.views import RecipeView


def register_prompts(mcp: ChefMCP, recipe_service: Callable[[], RecipeService]) -> None:
    """Add the prompts to `mcp`. Call `recipe_service()` inside, per call."""

    @mcp.prompt(
        title="Plan the week's dinners",
        description="Plan dinners for the coming days from your Chef recipes, with community "
        "recipes for variety, and get a combined shopping list.",
    )
    def plan_week(
        days: Annotated[str, Field(description="How many dinners to plan, 1-7 (default 5).")] = "5",
        wishes: Annotated[
            str,
            Field(
                description="Optional preferences, e.g. 'vegetarian, under 30 minutes, "
                "no fish, 2 people'."
            ),
        ] = "",
    ) -> str:
        if not days.isdigit() or not 1 <= int(days) <= 7:
            raise PromptError("days must be a whole number from 1 to 7")
        lines = [f"Plan dinners for the next {days} days from my Chef recipes."]
        if wishes:
            lines.append(f"Wishes: {wishes}")
            lines.append("Treat these wishes as requirements, and say if one cannot be met.")
        lines.append(WEEK_PLAN_STEPS)
        return "\n".join(lines)

    @mcp.prompt(
        title="Shopping list for recipes",
        description="One combined shopping list for up to 10 of your Chef recipes, scaled to "
        "the number of servings you need.",
    )
    def shopping_list(
        recipe_ids: Annotated[
            str,
            Field(
                description="1 to 10 Chef recipe ids, separated by commas or spaces "
                "(ids are shown in search results and in recipe resources)."
            ),
        ],
        servings: Annotated[
            str, Field(description="How many servings to shop for, 1-20, e.g. '4'.")
        ],
    ) -> list[Message]:
        if not servings.isdigit() or not 1 <= int(servings) <= 20:
            raise PromptError("servings must be a whole number from 1 to 20")
        ids = list(dict.fromkeys(recipe_ids.replace(",", " ").split()))
        if not 1 <= len(ids) <= 10:
            raise PromptError("recipe_ids must list 1 to 10 recipe ids")

        uid = current_uid()
        messages: list[Message] = []
        for recipe_id in ids:
            recipe = recipe_service().get(uid=uid, recipe_id=recipe_id)
            if recipe is None:
                raise PromptError("recipe_not_found")
            messages.append(
                UserMessage(
                    content=EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=AnyUrl(RECIPE_URI_TEMPLATE.format(recipe_id=recipe_id)),
                            mimeType="text/markdown",
                            text=recipe_markdown(RecipeView.of(recipe=recipe, viewer_uid=uid)),
                        ),
                    )
                )
            )
        messages.append(
            UserMessage(
                content=f"Make one combined shopping list for the {len(ids)} recipes above.\n"
                f"Servings: {servings}\n{SHOPPING_LIST_STEPS}"
            )
        )
        return messages


SHOPPING_LIST_STEPS = """
1. Scale every recipe to the servings above from its own servings. If a recipe's
   servings are unclear (e.g. "1 loaf" or "6-8"), state the assumption you made.
2. Merge the same ingredient across recipes into one line with the total quantity,
   converting to one unit (metric) where they differ. Round up to amounts a shop sells.
3. Group the list by store section (produce, meat and fish, dairy, dry goods, spices,
   frozen, other). Put pantry staples (salt, pepper, oil, common spices) in a separate
   "Check at home" group instead of the main list.
4. Write each line as a checklist item: "- [ ] 800 g chickpeas", and after it, in brackets,
   which recipes use it when it is more than one.

Do not change or save any recipe.
Answer in the language I write to you in; if you cannot tell, use English."""


WEEK_PLAN_STEPS = """
1. Find candidates with search_recipes using 1 to 3 English keywords per search
   (e.g. "vegetarian", "lentil", "halloumi"). Use scope "mine" for my own recipes and
   scope "others" for community inspiration. Aim for a mix of both, mostly mine.
2. Pick candidates from title and summary, then call get_recipe only for those.
   Check the ingredients against any dietary wishes: tags can be missing or wrong.
   For one of my recipes that almost fits, check list_recipe_variants for an adapted version.
3. Vary the main ingredient across the days, but look for shared ingredients and for
   leftovers that can be reused (e.g. cook extra rice on day 1 for day 2), and say where.

Present the plan day by day: title, whether it is my own or a community recipe,
total time and a one-line summary. Then invite changes ("swap day 2", "something quicker")
and offer a combined shopping list: scaled to the number of people (ask if you do not
know it), duplicates merged, grouped by store section, pantry staples listed separately.

Do not save any recipe unless I ask you to.
Answer in the language I write to you in; if you cannot tell, use the language of my wishes."""
