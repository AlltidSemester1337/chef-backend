"""Argument completion: suggestions while the user fills in a prompt or resource URI.

A client sends completion/complete with a reference (a prompt, or a resource
template), the argument's name and what the user has typed so far; the server
answers with up to 100 values. A value replaces the whole argument, and there is no
separate display label, so suggestions are the values themselves.
"""

import re
from collections.abc import Callable

from mcp.types import (
    Completion,
    CompletionArgument,
    CompletionContext,
    PromptReference,
    ResourceTemplateReference,
)

from chef_backend.mcp_server.auth import current_uid
from chef_backend.mcp_server.base import ChefMCP
from chef_backend.mcp_server.resources import RECIPE_URI_TEMPLATE
from chef_backend.recipes.service import MAX_QUERY_CHARS, RecipeService, Scope

MAX_SUGGESTIONS = 20
# How many matches are counted for `total`; a larger total is reported as this.
MAX_COUNTED_MATCHES = 1000
DAYS = [str(day) for day in range(1, 8)]


def register_completions(mcp: ChefMCP, recipe_service: Callable[[], RecipeService]) -> None:
    """Add the completion handler to `mcp`. Call `recipe_service()` inside, per call."""

    @mcp.completion()
    async def complete(
        ref: PromptReference | ResourceTemplateReference,
        argument: CompletionArgument,
        context: CompletionContext | None,
    ) -> Completion | None:
        match ref, argument.name:
            case PromptReference(name="shopping_list"), "recipe_ids":
                return capped(recipe_id_lists(recipe_service(), argument.value))
            case ResourceTemplateReference(uri=uri), "recipe_id" if uri == RECIPE_URI_TEMPLATE:
                return capped(matching_ids(recipe_service(), argument.value))
            case PromptReference(name="plan_week"), "days":
                return capped([day for day in DAYS if day.startswith(argument.value)])
            case _:
                return None


def recipe_id_lists(service: RecipeService, value: str) -> list[str]:
    """Complete the last id in a list like "id1, id2, ide", keeping the ids before it."""
    *before, fragment = re.split(r"[,\s]+", value)
    typed = [recipe_id for recipe_id in before if recipe_id]
    return [
        ", ".join([*typed, recipe_id])
        for recipe_id in matching_ids(service, fragment)
        if recipe_id not in typed
    ]


def matching_ids(service: RecipeService, query: str) -> list[str]:
    """Ids of the recipes the user may see that match `query`, newest first."""
    if len(query) > MAX_QUERY_CHARS:
        return []
    found = service.search(current_uid(), query, Scope.ALL, MAX_COUNTED_MATCHES)
    return [recipe.id for recipe in found]


def capped(values: list[str]) -> Completion:
    if len(values) <= MAX_SUGGESTIONS:
        return Completion(values=values)
    return Completion(values=values[:MAX_SUGGESTIONS], total=len(values), hasMore=True)
