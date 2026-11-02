"""Spec for the MCP prompts (chef_backend.mcp_server.prompts).

Task 1, plan_week(days="5", wishes=""): a dinner plan for the coming days from the
user's Chef recipes. Rendering gives exactly one user message (text) that asks the
model to plan `days` dinners and names the tools to use (search_recipes, and
get_recipe for details). When wishes are given, the message has a line
"Wishes: <the wishes>"; without wishes there is no such line. `days` must be a
whole number from 1 to 7, otherwise a PromptError with the message
"days must be a whole number from 1 to 7".

The prompt is listed with a title, a description and both arguments, optional
and described.

Task 2, shopping_list(recipe_ids, servings): one combined shopping list for a few
recipes. Both arguments are required. recipe_ids: 1 to 10 ids separated by commas and/or spaces,
duplicates ignored (first occurrence kept). Rendering gives one user message per
recipe that embeds the recipe as a resource (uri chef://recipes/{id}, mimeType
text/markdown, the same Markdown as the recipe resource), followed by one user
text message asking for the combined shopping list, with a line
"Servings: <servings>": every recipe is scaled to it, since the recipes' own
servings differ. Errors (PromptError):
- no ids or more than 10: "recipe_ids must list 1 to 10 recipe ids"
- a recipe the user may not see, or that does not exist: "recipe_not_found"
- servings not a whole number from 1 to 20: "servings must be a whole number from 1 to 20"
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_support import BASE_URL, FakeMcpTokenVerifier, FakeRecipeRepository, McpClient

from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.mcp_server.resources import recipe_markdown
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService
from chef_backend.recipes.views import RecipeView


@pytest.fixture
def mcp() -> Iterator[McpClient]:
    verifier = FakeMcpTokenVerifier()
    app = create_app()
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(FakeRecipeRepository())
    with TestClient(app, base_url=BASE_URL) as http:
        yield McpClient(http, verifier.issue())


def prompt(mcp: McpClient, name: str) -> dict[str, Any]:
    [found] = [p for p in mcp.list_prompts() if p["name"] == name]
    return found


def only_text(response: dict[str, Any]) -> str:
    assert "result" in response, response
    [message] = response["result"]["messages"]
    assert message["role"] == "user" and message["content"]["type"] == "text"
    return message["content"]["text"]


# --- Task 1: plan_week ------------------------------------------------------------------


def test_plan_week_is_listed_with_two_optional_described_arguments(mcp: McpClient) -> None:
    plan = prompt(mcp, "plan_week")

    assert plan.get("title")
    assert len(plan.get("description") or "") >= 30
    arguments = {a["name"]: a for a in plan["arguments"]}
    assert set(arguments) == {"days", "wishes"}
    for argument in arguments.values():
        assert argument.get("required") is not True and argument.get("description")


def test_plan_week_asks_for_the_given_number_of_dinners_with_the_tools(mcp: McpClient) -> None:
    text = only_text(mcp.get_prompt("plan_week", days="3", wishes="vegetarian, no fish"))

    assert "3" in text
    assert "search_recipes" in text and "get_recipe" in text
    assert "\nWishes: vegetarian, no fish" in "\n" + text


def test_plan_week_defaults_to_five_days_without_wishes(mcp: McpClient) -> None:
    text = only_text(mcp.get_prompt("plan_week"))

    assert "5" in text
    assert "Wishes:" not in text


@pytest.mark.parametrize("days", ["1", "7"])
def test_plan_week_accepts_one_to_seven_days_inclusive(mcp: McpClient, days: str) -> None:
    assert days in only_text(mcp.get_prompt("plan_week", days=days))


@pytest.mark.parametrize("days", ["0", "8", "many", "2.5", "-1"])
def test_plan_week_refuses_days_outside_one_to_seven(mcp: McpClient, days: str) -> None:
    response = mcp.get_prompt("plan_week", days=days)

    assert response["error"]["message"] == "days must be a whole number from 1 to 7"


# --- Task 2: shopping_list -------------------------------------------------------------


def messages(response: dict[str, Any]) -> list[dict[str, Any]]:
    assert "result" in response, response
    return response["result"]["messages"]


def embedded_uris(found: list[dict[str, Any]]) -> list[str]:
    return [m["content"]["resource"]["uri"] for m in found if m["content"]["type"] == "resource"]


def test_shopping_list_is_listed_with_its_arguments(mcp: McpClient) -> None:
    shopping = prompt(mcp, "shopping_list")

    assert shopping.get("title") and len(shopping.get("description") or "") >= 30
    arguments = {a["name"]: a for a in shopping["arguments"]}
    assert arguments["recipe_ids"].get("required") is True
    assert arguments["servings"].get("required") is True
    assert all(a.get("description") for a in arguments.values())


def test_each_recipe_is_embedded_as_a_markdown_resource(mcp: McpClient) -> None:
    found = messages(
        mcp.get_prompt("shopping_list", recipe_ids="mine, theirs-shared", servings="4")
    )

    assert [m["role"] for m in found] == ["user", "user", "user"]
    assert embedded_uris(found) == ["chef://recipes/mine", "chef://recipes/theirs-shared"]
    first = found[0]["content"]["resource"]
    expected = recipe_markdown(RecipeView.of(FakeRecipeRepository().recipes[0], "me-uid"))
    assert (first["mimeType"], first["text"]) == ("text/markdown", expected)


def test_the_last_message_asks_for_the_combined_list(mcp: McpClient) -> None:
    found = messages(mcp.get_prompt("shopping_list", recipe_ids="mine", servings="2"))

    last = found[-1]["content"]
    assert last["type"] == "text" and "shopping list" in last["text"].lower()
    assert "\nServings: 2\n" in "\n" + last["text"]


def test_servings_are_required(mcp: McpClient) -> None:
    message = mcp.get_prompt("shopping_list", recipe_ids="mine")["error"]["message"]

    assert message.startswith("Missing required arguments:") and "servings" in message


def test_ids_may_be_separated_by_commas_or_spaces_and_repeat(mcp: McpClient) -> None:
    found = messages(
        mcp.get_prompt("shopping_list", recipe_ids=" mine,theirs-shared  mine ", servings="4")
    )

    assert embedded_uris(found) == ["chef://recipes/mine", "chef://recipes/theirs-shared"]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"recipe_ids": " , ", "servings": "4"}, "recipe_ids must list 1 to 10 recipe ids"),
        (
            {"recipe_ids": ",".join(f"r{n}" for n in range(11)), "servings": "4"},
            "recipe_ids must list 1 to 10 recipe ids",
        ),
        ({"recipe_ids": "mine, theirs-private", "servings": "4"}, "recipe_not_found"),
        ({"recipe_ids": "does-not-exist", "servings": "4"}, "recipe_not_found"),
        ({"recipe_ids": "mine", "servings": "0"}, "servings must be a whole number from 1 to 20"),
        (
            {"recipe_ids": "mine", "servings": "many"},
            "servings must be a whole number from 1 to 20",
        ),
    ],
    ids=["no ids", "too many", "invisible", "missing", "zero servings", "servings not a number"],
)
def test_shopping_list_refuses_bad_arguments(
    mcp: McpClient, arguments: dict[str, str], error: str
) -> None:
    assert mcp.get_prompt("shopping_list", **arguments)["error"]["message"] == error
