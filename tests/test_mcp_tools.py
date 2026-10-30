"""Spec for the MCP recipe tools (chef_backend.mcp_server.tools).

Three read-only tools on top of RecipeService:
- search_recipes(query="", scope="all", limit=10) -> {"recipes": [summary, ...]}
- get_recipe(recipe_id) -> the full recipe
- list_recipe_variants(recipe_id) -> {"variants": [variant, ...]}

The caller is the user in the access token, never an argument. Output is the same
camelCase views as the HTTP API (isMine, never an owner uid). A recipe the user may
not see, or that does not exist, is a tool error "recipe_not_found" (no oracle).
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_support import (
    BASE_URL,
    ME,
    OTHER,
    FakeMcpTokenVerifier,
    FakeRecipeRepository,
    McpClient,
)

from chef_backend.main import create_app
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import RecipeService

TOOL_NAMES = {"search_recipes", "get_recipe", "list_recipe_variants"}


@pytest.fixture
def verifier() -> FakeMcpTokenVerifier:
    return FakeMcpTokenVerifier()


@pytest.fixture
def mcp(verifier: FakeMcpTokenVerifier) -> Iterator[McpClient]:
    app = create_app()
    app.dependency_overrides[get_mcp_token_verifier] = lambda: verifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(FakeRecipeRepository())
    with TestClient(app, base_url=BASE_URL) as http:
        yield McpClient(http, verifier.issue())


def ok(result: dict[str, Any]) -> Any:
    assert result["isError"] is False, result["content"]
    return result["structuredContent"]


def error_text(result: dict[str, Any]) -> str:
    assert result["isError"] is True, result
    return " ".join(block["text"] for block in result["content"])


# --- The tool list: what the model sees ----------------------------------------


def test_exactly_the_three_recipe_tools_are_offered(mcp: McpClient) -> None:
    assert {tool["name"] for tool in mcp.list_tools()} == TOOL_NAMES


def test_every_tool_is_described_for_the_model(mcp: McpClient) -> None:
    for tool in mcp.list_tools():
        assert len(tool.get("description") or "") >= 40, tool["name"]
        for name, schema in tool["inputSchema"].get("properties", {}).items():
            assert schema.get("description"), f"{tool['name']}.{name} has no description"


def test_every_tool_is_marked_read_only(mcp: McpClient) -> None:
    for tool in mcp.list_tools():
        annotations = tool.get("annotations") or {}
        assert annotations.get("readOnlyHint") is True, tool["name"]
        assert annotations.get("openWorldHint") is False, tool["name"]


def test_search_parameters_have_defaults_and_bounds(mcp: McpClient) -> None:
    [search] = [t for t in mcp.list_tools() if t["name"] == "search_recipes"]
    properties = search["inputSchema"]["properties"]

    assert search["inputSchema"].get("required", []) == []
    assert properties["query"]["default"] == ""
    assert properties["scope"]["default"] == "all"
    assert set(properties["scope"]["enum"]) == {"mine", "community", "all"}
    assert (properties["limit"]["default"], properties["limit"]["minimum"]) == (10, 1)
    assert properties["limit"]["maximum"] == 50


def test_recipe_id_is_required(mcp: McpClient) -> None:
    for tool in mcp.list_tools():
        if tool["name"] != "search_recipes":
            assert tool["inputSchema"]["required"] == ["recipe_id"], tool["name"]


# --- search_recipes ------------------------------------------------------------


def test_search_defaults_to_own_and_community_newest_first(mcp: McpClient) -> None:
    found = ok(mcp.call("search_recipes"))

    assert [r["id"] for r in found["recipes"]] == ["theirs-shared", "mine"]


def test_search_hits_are_summaries_with_is_mine(mcp: McpClient) -> None:
    [hit] = ok(mcp.call("search_recipes", scope="mine"))["recipes"]

    assert (hit["id"], hit["title"], hit["isMine"]) == ("mine", "Title mine", True)
    assert "ingredients" not in hit


def test_search_passes_query_scope_and_limit(mcp: McpClient) -> None:
    def ids(**arguments: Any) -> list[str]:
        return [r["id"] for r in ok(mcp.call("search_recipes", **arguments))["recipes"]]

    assert ids(scope="community") == ["theirs-shared"]
    assert ids(query="title mine") == ["mine"]
    assert ids(limit=1) == ["theirs-shared"]


@pytest.mark.parametrize(
    "arguments",
    [{"scope": "everyone"}, {"limit": 0}, {"limit": 51}, {"query": "x" * 201}],
    ids=["unknown scope", "limit 0", "limit too high", "query too long"],
)
def test_invalid_search_arguments_are_tool_errors(
    mcp: McpClient, arguments: dict[str, Any]
) -> None:
    error_text(mcp.call("search_recipes", **arguments))


def test_search_acts_for_the_user_in_the_token(
    mcp: McpClient, verifier: FakeMcpTokenVerifier
) -> None:
    other = McpClient(mcp._http, verifier.issue("other-token", uid=OTHER))  # pyright: ignore[reportPrivateUsage]

    found = ok(other.call("search_recipes", scope="mine"))

    assert {r["id"] for r in found["recipes"]} == {"theirs-shared", "theirs-private"}


# --- get_recipe ----------------------------------------------------------------


def test_get_recipe_returns_the_full_recipe(mcp: McpClient) -> None:
    found = ok(mcp.call("get_recipe", recipe_id="theirs-shared"))

    assert (found["id"], found["isMine"]) == ("theirs-shared", False)
    assert found["ingredients"] == [{"name": "onion", "quantity": "1", "unit": "pc"}]
    assert found["instructions"] == ["Chop."]


@pytest.mark.parametrize("recipe_id", ["theirs-private", "does-not-exist", "a/variants/b"])
def test_invisible_and_missing_recipes_are_the_same_error(mcp: McpClient, recipe_id: str) -> None:
    for tool in ["get_recipe", "list_recipe_variants"]:
        assert "recipe_not_found" in error_text(mcp.call(tool, recipe_id=recipe_id))


# --- list_recipe_variants ------------------------------------------------------


def test_list_recipe_variants(mcp: McpClient) -> None:
    found = ok(mcp.call("list_recipe_variants", recipe_id="mine"))

    [variant] = found["variants"]
    assert (variant["id"], variant["label"], variant["isPinned"]) == ("v1", "Mild", True)


# --- Never leak an owner -------------------------------------------------------


def test_no_tool_output_contains_an_owner_uid(mcp: McpClient) -> None:
    results = [
        mcp.call("search_recipes"),
        mcp.call("get_recipe", recipe_id="theirs-shared"),
        mcp.call("list_recipe_variants", recipe_id="theirs-shared"),
    ]
    for result in results:
        text = str(result)
        assert OTHER not in text and ME not in text
        assert "'uid'" not in text
