"""Spec for the MCP tool save_recipe_variant (chef_backend.mcp_server.write_tools).

save_recipe_variant(recipe_id, label, title, ingredients, instructions, summary="",
servings="", prep_time="", cooking_time="", difficulty=None, tips_and_tricks="",
nutrients_per_serving=[]) adds an adapted version of one of the user's own recipes
(e.g. label "Vegetarian" or "For two") via RecipeSaver.save_variant, and returns the
saved variant as list_recipe_variants shows it (VariantView). The variant is the
whole adapted recipe, not a diff: the app shows it instead of the original.

- No tags: variants have none.
- It needs the recipes:write scope; without it "insufficient_scope", nothing saved.
- Someone else's (visible) recipe: tool error "not_your_recipe" that points the model
  to save_recipe, which saves an adapted copy as the user's own recipe.
- A recipe the user may not see, or no such recipe: "recipe_not_found".
- Today's save budget (shared with save_recipe) used up: "daily_save_limit_reached".
- Invalid input (empty label, ...): "invalid_variant" naming the field, nothing saved.
- Annotations: not read-only, not destructive (the original is kept), not idempotent
  (each call adds a variant), open world: variants of a recipe shared with the
  community are visible to other users too.
"""

from typing import Any

import pytest
from mcp_support import ME, McpClient
from test_mcp_save_recipe import World, error_text

# The shared fixture; "as world" marks it as used on purpose.
from test_mcp_save_recipe import world as world

from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE

FOR_TWO: dict[str, Any] = {
    "recipe_id": "mine",
    "label": "For two",
    "title": "Chana masala for two",
    "servings": "2",
    "ingredients": [
        {"name": "chickpeas", "quantity": "400", "unit": "g"},
        {"name": "salt"},
    ],
    "instructions": ["Fry the onion.", "Add the spices.", "Simmer."],
}


def variant_tool(mcp: McpClient) -> dict[str, Any]:
    [tool] = [t for t in mcp.list_tools() if t["name"] == "save_recipe_variant"]
    return tool


def saved_variants(world: World, recipe_id: str = "mine") -> list[str]:
    return [v.label for v in world.repository.saved_variants.get(recipe_id, [])]


# --- What the model sees ----------------------------------------------------------------


def test_the_tool_is_an_open_world_non_destructive_write(world: World) -> None:
    annotations = variant_tool(world.writer()).get("annotations") or {}

    assert annotations.get("readOnlyHint") is False
    assert annotations.get("destructiveHint") is False
    assert annotations.get("idempotentHint") is False
    assert annotations.get("openWorldHint") is True


def test_the_description_says_when_to_use_save_recipe_instead(world: World) -> None:
    description = variant_tool(world.writer()).get("description") or ""

    assert len(description) >= 80
    assert "save_recipe" in description and "own" in description.lower()


def test_the_schema_has_a_label_but_no_tags(world: World) -> None:
    schema = variant_tool(world.writer())["inputSchema"]

    assert set(schema["required"]) == {
        "recipe_id",
        "label",
        "title",
        "ingredients",
        "instructions",
    }
    assert "tags" not in schema["properties"]
    for name, prop in schema["properties"].items():
        assert prop.get("description"), f"{name} has no description"


# --- Saving ------------------------------------------------------------------------------


def test_saving_returns_the_new_variant(world: World) -> None:
    result = world.writer().call("save_recipe_variant", **FOR_TWO)

    assert result["isError"] is False, result["content"]
    variant = result["structuredContent"]
    assert (variant["label"], variant["title"], variant["servings"]) == (
        "For two",
        "Chana masala for two",
        "2",
    )
    assert variant["isPinned"] is False and variant["id"]
    assert variant["ingredients"][1] == {"name": "salt", "quantity": "", "unit": ""}
    assert ME not in str(result)


def test_the_saved_variant_is_listed_with_the_recipe(world: World) -> None:
    writer = world.writer()
    saved_id = writer.call("save_recipe_variant", **FOR_TWO)["structuredContent"]["id"]

    listed = writer.call("list_recipe_variants", recipe_id="mine")["structuredContent"]

    assert [v["id"] for v in listed["variants"]][-1] == saved_id


def test_only_the_required_fields_are_enough(world: World) -> None:
    minimal = {k: FOR_TWO[k] for k in ("recipe_id", "label", "title", "ingredients")}
    minimal["instructions"] = FOR_TWO["instructions"]

    variant = world.writer().call("save_recipe_variant", **minimal)["structuredContent"]

    assert (variant["summary"], variant["difficulty"], variant["prepTime"]) == ("", None, "")


# --- Refusals ----------------------------------------------------------------------------


def test_without_the_write_scope_nothing_is_saved(world: World) -> None:
    reader = world.client(scopes=[RECIPES_READ_SCOPE])

    assert "insufficient_scope" in error_text(reader.call("save_recipe_variant", **FOR_TWO))
    assert saved_variants(world) == []


def test_someone_elses_recipe_points_to_save_recipe(world: World) -> None:
    text = error_text(
        world.writer().call("save_recipe_variant", **(FOR_TWO | {"recipe_id": "theirs-shared"}))
    )

    # One plain message the model can act on (not a tuple of strings).
    assert "not_your_recipe: " in text and "('" not in text and "save_recipe" in text
    assert saved_variants(world, "theirs-shared") == []


@pytest.mark.parametrize("recipe_id", ["theirs-private", "missing"])
def test_an_invisible_or_missing_recipe_is_not_found(world: World, recipe_id: str) -> None:
    text = error_text(
        world.writer().call("save_recipe_variant", **(FOR_TWO | {"recipe_id": recipe_id}))
    )

    assert "recipe_not_found" in text
    assert world.repository.saved_variants == {}


def test_variants_count_towards_the_daily_save_limit(world: World) -> None:
    writer = world.writer()
    writer.call("save_recipe_variant", **FOR_TWO)
    writer.call("save_recipe_variant", **FOR_TWO)

    text = error_text(writer.call("save_recipe_variant", **FOR_TWO))

    assert "daily_save_limit_reached" in text
    assert saved_variants(world) == ["For two", "For two"]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [({"label": ""}, "label"), ({"title": ""}, "title"), ({"ingredients": []}, "ingredients")],
    ids=["empty label", "empty title", "no ingredients"],
)
def test_an_invalid_variant_says_what_is_wrong(
    world: World, overrides: dict[str, Any], field: str
) -> None:
    text = error_text(world.writer().call("save_recipe_variant", **(FOR_TWO | overrides)))

    assert "invalid_variant" in text and field in text
    assert saved_variants(world) == []
