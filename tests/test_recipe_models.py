"""Spec for parse_recipe / parse_variant: Firestore documents to models, defensively.

Fixtures mirror what the Android app writes (chef/.ai/firestore-schema.md). A bad
document must never break a list: the parser returns None and the repository skips
it. Lenient where the apps are (missing or null optional fields), strict about
types (the security rules cannot check list elements, so anything can be in there).
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from chef_backend.recipes.models import parse_recipe, parse_variant

UPDATED = datetime(2026, 10, 20, 18, 30, tzinfo=UTC)


def recipe_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "id": "r1",
        "uid": "owner-1",
        "title": "Chana masala",
        "summary": "Kryddig kikärtsgryta.",
        "difficulty": "EASY",
        "cookingTime": "35 minutes",
        "prepTime": "15 minutes",
        "servings": "4 servings",
        "imageUrl": "https://firebasestorage.googleapis.com/v0/b/b/o/recipes%2Fa.jpg?alt=media",
        "videoUrl": None,
        "isFavourite": True,
        "copyId": None,
        "tipsAndTricks": "- Rosta kryddorna först.",
        "tags": ["vegetarian", "indian", "under 1 hour"],
        "ingredients": [
            {"name": "kikärtor", "quantity": "800", "unit": "g"},
            {"name": "garam masala", "quantity": "2", "unit": "tsk"},
        ],
        "instructions": ["Fräs löken.", "Tillsätt kryddorna.", "Koka i 20 minuter."],
        "nutrientsPerServing": [{"name": "protein", "quantity": "14", "unit": "g"}],
        "updatedAt": UPDATED,
    }
    return doc | overrides


def without(doc: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k not in keys}


# --- Recipes: the happy path ---------------------------------------------------


def test_full_document_is_parsed_into_snake_case_attributes() -> None:
    recipe = parse_recipe("r1", recipe_doc())

    assert recipe is not None
    assert (recipe.id, recipe.uid, recipe.title) == ("r1", "owner-1", "Chana masala")
    assert recipe.summary == "Kryddig kikärtsgryta."
    assert recipe.difficulty == "EASY"
    assert (recipe.prep_time, recipe.cooking_time, recipe.servings) == (
        "15 minutes",
        "35 minutes",
        "4 servings",
    )
    assert recipe.image_url is not None and recipe.image_url.startswith("https://")
    assert recipe.video_url is None
    assert recipe.is_favourite is True
    assert recipe.copy_id is None
    assert recipe.tips_and_tricks == "- Rosta kryddorna först."
    assert recipe.tags == ["vegetarian", "indian", "under 1 hour"]
    assert [(i.name, i.quantity, i.unit) for i in recipe.ingredients] == [
        ("kikärtor", "800", "g"),
        ("garam masala", "2", "tsk"),
    ]
    assert recipe.instructions[0] == "Fräs löken."
    assert [(n.name, n.quantity, n.unit) for n in recipe.nutrients_per_serving] == [
        ("protein", "14", "g")
    ]
    assert recipe.updated_at == UPDATED


def test_the_document_id_wins_over_the_id_field() -> None:
    recipe = parse_recipe("doc-id", recipe_doc(id="something-else"))

    assert recipe is not None and recipe.id == "doc-id"


def test_orphaned_recipe_has_no_owner() -> None:
    # Orphaned recipes have no uid field at all (never null).
    recipe = parse_recipe("r1", without(recipe_doc(), "uid"))

    assert recipe is not None and recipe.uid is None


OPTIONAL_FIELDS = [
    "summary",
    "difficulty",
    "cookingTime",
    "prepTime",
    "servings",
    "imageUrl",
    "videoUrl",
    "copyId",
    "tipsAndTricks",
    "tags",
    "ingredients",
    "instructions",
    "nutrientsPerServing",
    "updatedAt",
    "isFavourite",
]


@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_optional_fields_may_be_missing(field: str) -> None:
    assert parse_recipe("r1", without(recipe_doc(), field)) is not None


@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_optional_fields_may_be_null(field: str) -> None:
    # The Android SDK writes null for unset properties.
    assert parse_recipe("r1", recipe_doc(**{field: None})) is not None


def test_missing_optional_fields_get_empty_defaults() -> None:
    minimal = {"title": "Toast", "updatedAt": UPDATED}

    recipe = parse_recipe("r9", minimal)

    assert recipe is not None
    assert (recipe.summary, recipe.prep_time, recipe.cooking_time, recipe.servings) == (
        "",
        "",
        "",
        "",
    )
    assert recipe.tips_and_tricks == ""
    assert (recipe.tags, recipe.ingredients, recipe.instructions) == ([], [], [])
    assert recipe.nutrients_per_serving == []
    assert (recipe.image_url, recipe.video_url, recipe.copy_id, recipe.difficulty) == (
        None,
        None,
        None,
        None,
    )
    assert recipe.is_favourite is False


def test_null_optional_fields_get_the_same_defaults_as_missing_ones() -> None:
    recipe = parse_recipe("r1", recipe_doc(summary=None, tags=None, ingredients=None))

    assert recipe is not None
    assert (recipe.summary, recipe.tags, recipe.ingredients) == ("", [], [])


def test_ingredient_without_quantity_or_unit_is_kept() -> None:
    recipe = parse_recipe("r1", recipe_doc(ingredients=[{"name": "salt"}]))

    assert recipe is not None
    assert [(i.name, i.quantity, i.unit) for i in recipe.ingredients] == [("salt", "", "")]


def test_unknown_fields_are_ignored() -> None:
    # Forward compatible: a field added by a newer app version must not break reads.
    assert parse_recipe("r1", recipe_doc(embedding=[0.1, 0.2], newField="x")) is not None


# --- Recipes: malformed documents return None ----------------------------------

MALFORMED: dict[str, dict[str, Any]] = {
    "missing title": without(recipe_doc(), "title"),
    "empty title": recipe_doc(title=""),
    "title not a string": recipe_doc(title=42),
    "ingredients not a list": recipe_doc(ingredients="800 g kikärtor"),
    "ingredient not an object": recipe_doc(ingredients=["kikärtor"]),
    "ingredient without name": recipe_doc(ingredients=[{"quantity": "1"}]),
    "ingredient quantity a number": recipe_doc(ingredients=[{"name": "ägg", "quantity": 2}]),
    "tags not a list": recipe_doc(tags="vegetarian,indian"),
    "tag not a string": recipe_doc(tags=["vegetarian", 7]),
    "instructions not a list": recipe_doc(instructions="Fräs löken."),
    "isFavourite a string": recipe_doc(isFavourite="true"),
    "unknown difficulty": recipe_doc(difficulty="easy"),
    "updatedAt a string": recipe_doc(updatedAt="2026-10-20T18:30:00Z"),
    "uid not a string": recipe_doc(uid=123),
}


@pytest.mark.parametrize("doc", MALFORMED.values(), ids=MALFORMED.keys())
def test_malformed_documents_return_none(doc: dict[str, Any]) -> None:
    assert parse_recipe("r1", doc) is None


# --- Variants ------------------------------------------------------------------

CREATED = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def variant_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "id": "v1",
        "label": "Utan chili",
        "createdAt": CREATED,
        "isPinned": True,
        "title": "Mild chana masala",
        "summary": "Utan chili.",
        "servings": "4 servings",
        "prepTime": "15 minutes",
        "cookingTime": "35 minutes",
        "nutrientsPerServing": None,
        "ingredients": [{"name": "kikärtor", "quantity": "800", "unit": "g"}],
        "difficulty": "EASY",
        "instructions": ["Fräs löken."],
        "tipsAndTricks": None,
    }
    return doc | overrides


def test_full_variant_is_parsed() -> None:
    variant = parse_variant("v1", variant_doc())

    assert variant is not None
    assert (variant.id, variant.label, variant.title) == ("v1", "Utan chili", "Mild chana masala")
    assert variant.created_at == CREATED
    assert variant.is_pinned is True
    assert [i.name for i in variant.ingredients] == ["kikärtor"]
    assert variant.instructions == ["Fräs löken."]
    assert variant.nutrients_per_serving == []
    assert variant.tips_and_tricks == ""


def test_minimal_variant_is_parsed() -> None:
    variant = parse_variant("v2", {"label": "Halv sats", "title": "Liten", "createdAt": CREATED})

    assert variant is not None
    assert variant.is_pinned is False
    assert (variant.ingredients, variant.instructions) == ([], [])


MALFORMED_VARIANTS: dict[str, dict[str, Any]] = {
    "missing label": without(variant_doc(), "label"),
    "missing title": without(variant_doc(), "title"),
    "ingredients not a list": variant_doc(ingredients="kikärtor"),
    "isPinned a string": variant_doc(isPinned="yes"),
}


@pytest.mark.parametrize("doc", MALFORMED_VARIANTS.values(), ids=MALFORMED_VARIANTS.keys())
def test_malformed_variants_return_none(doc: dict[str, Any]) -> None:
    assert parse_variant("v1", doc) is None
