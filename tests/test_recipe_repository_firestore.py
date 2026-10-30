"""Spec for FirestoreRecipeRepository, against the Firestore emulator.

Skipped unless FIRESTORE_EMULATOR_HOST is set (see tests/test_quota_firestore.py
for how to start the emulator). The emulator is shared between tests, so each test
uses its own owner uid and recipe ids, and community assertions check containment.
"""

import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from google.cloud import firestore

from chef_backend.recipes.repository import (
    RECIPES_COLLECTION,
    VARIANTS_COLLECTION,
    FirestoreRecipeRepository,
)

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ, reason="needs the Firestore emulator"
)

T0 = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture
def client() -> firestore.Client:
    return firestore.Client(project="demo-chef-backend")


@pytest.fixture
def repo(client: firestore.Client) -> FirestoreRecipeRepository:
    return FirestoreRecipeRepository(client)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def add_recipe(client: firestore.Client, recipe_id: str, **fields: Any) -> str:
    doc: dict[str, Any] = {
        "id": recipe_id,
        "title": f"Recipe {recipe_id}",
        "ingredients": [{"name": "salt", "quantity": "1", "unit": "tsk"}],
        "instructions": ["Mix."],
        "tags": [],
        "isFavourite": False,
        "updatedAt": T0,
    } | fields
    client.collection(RECIPES_COLLECTION).document(recipe_id).set(doc)
    return recipe_id


def add_variant(client: firestore.Client, recipe_id: str, variant_id: str, **fields: Any) -> None:
    doc: dict[str, Any] = {
        "id": variant_id,
        "label": f"Variant {variant_id}",
        "title": "Variant",
        "createdAt": T0,
        "isPinned": False,
        "ingredients": [],
        "instructions": [],
    } | fields
    (
        client.collection(RECIPES_COLLECTION)
        .document(recipe_id)
        .collection(VARIANTS_COLLECTION)
        .document(variant_id)
        .set(doc)
    )


# --- list_by_owner -------------------------------------------------------------


def test_list_by_owner_returns_only_that_users_recipes_newest_first(
    client: firestore.Client, repo: FirestoreRecipeRepository
) -> None:
    me, other = new_id("me"), new_id("other")
    old = add_recipe(client, new_id("r"), uid=me, updatedAt=T0)
    new = add_recipe(client, new_id("r"), uid=me, updatedAt=T0 + timedelta(days=2))
    add_recipe(client, new_id("r"), uid=other)

    assert [r.id for r in repo.list_by_owner(me)] == [new, old]


def test_list_by_owner_for_a_user_without_recipes_is_empty(
    repo: FirestoreRecipeRepository,
) -> None:
    assert repo.list_by_owner(new_id("nobody")) == []


def test_malformed_documents_are_skipped_and_logged_by_id(
    client: firestore.Client, repo: FirestoreRecipeRepository, caplog: pytest.LogCaptureFixture
) -> None:
    me = new_id("me")
    good = add_recipe(client, new_id("r"), uid=me)
    bad = add_recipe(client, new_id("r"), uid=me, ingredients="PRIVATE free text")

    with caplog.at_level(logging.WARNING):
        recipes = repo.list_by_owner(me)

    assert [r.id for r in recipes] == [good]
    assert bad in caplog.text
    assert "PRIVATE" not in caplog.text


# --- list_community ------------------------------------------------------------


def test_list_community_returns_only_shared_recipes(
    client: firestore.Client, repo: FirestoreRecipeRepository
) -> None:
    shared = add_recipe(client, new_id("r"), uid=new_id("u"), isFavourite=True)
    private = add_recipe(client, new_id("r"), uid=new_id("u"), isFavourite=False)
    orphan = add_recipe(client, new_id("r"), isFavourite=True)  # no uid: still shared

    ids = {r.id for r in repo.list_community()}

    assert {shared, orphan} <= ids
    assert private not in ids


def test_list_community_is_newest_first(
    client: firestore.Client, repo: FirestoreRecipeRepository
) -> None:
    first = add_recipe(client, new_id("r"), isFavourite=True, updatedAt=T0 + timedelta(days=30))
    second = add_recipe(client, new_id("r"), isFavourite=True, updatedAt=T0 + timedelta(days=29))

    ids = [r.id for r in repo.list_community()]

    assert ids.index(first) < ids.index(second)


# --- get -----------------------------------------------------------------------


def test_get_returns_the_recipe(client: firestore.Client, repo: FirestoreRecipeRepository) -> None:
    recipe_id = add_recipe(client, new_id("r"), uid=new_id("u"), title="Pannkakor")

    recipe = repo.get(recipe_id)

    assert recipe is not None and recipe.title == "Pannkakor"


def test_get_missing_or_malformed_is_none(
    client: firestore.Client, repo: FirestoreRecipeRepository
) -> None:
    bad = add_recipe(client, new_id("r"), title="")

    assert repo.get(new_id("missing")) is None
    assert repo.get(bad) is None


# --- list_variants -------------------------------------------------------------


def test_list_variants_oldest_first_skipping_malformed(
    client: firestore.Client, repo: FirestoreRecipeRepository
) -> None:
    recipe_id = add_recipe(client, new_id("r"), uid=new_id("u"))
    add_variant(client, recipe_id, "later", createdAt=T0 + timedelta(hours=2))
    add_variant(client, recipe_id, "earlier", createdAt=T0 + timedelta(hours=1))
    add_variant(client, recipe_id, "broken", ingredients="not a list")

    assert [v.id for v in repo.list_variants(recipe_id)] == ["earlier", "later"]


def test_list_variants_of_a_recipe_without_variants_is_empty(
    client: firestore.Client, repo: FirestoreRecipeRepository
) -> None:
    assert repo.list_variants(add_recipe(client, new_id("r"))) == []
