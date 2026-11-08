"""FirestoreRecipeStore against the Firestore emulator: the document as the apps read it.

Skipped unless FIRESTORE_EMULATOR_HOST is set (see tests/test_quota_firestore.py).
"""

import os
import uuid
from datetime import UTC, datetime

import pytest
from google.cloud import firestore

from chef_backend.recipes.models import parse_recipe, parse_variant
from chef_backend.recipes.repository import (
    RECIPES_COLLECTION,
    VARIANTS_COLLECTION,
    FirestoreRecipeRepository,
)
from chef_backend.recipes.saving import FirestoreRecipeStore, RecipeDraft, VariantDraft

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ, reason="needs the Firestore emulator"
)

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def client() -> firestore.Client:
    return firestore.Client(project="demo-chef-backend")


def draft() -> RecipeDraft:
    return RecipeDraft.model_validate(
        {
            "title": "Toast",
            "ingredients": [{"name": "bread"}],
            "instructions": ["Toast the bread."],
        }
    )


def test_a_created_recipe_is_stored_and_readable_as_the_apps_would(
    client: firestore.Client,
) -> None:
    uid = f"cook-{uuid.uuid4()}"
    store = FirestoreRecipeStore(client)

    created = store.create(uid, draft(), NOW)

    snapshot = client.collection(RECIPES_COLLECTION).document(created.id).get()
    data = snapshot.to_dict()
    assert data is not None and data["id"] == created.id and data["uid"] == uid
    assert data["isFavourite"] is True and data["updatedAt"] == NOW
    assert parse_recipe(created.id, data) is not None
    assert [r.id for r in FirestoreRecipeRepository(client).list_by_owner(uid)] == [created.id]


def test_the_image_url_is_added_afterwards(client: firestore.Client) -> None:
    store = FirestoreRecipeStore(client)
    created = store.create(f"cook-{uuid.uuid4()}", draft(), NOW)

    store.set_image_url(created.id, "https://images.example/a.jpg")

    recipe = FirestoreRecipeRepository(client).get(created.id)
    assert recipe is not None and recipe.image_url == "https://images.example/a.jpg"


def test_a_created_variant_is_stored_under_its_recipe(client: firestore.Client) -> None:
    store = FirestoreRecipeStore(client)
    recipe = store.create(f"cook-{uuid.uuid4()}", draft(), NOW)
    variant = VariantDraft.model_validate(
        {
            "label": "Buttered",
            "title": "Buttered toast",
            "ingredients": [{"name": "bread"}, {"name": "butter"}],
            "instructions": ["Toast the bread.", "Butter it."],
        }
    )

    created = store.create_variant(recipe.id, variant, NOW)

    snapshot = (
        client.collection(RECIPES_COLLECTION)
        .document(recipe.id)
        .collection(VARIANTS_COLLECTION)
        .document(created.id)
        .get()
    )
    data = snapshot.to_dict()
    assert data is not None and data["id"] == created.id and data["createdAt"] == NOW
    assert parse_variant(created.id, data) is not None
    [listed] = FirestoreRecipeRepository(client).list_variants(recipe.id)
    assert (listed.id, listed.label, listed.is_pinned) == (created.id, "Buttered", False)
