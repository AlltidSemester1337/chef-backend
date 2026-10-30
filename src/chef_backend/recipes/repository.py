# The Firestore client is only partially typed; results are annotated where they land.
# pyright: reportUnknownMemberType=false
"""Reading recipes from Firestore.

The Admin SDK bypasses the security rules, so this layer only reads; deciding who
may see what is the service's job (chef_backend.recipes.access).
"""

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Protocol, cast

from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot

from chef_backend.recipes.models import Recipe, RecipeVariant, parse_recipe, parse_variant

RECIPES_COLLECTION = "recipes"
VARIANTS_COLLECTION = "variants"


class RecipeRepository(Protocol):
    def list_by_owner(self, uid: str) -> list[Recipe]:
        """The user's own recipes, newest `updatedAt` first."""
        ...

    def list_community(self) -> list[Recipe]:
        """Recipes shared with the community (`isFavourite == true`), newest first."""
        ...

    def get(self, recipe_id: str) -> Recipe | None:
        """One recipe, or None if it does not exist or is malformed."""
        ...

    def list_variants(self, recipe_id: str) -> list[RecipeVariant]:
        """The recipe's variants, oldest `createdAt` first."""
        ...


class FirestoreRecipeRepository:
    OLDEST = datetime.min.replace(tzinfo=UTC)

    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def _filter_and_sort_recipes(self, snapshots: Iterable[DocumentSnapshot]) -> list[Recipe]:
        recipes: list[Recipe] = []
        for snapshot in snapshots:
            recipe_data_dict = snapshot.to_dict()
            if not recipe_data_dict:
                logging.getLogger(__name__).warning("Recipe data empty: %s", snapshot.id)
                continue
            recipe = parse_recipe(doc_id=snapshot.id, data=recipe_data_dict)
            if recipe:
                recipes.append(recipe)
            else:
                logging.getLogger(__name__).warning("Failed to parse recipe: %s", snapshot.id)
        recipes.sort(key=lambda r: r.updated_at or self.OLDEST, reverse=True)
        return recipes

    def list_by_owner(self, uid: str) -> list[Recipe]:
        query = self._client.collection(RECIPES_COLLECTION).where(
            filter=firestore.FieldFilter("uid", "==", uid)
        )
        snapshots: Iterable[DocumentSnapshot] = query.stream()
        return self._filter_and_sort_recipes(snapshots)

    def list_community(self) -> list[Recipe]:
        query = self._client.collection(RECIPES_COLLECTION).where(
            filter=firestore.FieldFilter("isFavourite", "==", True)
        )
        snapshots: Iterable[DocumentSnapshot] = query.stream()
        return self._filter_and_sort_recipes(snapshots)

    def get(self, recipe_id: str) -> Recipe | None:
        recipe_id_match: DocumentSnapshot = (
            self._client.collection(RECIPES_COLLECTION).document(recipe_id).get()
        )
        if not recipe_id_match.exists:
            logging.getLogger(__name__).warning("Recipe %s not found", recipe_id)
            return None
        recipe_data = recipe_id_match.to_dict()
        if not recipe_data:
            logging.getLogger(__name__).warning("Recipe data empty: %s", recipe_id)
            return None
        recipe = parse_recipe(doc_id=recipe_id_match.id, data=recipe_data)
        if not recipe:
            logging.getLogger(__name__).warning("Failed to parse recipe: %s", recipe_id)
            return None
        return recipe

    def list_variants(self, recipe_id: str) -> list[RecipeVariant]:
        recipe_ref = self._client.collection(RECIPES_COLLECTION).document(recipe_id)
        snapshots = cast(
            Iterable[DocumentSnapshot],
            recipe_ref.collection(VARIANTS_COLLECTION).stream(),
        )
        recipe_variants: list[RecipeVariant] = []
        for snapshot in snapshots:
            recipe_variant_data_dict = snapshot.to_dict()
            if not recipe_variant_data_dict:
                logging.getLogger(__name__).warning("Recipe variant data empty: %s", snapshot.id)
                continue
            recipe_variant = parse_variant(doc_id=snapshot.id, data=recipe_variant_data_dict)
            if recipe_variant:
                recipe_variants.append(recipe_variant)
            else:
                logging.getLogger(__name__).warning(
                    "Failed to parse recipe variant: %s", snapshot.id
                )
        recipe_variants.sort(key=lambda r: r.created_at or self.OLDEST, reverse=False)
        return recipe_variants
