"""RecipeService: the access policy applied to every read, and the search scopes.

The search itself is replaced by a pass-through here (it has its own spec in
test_recipe_search.py), so these tests only check which candidates it is given.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from chef_backend.recipes import service as service_module
from chef_backend.recipes.access import is_visible_to
from chef_backend.recipes.models import Recipe, RecipeVariant
from chef_backend.recipes.service import RecipeService, Scope

ME, OTHER = "me", "other"
T0 = datetime(2026, 10, 1, tzinfo=UTC)


def recipe(recipe_id: str, uid: str | None, *, shared: bool = False, day: int = 0) -> Recipe:
    return Recipe(
        id=recipe_id,
        uid=uid,
        title=f"Recipe {recipe_id}",
        isFavourite=shared,
        updatedAt=T0 + timedelta(days=day),
    )


MY_PRIVATE = recipe("my-private", ME, day=1)
MY_SHARED = recipe("my-shared", ME, shared=True, day=3)
OTHERS_PRIVATE = recipe("others-private", OTHER, day=4)
OTHERS_SHARED = recipe("others-shared", OTHER, shared=True, day=2)
ORPHAN_SHARED = recipe("orphan-shared", None, shared=True, day=5)
ORPHAN_PRIVATE = recipe("orphan-private", None, day=6)
EVERYTHING = [MY_PRIVATE, MY_SHARED, OTHERS_PRIVATE, OTHERS_SHARED, ORPHAN_SHARED, ORPHAN_PRIVATE]


def newest_first(recipes: Iterable[Recipe]) -> list[Recipe]:
    return sorted(recipes, key=lambda r: r.updated_at or T0, reverse=True)


@dataclass
class FakeRepository:
    recipes: list[Recipe] = field(default_factory=lambda: list(EVERYTHING))
    variant_reads: list[str] = field(default_factory=list[str])

    def list_by_owner(self, uid: str) -> list[Recipe]:
        return newest_first(r for r in self.recipes if r.uid == uid)

    def list_community(self) -> list[Recipe]:
        return newest_first(r for r in self.recipes if r.is_favourite)

    def get(self, recipe_id: str) -> Recipe | None:
        return next((r for r in self.recipes if r.id == recipe_id), None)

    def list_variants(self, recipe_id: str) -> list[RecipeVariant]:
        self.variant_reads.append(recipe_id)
        return [RecipeVariant(id="v1", label="Mild", title=f"{recipe_id} (mild)")]


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def service(repository: FakeRepository) -> RecipeService:
    return RecipeService(repository)


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the search with a pass-through that records its arguments."""
    calls: list[dict[str, Any]] = []

    def fake_search(recipes: Iterable[Recipe], query: str, limit: int) -> list[Recipe]:
        candidates = list(recipes)
        calls.append({"ids": [r.id for r in candidates], "query": query, "limit": limit})
        return candidates[:limit]

    monkeypatch.setattr(service_module, "search_recipes", fake_search)
    return calls


# --- The policy ----------------------------------------------------------------

VISIBLE_TO_ME = {"my-private", "my-shared", "others-shared", "orphan-shared"}


@pytest.mark.parametrize("candidate", EVERYTHING, ids=[r.id for r in EVERYTHING])
def test_a_user_sees_own_and_community_recipes_only(candidate: Recipe) -> None:
    assert is_visible_to(candidate, ME) == (candidate.id in VISIBLE_TO_ME)


# --- get -----------------------------------------------------------------------


@pytest.mark.parametrize("recipe_id", sorted(VISIBLE_TO_ME))
def test_get_returns_a_visible_recipe(service: RecipeService, recipe_id: str) -> None:
    found = service.get(ME, recipe_id)

    assert found is not None and found.id == recipe_id


@pytest.mark.parametrize("recipe_id", ["others-private", "orphan-private", "does-not-exist"])
def test_get_hides_invisible_and_missing_recipes_alike(
    service: RecipeService, recipe_id: str
) -> None:
    assert service.get(ME, recipe_id) is None


@pytest.mark.parametrize(
    "recipe_id",
    ["my-private/variants/v1", "", "..", "my private", "my-private\n", "x" * 129],
    ids=["path", "empty", "dots", "space", "newline", "too long"],
)
def test_ids_that_are_not_plain_document_ids_are_never_looked_up(
    service: RecipeService, repository: FakeRepository, recipe_id: str
) -> None:
    # A "/" would make document() address a subcollection document instead.
    repository.recipes.append(recipe(recipe_id, ME))

    assert service.get(ME, recipe_id) is None
    assert service.list_variants(ME, recipe_id) is None


def test_migrated_push_ids_and_auto_ids_are_accepted(
    service: RecipeService, repository: FakeRepository
) -> None:
    repository.recipes += [recipe("-NxY_a1B2c3D4e5F6g7H", ME), recipe("aB3dE5gH7jK9mN1pQ3sT", ME)]

    assert service.get(ME, "-NxY_a1B2c3D4e5F6g7H") is not None
    assert service.get(ME, "aB3dE5gH7jK9mN1pQ3sT") is not None


# --- list_variants -------------------------------------------------------------


def test_variants_of_a_visible_recipe_are_returned(service: RecipeService) -> None:
    variants = service.list_variants(ME, "others-shared")

    assert variants is not None and [v.id for v in variants] == ["v1"]


@pytest.mark.parametrize("recipe_id", ["others-private", "does-not-exist"])
def test_variants_of_an_invisible_recipe_are_none_and_never_read(
    service: RecipeService, repository: FakeRepository, recipe_id: str
) -> None:
    assert service.list_variants(ME, recipe_id) is None
    assert repository.variant_reads == []


# --- search scopes -------------------------------------------------------------


def test_scope_mine_searches_only_own_recipes_newest_first(
    service: RecipeService, searches: list[dict[str, Any]]
) -> None:
    service.search(ME, "soup", Scope.MINE, limit=10)

    assert searches == [{"ids": ["my-shared", "my-private"], "query": "soup", "limit": 10}]


def test_scope_community_searches_shared_recipes_including_own(
    service: RecipeService, searches: list[dict[str, Any]]
) -> None:
    service.search(ME, "", Scope.COMMUNITY, limit=10)

    assert searches[0]["ids"] == ["orphan-shared", "my-shared", "others-shared"]


def test_scope_all_merges_both_newest_first_without_duplicates(
    service: RecipeService, searches: list[dict[str, Any]]
) -> None:
    service.search(ME, "", Scope.ALL, limit=10)

    assert searches[0]["ids"] == ["orphan-shared", "my-shared", "others-shared", "my-private"]


def test_search_never_returns_invisible_recipes(
    service: RecipeService, searches: list[dict[str, Any]]
) -> None:
    for scope in Scope:
        found = service.search(ME, "", scope, limit=100)
        assert {r.id for r in found} <= VISIBLE_TO_ME


def test_search_returns_what_the_search_returns(
    service: RecipeService, searches: list[dict[str, Any]]
) -> None:
    found = service.search(ME, "", Scope.ALL, limit=2)

    assert [r.id for r in found] == ["orphan-shared", "my-shared"]
    assert searches[0]["limit"] == 2
