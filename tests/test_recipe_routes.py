"""GET /v1/recipes, /v1/recipes/{id} and /v1/recipes/{id}/variants.

The service and its access policy have their own tests; these check the HTTP layer:
parameters, status codes, the response shape and that no owner uid ever leaves.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from chef_backend.auth import AuthenticatedUser, Claims, get_current_user, get_token_verifier
from chef_backend.main import create_app
from chef_backend.recipes.models import Ingredient, Recipe, RecipeVariant
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.service import MAX_QUERY_CHARS, MAX_SEARCH_LIMIT, RecipeService

ME = AuthenticatedUser(uid="me-uid", email="cook@example.com", email_verified=True)
OTHER_UID = "other-secret-uid"
T0 = datetime(2026, 10, 1, tzinfo=UTC)


def recipe(recipe_id: str, uid: str | None, *, shared: bool = False, day: int = 0) -> Recipe:
    return Recipe.model_validate(
        {
            "id": recipe_id,
            "uid": uid,
            "title": f"Title {recipe_id}",
            "summary": "Kort.",
            "difficulty": "EASY",
            "prepTime": "10 minutes",
            "cookingTime": "20 minutes",
            "servings": "4",
            "tags": ["vegetarian"],
            "imageUrl": "https://example.com/a.jpg",
            "isFavourite": shared,
            "copyId": "original-recipe",
            "ingredients": [Ingredient(name="lök", quantity="1", unit="st")],
            "instructions": ["Hacka."],
            "tipsAndTricks": "Tips.",
            "nutrientsPerServing": [Ingredient(name="protein", quantity="5", unit="g")],
            "updatedAt": T0 + timedelta(days=day),
        }
    )


@dataclass
class FakeRepository:
    recipes: list[Recipe] = field(
        default_factory=lambda: [
            recipe("mine", ME.uid, day=1),
            recipe("theirs-shared", OTHER_UID, shared=True, day=2),
            recipe("theirs-private", OTHER_UID, day=3),
        ]
    )

    def list_by_owner(self, uid: str) -> list[Recipe]:
        return [r for r in reversed(self.recipes) if r.uid == uid]

    def list_community(self) -> list[Recipe]:
        return [r for r in reversed(self.recipes) if r.is_favourite]

    def get(self, recipe_id: str) -> Recipe | None:
        return next((r for r in self.recipes if r.id == recipe_id), None)

    def list_variants(self, recipe_id: str) -> list[RecipeVariant]:
        return [
            RecipeVariant.model_validate(
                {"id": "v1", "label": "Mild", "title": "Mild", "createdAt": T0, "isPinned": True}
            )
        ]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: ME
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(FakeRepository())
    with TestClient(app) as test_client:
        yield test_client


def recipe_ids(response: Any) -> list[str]:
    return [r["id"] for r in response.json()["recipes"]]


# --- GET /v1/recipes -----------------------------------------------------------


def test_search_defaults_to_own_and_community_newest_first(client: TestClient) -> None:
    response = client.get("/v1/recipes")

    assert response.status_code == 200
    assert recipe_ids(response) == ["theirs-shared", "mine"]


def test_search_hits_are_summaries_in_camel_case(client: TestClient) -> None:
    [hit] = client.get("/v1/recipes", params={"scope": "mine"}).json()["recipes"]

    assert hit == {
        "id": "mine",
        "title": "Title mine",
        "summary": "Kort.",
        "difficulty": "EASY",
        "prepTime": "10 minutes",
        "cookingTime": "20 minutes",
        "servings": "4",
        "tags": ["vegetarian"],
        "imageUrl": "https://example.com/a.jpg",
        "isMine": True,
        "isFavourite": False,
        "updatedAt": "2026-10-02T00:00:00Z",
    }


def test_search_passes_query_scope_and_limit(client: TestClient) -> None:
    assert recipe_ids(client.get("/v1/recipes", params={"scope": "community"})) == ["theirs-shared"]
    assert recipe_ids(client.get("/v1/recipes", params={"scope": "others"})) == ["theirs-shared"]
    assert recipe_ids(client.get("/v1/recipes", params={"q": "title mine"})) == ["mine"]
    assert recipe_ids(client.get("/v1/recipes", params={"limit": 1})) == ["theirs-shared"]


@pytest.mark.parametrize(
    "params",
    [
        {"scope": "everyone"},
        {"limit": 0},
        {"limit": MAX_SEARCH_LIMIT + 1},
        {"limit": "many"},
        {"q": "x" * (MAX_QUERY_CHARS + 1)},
    ],
    ids=["unknown scope", "limit 0", "limit too high", "limit not a number", "query too long"],
)
def test_invalid_search_parameters_are_422(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get("/v1/recipes", params=params).status_code == 422


# --- GET /v1/recipes/{id} ------------------------------------------------------


def test_get_returns_the_full_recipe(client: TestClient) -> None:
    response = client.get("/v1/recipes/mine")

    assert response.status_code == 200
    body = response.json()
    assert body["isMine"] is True
    assert body["ingredients"] == [{"name": "lök", "quantity": "1", "unit": "st"}]
    assert body["instructions"] == ["Hacka."]
    assert body["tipsAndTricks"] == "Tips."
    assert body["nutrientsPerServing"] == [{"name": "protein", "quantity": "5", "unit": "g"}]
    assert body["videoUrl"] is None


def test_another_users_shared_recipe_does_not_reveal_the_owner(client: TestClient) -> None:
    response = client.get("/v1/recipes/theirs-shared")

    assert response.status_code == 200
    assert response.json()["isMine"] is False
    assert "uid" not in response.json()
    assert OTHER_UID not in response.text


def test_no_response_contains_an_owner_uid_or_internal_fields(client: TestClient) -> None:
    for url in ["/v1/recipes", "/v1/recipes/theirs-shared", "/v1/recipes/theirs-shared/variants"]:
        text = client.get(url).text
        assert OTHER_UID not in text and ME.uid not in text
        assert '"uid"' not in text and "copyId" not in text


@pytest.mark.parametrize("recipe_id", ["theirs-private", "does-not-exist", "bad.id"])
def test_invisible_missing_and_malformed_ids_are_the_same_404(
    client: TestClient, recipe_id: str
) -> None:
    for url in [f"/v1/recipes/{recipe_id}", f"/v1/recipes/{recipe_id}/variants"]:
        response = client.get(url)
        assert (response.status_code, response.json()) == (404, {"detail": "recipe_not_found"})


# --- GET /v1/recipes/{id}/variants ---------------------------------------------


def test_variants_of_a_visible_recipe(client: TestClient) -> None:
    response = client.get("/v1/recipes/theirs-shared/variants")

    assert response.status_code == 200
    [variant] = response.json()["variants"]
    assert (variant["id"], variant["label"], variant["isPinned"]) == ("v1", "Mild", True)
    assert variant["createdAt"] == "2026-10-01T00:00:00Z"


# --- Authentication ------------------------------------------------------------


class UnusedVerifier:
    def verify_id_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")

    def verify_app_check_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")


@pytest.mark.parametrize("url", ["/v1/recipes", "/v1/recipes/mine", "/v1/recipes/mine/variants"])
def test_every_route_requires_authentication(url: str) -> None:
    app = create_app()
    app.dependency_overrides[get_token_verifier] = UnusedVerifier
    app.dependency_overrides[get_recipe_service] = lambda: RecipeService(FakeRepository())

    assert TestClient(app).get(url).status_code == 401
