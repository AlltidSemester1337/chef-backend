"""Recipe models as stored in Firestore (see chef/.ai/firestore-schema.md).

These mirror the documents the apps write. They are internal: the HTTP API and the
MCP tools expose a narrower view (never another user's uid).
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError


def _default_if_none(default: Any) -> BeforeValidator:
    """The Android SDK writes null for unset properties: treat null like a missing field.

    Runs before type validation, so only None is replaced; every other value is still
    type-checked (strictly, see _Document).
    """

    def replace(value: object) -> object:
        return default() if value is None else value

    return BeforeValidator(replace)


Text = Annotated[str, _default_if_none(str)]
Flag = Annotated[bool, _default_if_none(bool)]
NonEmptyText = Annotated[str, Field(min_length=1)]
Difficulty = Literal["EASY", "MEDIUM", "HARD"]


class _Document(BaseModel):
    # strict: no type coercion, so "true" is not silently a bool and 2 not a str.
    # extra="ignore": a field added by a newer app version must not break reads.
    # populate_by_name: the camelCase Firestore names are aliases of the attributes.
    model_config = ConfigDict(strict=True, extra="ignore", populate_by_name=True)


class Ingredient(_Document):
    """Also used for nutrients, which have the same shape."""

    name: NonEmptyText
    quantity: Text = ""
    unit: Text = ""


IngredientList = Annotated[list[Ingredient], _default_if_none(list)]
TextList = Annotated[list[str], _default_if_none(list)]


class Recipe(_Document):
    id: str
    uid: str | None = None  # absent = orphaned recipe
    title: NonEmptyText
    summary: Text = ""
    difficulty: Difficulty | None = None
    cooking_time: Text = Field(default="", alias="cookingTime")
    prep_time: Text = Field(default="", alias="prepTime")
    servings: Text = ""
    image_url: str | None = Field(default=None, alias="imageUrl")
    video_url: str | None = Field(default=None, alias="videoUrl")
    is_favourite: Flag = Field(default=False, alias="isFavourite")
    copy_id: str | None = Field(default=None, alias="copyId")
    tips_and_tricks: Text = Field(default="", alias="tipsAndTricks")
    tags: TextList = []
    ingredients: IngredientList = []
    instructions: TextList = []
    nutrients_per_serving: IngredientList = Field(default=[], alias="nutrientsPerServing")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")


class RecipeVariant(_Document):
    id: str
    label: NonEmptyText
    created_at: datetime | None = Field(default=None, alias="createdAt")
    is_pinned: Flag = Field(default=False, alias="isPinned")
    title: NonEmptyText
    summary: Text = ""
    servings: Text = ""
    prep_time: Text = Field(default="", alias="prepTime")
    cooking_time: Text = Field(default="", alias="cookingTime")
    nutrients_per_serving: IngredientList = Field(default=[], alias="nutrientsPerServing")
    ingredients: IngredientList = []
    difficulty: Difficulty | None = None
    instructions: TextList = []
    tips_and_tricks: Text = Field(default="", alias="tipsAndTricks")


def parse_recipe(doc_id: str, data: Mapping[str, Any]) -> Recipe | None:
    """Recipe from a Firestore document, or None if the document is malformed."""
    try:
        return Recipe.model_validate({**data, "id": doc_id})
    except ValidationError:
        return None


def parse_variant(doc_id: str, data: Mapping[str, Any]) -> RecipeVariant | None:
    """Variant from a Firestore document, or None if the document is malformed."""
    try:
        return RecipeVariant.model_validate({**data, "id": doc_id})
    except ValidationError:
        return None
