"""Recipe images: generated on the server, stored in Cloud Storage for Firebase.

The prompt template, the model and the upload all live here, so clients only send
the recipe text and get back a URL. Images are stored under recipes/{uuid}.jpg,
the same path the apps have always used, so existing readers keep working.
"""

import io
import logging
import uuid
from datetime import date
from functools import lru_cache
from typing import Annotated, Protocol
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from google import genai
from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage  # pyright: ignore[reportMissingTypeStubs]
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from chef_backend.access import require_budget, require_quota
from chef_backend.ai.client import UpstreamError
from chef_backend.ai.dependencies import (
    get_beta_interaction_limit,
    get_budget_store,
    get_quota_store,
    get_today,
)
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.budget import BudgetStore, UsageKind, check_budget
from chef_backend.config import get_settings
from chef_backend.quota import QuotaStore, check_quota

logger = logging.getLogger(__name__)

MAX_RECIPE_CHARS = 8_000
JPEG_QUALITY = 90
IMAGE_PROMPT_TEMPLATE = (
    "As a professional photographer specializing in 100mm Macro lens natural lighting food "
    "photography, create a photorealistic, colorful, visually appealing image of a single "
    "serving for the following recipe: "
)


class RecipeImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe: Annotated[str, Field(min_length=1, max_length=MAX_RECIPE_CHARS)]


class RecipeImageResponse(BaseModel):
    image_url: str = Field(serialization_alias="imageUrl")


class ImageGenerator(Protocol):
    def generate(self, prompt: str) -> bytes:
        """Return the encoded image (PNG, JPEG, ...) for the prompt, or raise UpstreamError."""
        ...


class ImageStore(Protocol):
    def upload_jpeg(self, data: bytes) -> str:
        """Store a JPEG under a new recipes/{uuid}.jpg path and return its public URL."""
        ...


class VertexImageGenerator:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, prompt: str) -> bytes:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
        except genai_errors.APIError as error:
            raise UpstreamError(f"image request failed: HTTP {error.code}") from error
        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.inline_data is not None and part.inline_data.data:
                    return part.inline_data.data
        raise UpstreamError("no image in the model response")


def to_jpeg(image: bytes) -> bytes:
    """Re-encode whatever the model returned as an RGB JPEG, like the apps did."""
    try:
        with Image.open(io.BytesIO(image)) as decoded:
            output = io.BytesIO()
            decoded.convert("RGB").save(output, format="JPEG", quality=JPEG_QUALITY)
            return output.getvalue()
    except (UnidentifiedImageError, OSError) as error:
        raise UpstreamError("the model returned data that is not an image") from error


def public_url(bucket: str, path: str) -> str:
    # Same URL shape the apps build today: readable without a token because the
    # Storage rules make recipes/ publicly readable.
    return (
        f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{quote(path, safe='')}?alt=media"
    )


class FirebaseImageStore:
    # google-cloud-storage is only partly typed; its calls are confined to this class.
    def __init__(self, client: storage.Client, bucket_name: str) -> None:
        self._bucket = client.bucket(bucket_name)  # pyright: ignore[reportUnknownMemberType]
        self._bucket_name = bucket_name

    def upload_jpeg(self, data: bytes) -> str:
        path = f"recipes/{uuid.uuid4()}.jpg"
        blob = self._bucket.blob(path)  # pyright: ignore[reportUnknownMemberType]
        try:
            # if_generation_match=0: create only, never overwrite an existing object.
            blob.upload_from_string(  # pyright: ignore[reportUnknownMemberType]
                data, content_type="image/jpeg", if_generation_match=0
            )
        except GoogleAPIError as error:
            raise UpstreamError(f"image upload failed: {type(error).__name__}") from error
        return public_url(self._bucket_name, path)


@lru_cache
def get_image_generator() -> ImageGenerator:
    settings = get_settings()
    if not settings.gcp_project_id:
        raise RuntimeError("CHEF_GCP_PROJECT_ID must be set")
    # Vertex AI with ADC: the Cloud Run service account needs roles/aiplatform.user.
    client = genai.Client(
        vertexai=True, project=settings.gcp_project_id, location=settings.vertex_location
    )
    return VertexImageGenerator(client, settings.image_model)


@lru_cache
def get_image_store() -> ImageStore:
    settings = get_settings()
    if not settings.gcp_project_id or not settings.storage_bucket:
        raise RuntimeError("CHEF_GCP_PROJECT_ID and CHEF_STORAGE_BUCKET must be set")
    return FirebaseImageStore(
        storage.Client(project=settings.gcp_project_id), settings.storage_bucket
    )


def get_daily_image_limit() -> int:
    return get_settings().daily_image_limit


router = APIRouter(prefix="/v1/images", tags=["images"])


@router.post("/recipe", response_model_by_alias=True)
def create_recipe_image(
    request: RecipeImageRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    generator: Annotated[ImageGenerator, Depends(get_image_generator)],
    store: Annotated[ImageStore, Depends(get_image_store)],
    quota_store: Annotated[QuotaStore, Depends(get_quota_store)],
    limit: Annotated[int, Depends(get_beta_interaction_limit)],
    budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
    daily_limit: Annotated[int, Depends(get_daily_image_limit)],
    today: Annotated[date, Depends(get_today)],
) -> RecipeImageResponse:
    """Generate, store and return an image for one recipe. Uses the daily image budget."""
    require_quota(check_quota(user, quota_store, counted=False, limit=limit))
    require_budget(
        check_budget(
            user, budget_store, kind=UsageKind.IMAGES, amount=1, limit=daily_limit, day=today
        )
    )

    try:
        jpeg = to_jpeg(generator.generate(IMAGE_PROMPT_TEMPLATE + request.recipe))
        image_url = store.upload_jpeg(jpeg)
    except UpstreamError:
        # The recipe text is user data and stays out of the log.
        logger.exception("Recipe image generation or upload failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "upstream_error") from None

    return RecipeImageResponse(image_url=image_url)
