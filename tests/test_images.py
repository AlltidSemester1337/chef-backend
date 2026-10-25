import io
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from google import genai
from google.api_core.exceptions import ServiceUnavailable
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from chef_backend.ai.client import UpstreamError
from chef_backend.ai.dependencies import (
    get_beta_interaction_limit,
    get_budget_store,
    get_quota_store,
    get_today,
)
from chef_backend.auth import AuthenticatedUser, Claims, get_current_user, get_token_verifier
from chef_backend.budget import UsageKind
from chef_backend.images import (
    IMAGE_PROMPT_TEMPLATE,
    MAX_RECIPE_CHARS,
    FirebaseImageStore,
    VertexImageGenerator,
    get_daily_image_limit,
    get_image_generator,
    get_image_store,
    public_url,
    to_jpeg,
)
from chef_backend.main import create_app

URL = "/v1/images/recipe"
TODAY = date(2026, 10, 25)
DAILY_LIMIT = 3
VERIFIED = AuthenticatedUser(uid="u1", email="cook@example.com", email_verified=True)
RECIPE = "Chana masala: chickpeas, tomatoes, garam masala."


def png(mode: str = "RGBA", size: tuple[int, int] = (4, 4)) -> bytes:
    output = io.BytesIO()
    Image.new(mode, size, color=(200, 100, 50, 255) if mode == "RGBA" else 128).save(
        output, format="PNG"
    )
    return output.getvalue()


# --- to_jpeg / public_url ------------------------------------------------------


@pytest.mark.parametrize("mode", ["RGBA", "RGB", "L", "P"])
def test_to_jpeg_reencodes_any_image_as_rgb_jpeg(mode: str) -> None:
    jpeg = to_jpeg(png(mode))

    with Image.open(io.BytesIO(jpeg)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.mode == "RGB"
        assert decoded.size == (4, 4)


def test_to_jpeg_rejects_data_that_is_not_an_image() -> None:
    with pytest.raises(UpstreamError):
        to_jpeg(b"<html>not an image</html>")


def test_public_url_encodes_the_object_path() -> None:
    assert public_url("my-bucket", "recipes/abc.jpg") == (
        "https://firebasestorage.googleapis.com/v0/b/my-bucket/o/recipes%2Fabc.jpg?alt=media"
    )


# --- VertexImageGenerator ------------------------------------------------------


class FakeModels:
    def __init__(self, response: types.GenerateContentResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def vertex_with(response: types.GenerateContentResponse | Exception) -> tuple[Any, FakeModels]:
    models = FakeModels(response)
    client = cast(genai.Client, SimpleNamespace(models=models))
    return VertexImageGenerator(client, "gemini-test-image"), models


def response_with(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def test_vertex_returns_the_inline_image_bytes() -> None:
    image = png()
    generator, _ = vertex_with(
        response_with(
            types.Part(text="Here is your image."),
            types.Part(inline_data=types.Blob(data=image, mime_type="image/png")),
        )
    )

    assert generator.generate("prompt") == image


def test_vertex_asks_for_an_image_from_the_configured_model() -> None:
    generator, models = vertex_with(
        response_with(types.Part(inline_data=types.Blob(data=png(), mime_type="image/png")))
    )

    generator.generate("a prompt")

    [call] = models.calls
    assert call["model"] == "gemini-test-image"
    assert call["contents"] == "a prompt"
    assert "IMAGE" in call["config"].response_modalities


@pytest.mark.parametrize(
    "response",
    [
        response_with(types.Part(text="I cannot draw that.")),
        types.GenerateContentResponse(candidates=[]),
        genai_errors.APIError(429, {"error": {"message": "quota"}}),
        genai_errors.APIError(500, {"error": {"message": "internal"}}),
    ],
    ids=["text only", "no candidates", "429", "500"],
)
def test_vertex_failures_become_upstream_error(
    response: types.GenerateContentResponse | Exception,
) -> None:
    generator, _ = vertex_with(response)

    with pytest.raises(UpstreamError):
        generator.generate("prompt")


# --- FirebaseImageStore --------------------------------------------------------


@dataclass
class FakeBlob:
    name: str
    error: Exception | None = None
    uploads: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def upload_from_string(self, data: bytes, **kwargs: Any) -> None:
        if self.error is not None:
            raise self.error
        self.uploads.append({"data": data, **kwargs})


@dataclass
class FakeStorage:
    error: Exception | None = None
    blobs: list[FakeBlob] = field(default_factory=list[FakeBlob])
    bucket_names: list[str] = field(default_factory=list[str])

    def bucket(self, name: str) -> Any:
        self.bucket_names.append(name)
        return SimpleNamespace(blob=self._blob)

    def _blob(self, name: str) -> FakeBlob:
        blob = FakeBlob(name, self.error)
        self.blobs.append(blob)
        return blob


def store_with(fake: FakeStorage) -> FirebaseImageStore:
    return FirebaseImageStore(cast(Any, fake), "my-bucket")


def test_store_uploads_a_new_jpeg_under_recipes_and_returns_its_url() -> None:
    fake = FakeStorage()

    url = store_with(fake).upload_jpeg(b"jpeg-bytes")

    [blob] = fake.blobs
    assert re.fullmatch(r"recipes/[0-9a-f-]{36}\.jpg", blob.name)
    assert blob.uploads == [
        {"data": b"jpeg-bytes", "content_type": "image/jpeg", "if_generation_match": 0}
    ]
    assert url == public_url("my-bucket", blob.name)
    assert fake.bucket_names == ["my-bucket"]


def test_store_never_reuses_a_path() -> None:
    fake = FakeStorage()
    store = store_with(fake)

    store.upload_jpeg(b"a")
    store.upload_jpeg(b"b")

    assert fake.blobs[0].name != fake.blobs[1].name


def test_store_upload_failure_becomes_upstream_error() -> None:
    with pytest.raises(UpstreamError):
        store_with(FakeStorage(error=ServiceUnavailable("down"))).upload_jpeg(b"x")


# --- POST /v1/images/recipe ----------------------------------------------------


@dataclass
class FakeGenerator:
    error: Exception | None = None
    image: bytes = field(default_factory=png)
    prompts: list[str] = field(default_factory=list[str])

    def generate(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.image


@dataclass
class FakeImageStore:
    error: Exception | None = None
    uploads: list[bytes] = field(default_factory=list[bytes])

    def upload_jpeg(self, data: bytes) -> str:
        if self.error is not None:
            raise self.error
        self.uploads.append(data)
        return f"https://images.test/{len(self.uploads)}.jpg"


@dataclass
class FakeQuotaStore:
    increments: int = 0

    def increment_and_get(self, uid: str) -> int:
        self.increments += 1
        return self.increments


@dataclass
class FakeBudgetStore:
    usage: dict[tuple[str, date, UsageKind], int] = field(
        default_factory=dict[tuple[str, date, UsageKind], int]
    )

    def add_and_get(self, uid: str, day: date, kind: UsageKind, amount: int) -> int:
        key = (uid, day, kind)
        self.usage[key] = self.usage.get(key, 0) + amount
        return self.usage[key]


@dataclass
class World:
    user: AuthenticatedUser = VERIFIED
    generator: FakeGenerator = field(default_factory=FakeGenerator)
    store: FakeImageStore = field(default_factory=FakeImageStore)
    quota: FakeQuotaStore = field(default_factory=FakeQuotaStore)
    budget: FakeBudgetStore = field(default_factory=FakeBudgetStore)

    def images_today(self) -> int:
        return self.budget.usage.get((self.user.uid, TODAY, UsageKind.IMAGES), 0)


@pytest.fixture
def world() -> World:
    return World()


def overrides(world: World) -> dict[Any, Any]:
    return {
        get_image_generator: lambda: world.generator,
        get_image_store: lambda: world.store,
        get_quota_store: lambda: world.quota,
        get_budget_store: lambda: world.budget,
        get_beta_interaction_limit: lambda: 20,
        get_daily_image_limit: lambda: DAILY_LIMIT,
        get_today: lambda: TODAY,
    }


@pytest.fixture
def client(world: World) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides.update(overrides(world) | {get_current_user: lambda: world.user})
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_returns_the_url_of_the_stored_jpeg(client: TestClient, world: World) -> None:
    response = client.post(URL, json={"recipe": RECIPE})

    assert response.status_code == 200
    assert response.json() == {"imageUrl": "https://images.test/1.jpg"}
    [uploaded] = world.store.uploads
    assert uploaded[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_prompt_is_the_server_template_plus_the_recipe(client: TestClient, world: World) -> None:
    client.post(URL, json={"recipe": RECIPE})

    assert world.generator.prompts == [IMAGE_PROMPT_TEMPLATE + RECIPE]


def test_each_image_uses_the_daily_image_budget(client: TestClient, world: World) -> None:
    statuses = [client.post(URL, json={"recipe": RECIPE}).status_code for _ in range(4)]

    assert statuses == [200, 200, 200, 429]
    assert client.post(URL, json={"recipe": RECIPE}).json() == {"detail": "rate_limited"}
    assert len(world.generator.prompts) == DAILY_LIMIT


def test_images_do_not_consume_the_beta_quota(client: TestClient, world: World) -> None:
    client.post(URL, json={"recipe": RECIPE})

    assert world.quota.increments == 0


def test_admin_has_no_image_budget(client: TestClient, world: World) -> None:
    world.user = replace(VERIFIED, is_admin=True)

    statuses = {client.post(URL, json={"recipe": RECIPE}).status_code for _ in range(5)}

    assert statuses == {200}
    assert world.images_today() == 0


@pytest.mark.parametrize(
    ("user", "status_code", "detail"),
    [
        (
            replace(VERIFIED, email=None, email_verified=False, is_anonymous=True),
            429,
            "quota_exceeded",
        ),
        (replace(VERIFIED, email_verified=False), 403, "email_verification_required"),
    ],
    ids=["anonymous", "unverified"],
)
def test_ineligible_users_are_rejected_before_generation(
    client: TestClient, world: World, user: AuthenticatedUser, status_code: int, detail: str
) -> None:
    world.user = user

    response = client.post(URL, json={"recipe": RECIPE})

    assert (response.status_code, response.json()) == (status_code, {"detail": detail})
    assert world.generator.prompts == []
    assert world.budget.usage == {}


@pytest.mark.parametrize(
    "failure",
    ["generation", "not an image", "upload"],
)
def test_failures_are_502_without_details_and_still_use_the_budget(
    client: TestClient, world: World, failure: str
) -> None:
    if failure == "generation":
        world.generator.error = UpstreamError("image request failed: HTTP 500 secret-detail")
    elif failure == "not an image":
        world.generator.image = b"secret-detail"
    else:
        world.store.error = UpstreamError("image upload failed: secret-detail")

    response = client.post(URL, json={"recipe": RECIPE})

    assert (response.status_code, response.json()) == (502, {"detail": "upstream_error"})
    assert "secret-detail" not in response.text
    assert world.images_today() == 1


def test_failure_log_does_not_contain_the_recipe(
    client: TestClient, world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.generator.error = UpstreamError("image request failed: HTTP 500")

    client.post(URL, json={"recipe": "PRIVATE family recipe"})

    assert "PRIVATE" not in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        {"recipe": ""},
        {"recipe": "x" * (MAX_RECIPE_CHARS + 1)},
        {"recipe": RECIPE, "prompt": "draw a cat"},
        {"recipe": RECIPE, "model": "imagen-ultra"},
        {},
    ],
    ids=["empty", "too long", "client prompt", "client model", "missing recipe"],
)
def test_invalid_requests_are_422_and_cost_nothing(
    client: TestClient, world: World, body: dict[str, Any]
) -> None:
    response = client.post(URL, json=body)

    assert response.status_code == 422
    assert world.generator.prompts == []
    assert world.budget.usage == {}


class UnusedVerifier:
    def verify_id_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")

    def verify_app_check_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")


def test_requires_authentication(world: World) -> None:
    app = create_app()
    app.dependency_overrides.update(overrides(world) | {get_token_verifier: UnusedVerifier})

    response = TestClient(app).post(URL, json={"recipe": RECIPE})

    assert response.status_code == 401
    assert world.generator.prompts == []
