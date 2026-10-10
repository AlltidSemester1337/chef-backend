from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import PermissionDenied, ServiceUnavailable
from google.cloud import texttospeech

from chef_backend.ai.client import UpstreamError
from chef_backend.ai.dependencies import get_beta_interaction_limit, get_quota_store
from chef_backend.auth import AuthenticatedUser, Claims, get_current_user, get_token_verifier
from chef_backend.main import create_app
from chef_backend.tts import (
    MAX_TTS_CHARS,
    VOICE_NAME,
    GoogleSpeechSynthesizer,
    get_speech_synthesizer,
)

URL = "/v1/tts/synthesize"
MP3 = b"ID3\x04fake-mp3"
VERIFIED = AuthenticatedUser(uid="u1", email="cook@example.com", email_verified=True)

# --- GoogleSpeechSynthesizer ---------------------------------------------------


class FakeTtsApi:
    """Stands in for texttospeech.TextToSpeechClient; records the keyword arguments."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def synthesize_speech(self, **kwargs: Any) -> texttospeech.SynthesizeSpeechResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return texttospeech.SynthesizeSpeechResponse(audio_content=MP3)


def synthesizer_with(api: FakeTtsApi) -> GoogleSpeechSynthesizer:
    return GoogleSpeechSynthesizer(cast(texttospeech.TextToSpeechClient, api))


def test_synthesizer_returns_the_audio_bytes() -> None:
    assert synthesizer_with(FakeTtsApi()).synthesize("Preheat the oven.") == MP3


def test_synthesizer_uses_the_server_side_voice_and_format() -> None:
    api = FakeTtsApi()

    synthesizer_with(api).synthesize("Preheat the oven.")

    [kwargs] = api.calls
    assert kwargs["input"].text == "Preheat the oven."
    assert kwargs["voice"].name == VOICE_NAME
    assert kwargs["voice"].language_code == "en-US"
    assert kwargs["audio_config"].audio_encoding == texttospeech.AudioEncoding.MP3
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "error", [ServiceUnavailable("down"), PermissionDenied("no access")], ids=["503", "403"]
)
def test_synthesizer_maps_api_errors_to_upstream_error(error: Exception) -> None:
    with pytest.raises(UpstreamError):
        synthesizer_with(FakeTtsApi(error)).synthesize("Hello.")


# --- POST /v1/tts/synthesize ---------------------------------------------------


@dataclass
class FakeSynthesizer:
    error: Exception | None = None
    texts: list[str] = field(default_factory=list[str])

    def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        return MP3


@dataclass
class FakeQuotaStore:
    increments: int = 0

    def increment_and_get(self, uid: str) -> int:
        self.increments += 1
        return self.increments


@dataclass
class World:
    user: AuthenticatedUser = VERIFIED
    tts: FakeSynthesizer = field(default_factory=FakeSynthesizer)
    quota: FakeQuotaStore = field(default_factory=FakeQuotaStore)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def client(world: World) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides.update(
        {
            get_current_user: lambda: world.user,
            get_speech_synthesizer: lambda: world.tts,
            get_quota_store: lambda: world.quota,
            get_beta_interaction_limit: lambda: 20,
        }
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_returns_mp3_audio(client: TestClient, world: World) -> None:
    response = client.post(URL, json={"text": "Preheat the oven."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == MP3
    assert world.tts.texts == ["Preheat the oven."]


def test_does_not_consume_quota(client: TestClient, world: World) -> None:
    client.post(URL, json={"text": "Hello."})

    assert world.quota.increments == 0


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
def test_ineligible_users_are_rejected_before_synthesis(
    client: TestClient, world: World, user: AuthenticatedUser, status_code: int, detail: str
) -> None:
    world.user = user

    response = client.post(URL, json={"text": "Hello."})

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert world.tts.texts == []


def test_tts_failure_is_502_without_details(client: TestClient, world: World) -> None:
    world.tts.error = UpstreamError("TTS request failed: PermissionDenied('secret detail')")

    response = client.post(URL, json={"text": "Hello."})

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream_error"}
    assert "secret detail" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {"text": ""},
        {"text": "x" * (MAX_TTS_CHARS + 1)},
        {"text": "Hello.", "voice": "en-US-Studio-O"},
        {},
    ],
    ids=["empty", "too long", "client voice", "missing text"],
)
def test_invalid_requests_are_422_and_cost_nothing(
    client: TestClient, world: World, body: dict[str, Any]
) -> None:
    response = client.post(URL, json=body)

    assert response.status_code == 422
    assert world.tts.texts == []


class UnusedVerifier:
    def verify_id_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")

    def verify_app_check_token(self, token: str) -> Claims:
        raise AssertionError("verifier should not be called")


def test_requires_authentication(world: World) -> None:
    app = create_app()
    app.dependency_overrides.update(
        {
            get_token_verifier: UnusedVerifier,
            get_speech_synthesizer: lambda: world.tts,
            get_quota_store: lambda: world.quota,
        }
    )

    response = TestClient(app).post(URL, json={"text": "Hello."})

    assert response.status_code == 401
    assert world.tts.texts == []
