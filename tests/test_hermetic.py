"""Proves the guards in conftest.py: no test can reach a real cloud service.

Each check matches the guard's own message, so a test cannot pass because of some
other failure (e.g. credentials that merely happen to be missing on this machine).
"""

import google.auth
import pytest
from google.auth import default as imported_default
from google.cloud import firestore, texttospeech

from chef_backend.ai import dependencies
from chef_backend.config import Settings

NO_CREDENTIALS = "tests must not use Google credentials"
FIRESTORE_GUARD = "Firestore client .*in tests"


def test_application_default_credentials_are_unavailable() -> None:
    with pytest.raises(RuntimeError, match=NO_CREDENTIALS):
        google.auth.default()


def test_credentials_are_unavailable_even_via_an_early_import() -> None:
    # Libraries that did "from google.auth import default" still hit the guard,
    # because conftest replaces it before any test module imports google.auth.
    with pytest.raises(RuntimeError, match=NO_CREDENTIALS):
        imported_default()


def test_google_clients_cannot_find_credentials() -> None:
    with pytest.raises(RuntimeError, match=NO_CREDENTIALS):
        texttospeech.TextToSpeechClient()


@pytest.mark.parametrize("project", ["my-real-project", "", None])
def test_firestore_rejects_non_demo_projects(project: str | None) -> None:
    with pytest.raises(RuntimeError, match=FIRESTORE_GUARD):
        firestore.Client(project=project)


def test_firestore_demo_project_needs_the_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)

    with pytest.raises(RuntimeError, match=FIRESTORE_GUARD):
        firestore.Client(project="demo-chef-backend")


def test_a_dotenv_project_id_cannot_leak_into_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if a real project ID reached the settings, the store could not be created.
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(gcp_project_id="my-real-project")
    )
    dependencies.get_firestore_client.cache_clear()

    try:
        with pytest.raises(RuntimeError, match=FIRESTORE_GUARD):
            dependencies.get_firestore_client()
    finally:
        dependencies.get_firestore_client.cache_clear()


def test_settings_from_dotenv_are_overridden() -> None:
    settings = Settings()

    assert settings.gcp_project_id == ""
    assert settings.environment == "test"


def test_storage_client_cannot_find_credentials() -> None:
    from google.cloud import storage  # pyright: ignore[reportMissingTypeStubs]

    with pytest.raises(RuntimeError, match=NO_CREDENTIALS):
        storage.Client(project="my-real-project")


def test_vertex_calls_cannot_find_credentials() -> None:
    from google import genai

    # genai loads credentials lazily, so the guard fires on the first call, before
    # any request leaves the process.
    client = genai.Client(vertexai=True, project="my-real-project", location="global")

    with pytest.raises(RuntimeError, match=NO_CREDENTIALS):
        client.models.generate_content(model="gemini-2.5-flash-image", contents="x")
