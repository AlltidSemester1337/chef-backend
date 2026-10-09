import pytest
from fastapi.testclient import TestClient

from chef_backend.ai.dependencies import get_purposes
from chef_backend.config import Settings
from chef_backend.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_production_refuses_to_start_with_missing_config() -> None:
    app = create_app(Settings(environment="prod", chat_prompt_file=None))

    with pytest.raises(RuntimeError), TestClient(app):
        pass

    get_purposes.cache_clear()


def test_local_starts_without_ai_config() -> None:
    with TestClient(create_app(Settings(environment="local"))) as client:
        assert client.get("/health").status_code == 200
