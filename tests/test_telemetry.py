from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from pydantic import SecretStr

from chef_backend.ai import dependencies
from chef_backend.ai.client import BergetClient
from chef_backend.ai.tracing import TracingChatClient
from chef_backend.config import Settings
from chef_backend.telemetry import build_tracer_provider

TRACED = Settings(
    phoenix_endpoint="https://phoenix.test/v1/traces",
    phoenix_api_key=SecretStr("test-key"),
    phoenix_project_name="chef-backend-test",
)


@pytest.mark.parametrize(
    "settings",
    [
        Settings(phoenix_endpoint=None, phoenix_api_key=None),
        Settings(phoenix_endpoint="https://phoenix.test/v1/traces", phoenix_api_key=None),
        Settings(phoenix_endpoint=None, phoenix_api_key=SecretStr("test-key")),
    ],
    ids=["nothing", "endpoint only", "key only"],
)
def test_tracing_is_off_unless_fully_configured(settings: Settings) -> None:
    assert build_tracer_provider(settings) is None


def test_missing_tracing_config_is_a_warning_in_production(
    caplog: pytest.LogCaptureFixture,
) -> None:
    build_tracer_provider(Settings(environment="prod", phoenix_endpoint=None))

    assert "not configured" in caplog.text


def test_provider_names_the_service_and_phoenix_project() -> None:
    provider = build_tracer_provider(TRACED)

    assert provider is not None
    try:
        assert provider.resource.attributes["service.name"] == "chef-backend"
        assert provider.resource.attributes["openinference.project.name"] == "chef-backend-test"
    finally:
        provider.shutdown()


@pytest.fixture
def fresh_chat_client() -> Iterator[None]:
    dependencies.get_chat_client.cache_clear()
    yield
    dependencies.get_chat_client.cache_clear()


@pytest.mark.usefixtures("fresh_chat_client")
def test_chat_client_is_traced_when_tracing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(berget_api_key=SecretStr("k"))
    )
    monkeypatch.setattr(dependencies, "get_tracer_provider", TracerProvider)

    assert isinstance(dependencies.get_chat_client(), TracingChatClient)


@pytest.mark.usefixtures("fresh_chat_client")
def test_chat_client_is_untraced_without_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(berget_api_key=SecretStr("k"))
    )
    monkeypatch.setattr(dependencies, "get_tracer_provider", lambda: None)

    assert isinstance(dependencies.get_chat_client(), BergetClient)
