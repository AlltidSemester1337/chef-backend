from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (prefix CHEF_).

    Secrets are never given defaults here. In Cloud Run they are injected from
    Secret Manager; locally they come from an untracked .env file.
    """

    model_config = SettingsConfigDict(env_prefix="CHEF_", env_file=".env", extra="ignore")

    environment: str = "local"
    gcp_project_id: str | None = None

    # AI proxy. The key comes from Secret Manager as an env var; the main chat prompt
    # is private and mounted from Secret Manager as a file (locally: an untracked file).
    berget_api_key: SecretStr | None = None
    berget_base_url: str = "https://api.berget.ai/v1"
    berget_timeout_seconds: float = 60.0
    chat_prompt_file: Path | None = None

    beta_interaction_limit: int = 20
    # Daily per-user budgets (Europe/Stockholm days) for calls the beta quota does
    # not count: background AI purposes and text-to-speech.
    daily_ai_uncounted_call_limit: int = 50
    daily_tts_char_limit: int = 25_000
    daily_image_limit: int = 20

    # Recipe images: generated with Vertex AI (ADC) and stored in Cloud Storage for
    # Firebase. The bucket name is project-specific config, so it has no default.
    storage_bucket: str | None = None
    vertex_location: str = "global"
    image_model: str = "gemini-2.5-flash-image"

    # Public base URL of this service: the MCP resource and OAuth issuer identifier.
    # Project-specific (the Cloud Run URL), so the default is for local runs only.
    public_base_url: str = "http://localhost:8080"

    # Tracing to Arize Phoenix (OTLP over HTTP). Optional: without endpoint and key
    # the backend runs untraced. The endpoint is project-specific config, not a secret.
    phoenix_endpoint: str | None = None
    phoenix_api_key: SecretStr | None = None
    phoenix_project_name: str = "chef-backend"


@lru_cache
def get_settings() -> Settings:
    return Settings()
