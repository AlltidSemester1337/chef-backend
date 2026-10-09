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


@lru_cache
def get_settings() -> Settings:
    return Settings()
