from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (prefix CHEF_).

    Secrets are never given defaults here. In Cloud Run they are injected from
    Secret Manager; locally they come from an untracked .env file.
    """

    model_config = SettingsConfigDict(env_prefix="CHEF_", env_file=".env", extra="ignore")

    environment: str = "local"
    gcp_project_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
