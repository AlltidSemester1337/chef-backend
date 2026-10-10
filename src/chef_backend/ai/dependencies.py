"""Providers for the AI proxy's collaborators.

Each is created lazily on first use and cached, so /health never touches
Secret Manager, Firestore or Berget. Missing configuration fails closed (500).
Tests replace these through app.dependency_overrides.
"""

from collections.abc import Mapping
from functools import lru_cache

from google.cloud import firestore

from chef_backend.ai.client import BergetClient, ChatCompletionClient
from chef_backend.ai.purposes import Purpose, PurposeConfig, build_purposes
from chef_backend.ai.tracing import TracingChatClient
from chef_backend.config import get_settings
from chef_backend.quota import FirestoreQuotaStore, QuotaStore
from chef_backend.telemetry import get_tracer_provider


@lru_cache
def get_purposes() -> Mapping[Purpose, PurposeConfig]:
    prompt_file = get_settings().chat_prompt_file
    if prompt_file is None:
        raise RuntimeError("CHEF_CHAT_PROMPT_FILE must point to the main chat system prompt")
    chat_prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not chat_prompt:
        raise RuntimeError("the main chat system prompt is empty")
    return build_purposes(chat_prompt)


@lru_cache
def get_chat_client() -> ChatCompletionClient:
    settings = get_settings()
    if settings.berget_api_key is None:
        raise RuntimeError("CHEF_BERGET_API_KEY must be set")
    client = BergetClient.create(
        base_url=settings.berget_base_url,
        api_key=settings.berget_api_key.get_secret_value(),
        timeout_seconds=settings.berget_timeout_seconds,
    )
    tracer_provider = get_tracer_provider()
    if tracer_provider is None:
        return client
    return TracingChatClient(client, tracer_provider.get_tracer("chef_backend.ai"))


@lru_cache
def get_quota_store() -> QuotaStore:
    project_id = get_settings().gcp_project_id
    if not project_id:
        raise RuntimeError("CHEF_GCP_PROJECT_ID must be set")
    # Credentials via ADC: the Cloud Run service account, locally gcloud's user login.
    return FirestoreQuotaStore(firestore.Client(project=project_id))


def get_beta_interaction_limit() -> int:
    return get_settings().beta_interaction_limit
