"""Chat completion client for Berget.ai (OpenAI-compatible API)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from chef_backend.ai.models import ChatMessage
from chef_backend.ai.purposes import Purpose, PurposeConfig


class UpstreamError(Exception):
    """The AI provider failed or answered with something unusable (maps to 502).

    The message is for our logs only; never return it to the client, since it may
    contain provider error details.
    """


@dataclass(frozen=True, slots=True)
class CompletionCall:
    """Everything known about one AI call. Wrappers (e.g. tracing) can use the context
    fields; the provider client only needs config and messages."""

    purpose: Purpose
    config: PurposeConfig
    messages: Sequence[ChatMessage]
    user_id: str


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    # Token usage as reported by the provider, when it reports it.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatCompletionClient(Protocol):
    def complete(self, call: CompletionCall) -> Completion:
        """Return the assistant's reply, or raise UpstreamError."""
        ...


class BergetClient:
    def __init__(self, http: httpx.Client) -> None:
        # The httpx.Client carries base URL, API key and timeout, so tests can
        # pass one with a mock transport.
        self._http = http

    @classmethod
    def create(cls, base_url: str, api_key: str, timeout_seconds: float) -> "BergetClient":
        return cls(
            httpx.Client(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(timeout_seconds, connect=15.0),
            )
        )

    def complete(self, call: CompletionCall) -> Completion:
        try:
            response = self._http.post(
                "chat/completions", json=build_request_body(call.config, call.messages)
            )
        except httpx.HTTPError as error:
            raise UpstreamError(f"request to AI provider failed: {error!r}") from error

        if response.status_code != 200:
            raise UpstreamError(f"AI provider returned HTTP {response.status_code}")
        try:
            return parse_completion(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise UpstreamError("malformed response from AI provider") from error


def invocation_parameters(config: PurposeConfig) -> dict[str, Any]:
    """The sampling parameters sent to the provider. Also used for tracing, so traces
    always show exactly what was sent."""
    parameters: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    if config.json_mode:
        parameters["response_format"] = {"type": "json_object"}
    return parameters


def build_request_body(config: PurposeConfig, messages: Sequence[ChatMessage]) -> dict[str, Any]:
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            *({"role": m.role, "content": m.content} for m in messages),
        ],
        **invocation_parameters(config),
    }


def parse_completion(payload: Any) -> Completion:
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("content is not a string")
    usage: dict[str, Any] = payload.get("usage") or {}
    return Completion(
        content=content,
        prompt_tokens=_token_count(usage.get("prompt_tokens")),
        completion_tokens=_token_count(usage.get("completion_tokens")),
    )


def _token_count(value: Any) -> int | None:
    # Usage is optional and only informational: ignore anything that is not a count.
    return value if isinstance(value, int) and not isinstance(value, bool) else None
