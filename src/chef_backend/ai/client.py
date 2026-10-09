"""Chat completion client for Berget.ai (OpenAI-compatible API)."""

from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from chef_backend.ai.models import ChatMessage
from chef_backend.ai.purposes import PurposeConfig


class UpstreamError(Exception):
    """The AI provider failed or answered with something unusable (maps to 502).

    The message is for our logs only; never return it to the client, since it may
    contain provider error details.
    """


class ChatCompletionClient(Protocol):
    def complete(self, config: PurposeConfig, messages: Sequence[ChatMessage]) -> str:
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

    def complete(self, config: PurposeConfig, messages: Sequence[ChatMessage]) -> str:
        try:
            response = self._http.post(
                "chat/completions", json=build_request_body(config, messages)
            )
        except httpx.HTTPError as error:
            raise UpstreamError(f"request to AI provider failed: {error!r}") from error

        if response.status_code != 200:
            raise UpstreamError(f"AI provider returned HTTP {response.status_code}")
        try:
            return parse_content(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise UpstreamError("malformed response from AI provider") from error


def build_request_body(config: PurposeConfig, messages: Sequence[ChatMessage]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            *({"role": m.role, "content": m.content} for m in messages),
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    if config.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def parse_content(payload: Any) -> str:
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("content is not a string")
    return content
