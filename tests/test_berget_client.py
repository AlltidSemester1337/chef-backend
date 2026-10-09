"""BergetClient against httpx.MockTransport: no network, real HTTP handling."""

import json
from collections.abc import Callable

import httpx
import pytest

from chef_backend.ai.client import BergetClient, UpstreamError
from chef_backend.ai.models import ChatMessage
from chef_backend.ai.purposes import PurposeConfig

CONFIG = PurposeConfig(system_prompt="You are Chef.", temperature=0.7, top_p=0.95, max_tokens=512)
MESSAGES = [
    ChatMessage(role="user", content="Hej"),
    ChatMessage(role="assistant", content="Hej!"),
    ChatMessage(role="user", content="Pasta?"),
]


def ok(content: object = "Gör en carbonara.") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def client_with(handler: Callable[[httpx.Request], httpx.Response]) -> BergetClient:
    return BergetClient(
        httpx.Client(
            base_url="https://berget.test/v1",
            headers={"Authorization": "Bearer test-key"},
            transport=httpx.MockTransport(handler),
        )
    )


def test_returns_the_assistant_content() -> None:
    assert client_with(lambda _: ok()).complete(CONFIG, MESSAGES) == "Gör en carbonara."


def test_sends_server_side_prompt_and_parameters() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok()

    client_with(handler).complete(CONFIG, MESSAGES)

    [request] = seen
    assert request.method == "POST"
    assert request.url == "https://berget.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["messages"] == [
        {"role": "system", "content": "You are Chef."},
        {"role": "user", "content": "Hej"},
        {"role": "assistant", "content": "Hej!"},
        {"role": "user", "content": "Pasta?"},
    ]
    assert body["model"] == CONFIG.model
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.7, 0.95, 512)
    assert "response_format" not in body


def test_json_mode_requests_a_json_object() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok("{}")

    json_config = PurposeConfig(
        system_prompt="Extract.", temperature=0.2, top_p=0.95, max_tokens=512, json_mode=True
    )
    client_with(handler).complete(json_config, MESSAGES)

    assert json.loads(seen[0].content)["response_format"] == {"type": "json_object"}


def raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


FAILURES: dict[str, Callable[[httpx.Request], httpx.Response]] = {
    "server error": lambda _: httpx.Response(500, text="internal error"),
    "rate limited": lambda _: httpx.Response(429, text="slow down"),
    "bad key": lambda _: httpx.Response(401, text="invalid api key"),
    "timeout": raise_timeout,
    "not json": lambda _: httpx.Response(200, text="<html>"),
    "no choices": lambda _: httpx.Response(200, json={"choices": []}),
    "null content": lambda _: ok(None),
}


@pytest.mark.parametrize("handler", FAILURES.values(), ids=FAILURES.keys())
def test_provider_failures_become_upstream_error(
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    with pytest.raises(UpstreamError):
        client_with(handler).complete(CONFIG, MESSAGES)
