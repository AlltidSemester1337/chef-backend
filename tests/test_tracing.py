"""Spec for TracingChatClient: one OpenInference LLM span per AI call.

Spans go to an InMemorySpanExporter through a SimpleSpanProcessor (exports
synchronously when a span ends), so the test can inspect them right away.
"""

import json
from dataclasses import dataclass, field

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from chef_backend.ai.client import Completion, CompletionCall, UpstreamError
from chef_backend.ai.models import ChatMessage
from chef_backend.ai.purposes import Purpose, PurposeConfig
from chef_backend.ai.tracing import TracingChatClient

PRIVATE_SYSTEM_PROMPT = "PRIVATE SYSTEM PROMPT, must not reach Phoenix"
CONFIG = PurposeConfig(
    system_prompt=PRIVATE_SYSTEM_PROMPT,
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
)
CALL = CompletionCall(
    purpose=Purpose.CHAT,
    config=CONFIG,
    messages=[
        ChatMessage(role="user", content="Hej"),
        ChatMessage(role="assistant", content="Hej! Vad vill du laga?"),
        ChatMessage(role="user", content="Något med kikärtor"),
    ],
    user_id="user-123",
)
REPLY = Completion(content="Prova en chana masala.", prompt_tokens=120, completion_tokens=30)


@dataclass
class FakeInner:
    reply: Completion = REPLY
    error: Exception | None = None
    calls: list[CompletionCall] = field(default_factory=list[CompletionCall])

    def complete(self, call: CompletionCall) -> Completion:
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def inner() -> FakeInner:
    return FakeInner()


@pytest.fixture
def client(inner: FakeInner, exporter: InMemorySpanExporter) -> TracingChatClient:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TracingChatClient(inner, provider.get_tracer("test"))


def only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    [span] = exporter.get_finished_spans()
    return span


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


# --- Delegation: tracing must not change behaviour -----------------------------


def test_delegates_and_returns_the_inner_result_unchanged(
    client: TracingChatClient, inner: FakeInner
) -> None:
    assert client.complete(CALL) is REPLY
    assert inner.calls == [CALL]


def test_errors_are_re_raised_unchanged(client: TracingChatClient, inner: FakeInner) -> None:
    error = UpstreamError("AI provider returned HTTP 500")
    inner.error = error

    with pytest.raises(UpstreamError) as raised:
        client.complete(CALL)

    assert raised.value is error


# --- One LLM span per call -----------------------------------------------------


def test_one_span_named_after_the_purpose(
    client: TracingChatClient, exporter: InMemorySpanExporter
) -> None:
    client.complete(CALL)

    span = only_span(exporter)
    assert span.name == "ai.chat"
    assert span.status.status_code == StatusCode.OK


def test_span_describes_the_llm_call(
    client: TracingChatClient, exporter: InMemorySpanExporter
) -> None:
    client.complete(CALL)

    a = attrs(only_span(exporter))
    assert a["openinference.span.kind"] == "LLM"
    assert a["llm.provider"] == "berget"
    assert a["llm.system"] == "mistralai"
    assert a["llm.model_name"] == CONFIG.model
    assert json.loads(str(a["llm.invocation_parameters"])) == {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
    }
    assert a["chef.purpose"] == "chat"
    assert a["user.id"] == "user-123"


def test_span_contains_the_conversation_but_not_the_system_prompt(
    client: TracingChatClient, exporter: InMemorySpanExporter
) -> None:
    client.complete(CALL)

    a = attrs(only_span(exporter))
    assert [
        (a[f"llm.input_messages.{i}.message.role"], a[f"llm.input_messages.{i}.message.content"])
        for i in range(3)
    ] == [
        ("user", "Hej"),
        ("assistant", "Hej! Vad vill du laga?"),
        ("user", "Något med kikärtor"),
    ]
    assert "llm.input_messages.3.message.role" not in a
    assert json.loads(str(a["input.value"])) == [m.model_dump() for m in CALL.messages]
    assert a["input.mime_type"] == "application/json"
    assert PRIVATE_SYSTEM_PROMPT not in json.dumps(a, ensure_ascii=False)


def test_span_contains_the_reply_and_token_counts(
    client: TracingChatClient, exporter: InMemorySpanExporter
) -> None:
    client.complete(CALL)

    a = attrs(only_span(exporter))
    assert a["llm.output_messages.0.message.role"] == "assistant"
    assert a["llm.output_messages.0.message.content"] == "Prova en chana masala."
    assert a["output.value"] == "Prova en chana masala."
    assert a["output.mime_type"] == "text/plain"
    assert a["llm.token_count.prompt"] == 120
    assert a["llm.token_count.completion"] == 30
    assert a["llm.token_count.total"] == 150


def test_unknown_token_counts_are_left_out(
    client: TracingChatClient, inner: FakeInner, exporter: InMemorySpanExporter
) -> None:
    inner.reply = Completion(content="Svar")

    client.complete(CALL)

    assert not any(key.startswith("llm.token_count") for key in attrs(only_span(exporter)))


def test_json_mode_is_part_of_the_invocation_parameters(
    client: TracingChatClient, exporter: InMemorySpanExporter
) -> None:
    json_config = PurposeConfig(
        system_prompt="Extract.", temperature=0.2, top_p=0.95, max_tokens=8192, json_mode=True
    )
    call = CompletionCall(
        purpose=Purpose.DERIVE_RECIPE_JSON,
        config=json_config,
        messages=CALL.messages,
        user_id="user-123",
    )

    client.complete(call)

    span = only_span(exporter)
    assert span.name == "ai.derive_recipe_json"
    assert json.loads(str(attrs(span)["llm.invocation_parameters"]))["response_format"] == {
        "type": "json_object"
    }


def test_failed_call_gives_an_error_span_with_the_exception(
    client: TracingChatClient, inner: FakeInner, exporter: InMemorySpanExporter
) -> None:
    inner.error = UpstreamError("AI provider returned HTTP 500")

    with pytest.raises(UpstreamError):
        client.complete(CALL)

    span = only_span(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]
    assert "output.value" not in attrs(span)


def test_zero_token_counts_are_kept(
    client: TracingChatClient, inner: FakeInner, exporter: InMemorySpanExporter
) -> None:
    inner.reply = Completion(content="", prompt_tokens=50, completion_tokens=0)

    client.complete(CALL)

    a = attrs(only_span(exporter))
    assert (a["llm.token_count.completion"], a["llm.token_count.total"]) == (0, 50)


def test_failed_span_still_shows_what_was_sent(
    client: TracingChatClient, inner: FakeInner, exporter: InMemorySpanExporter
) -> None:
    inner.error = UpstreamError("timeout")

    with pytest.raises(UpstreamError):
        client.complete(CALL)

    a = attrs(only_span(exporter))
    assert a["llm.input_messages.2.message.content"] == "Något med kikärtor"
    assert json.loads(str(a["input.value"])) == [m.model_dump() for m in CALL.messages]
