"""Tracing for AI calls, as a wrapper around any ChatCompletionClient.

This is the decorator pattern: TracingChatClient implements the same protocol as
the client it wraps, adds a span around each call and delegates the actual work.
The endpoint does not know whether it talks to a traced or an untraced client.

Span attributes follow the OpenInference conventions that Phoenix understands:
https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
"""

import json

from opentelemetry.trace import StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from chef_backend.ai.client import (
    ChatCompletionClient,
    Completion,
    CompletionCall,
    invocation_parameters,
)


class TracingChatClient:
    def __init__(self, inner: ChatCompletionClient, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def complete(self, call: CompletionCall) -> Completion:
        # Everything known before the call goes on the span from the start, so a failed
        # call still shows what was sent. The system prompt is deliberately left out.
        attributes: dict[str, AttributeValue] = {
            "openinference.span.kind": "LLM",
            "llm.provider": "berget",
            "llm.system": call.config.model.partition("/")[0],
            "llm.model_name": call.config.model,
            "llm.invocation_parameters": json.dumps(invocation_parameters(call.config)),
            "chef.purpose": call.purpose.value,
            "user.id": call.user_id,
            "input.value": json.dumps(
                [message.model_dump() for message in call.messages], ensure_ascii=False
            ),
            "input.mime_type": "application/json",
        }
        # OTel attributes cannot hold lists of objects, so OpenInference flattens the
        # conversation into one key pair per message, with the index in the key.
        for i, message in enumerate(call.messages):
            attributes[f"llm.input_messages.{i}.message.role"] = message.role
            attributes[f"llm.input_messages.{i}.message.content"] = message.content

        # An exception leaving the block is recorded on the span and sets status ERROR.
        with self._tracer.start_as_current_span(
            f"ai.{call.purpose.value}", attributes=attributes
        ) as span:
            completion = self._inner.complete(call)
            span.set_attributes(
                {
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": completion.content,
                    "output.value": completion.content,
                    "output.mime_type": "text/plain",
                }
            )
            prompt, generated = completion.prompt_tokens, completion.completion_tokens
            if prompt is not None and generated is not None:
                span.set_attributes(
                    {
                        "llm.token_count.prompt": prompt,
                        "llm.token_count.completion": generated,
                        "llm.token_count.total": prompt + generated,
                    }
                )
            span.set_status(StatusCode.OK)
        return completion
