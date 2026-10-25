"""Request and response bodies for the AI proxy.

The limits bound what a single request can cost. They are generous compared to
what the apps send today (user input is capped at 4,000 characters client-side;
history is compacted to the last 20 entries).
"""

from itertools import groupby
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chef_backend.ai.purposes import CONVERSATION_PURPOSES, INTERNAL_PURPOSES, Purpose

MAX_MESSAGES = 100
MAX_MESSAGE_CHARS = 100_000
MAX_TOTAL_CHARS = 200_000


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # No "system": system prompts are chosen by the server from the purpose.
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]


def normalize_messages(purpose: Purpose, messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return the messages in the shape the purpose accepts, or raise ValueError (422).

    - Single-shot purposes: exactly one message, from the user.
    - Conversations (CONVERSATION_PURPOSES): consecutive messages with the same role
      are merged into one, joined with a blank line, so the result alternates. The
      first and the last message must then be from the user.

    Error messages state the rule only: message content is user data.
    """
    if purpose not in CONVERSATION_PURPOSES:
        if len(messages) != 1 or messages[0].role != "user":
            raise ValueError("this purpose takes exactly one user message")
        return list(messages)

    # groupby groups *adjacent* items with the same key: exactly one run per role change.
    merged = [_merge_run(list(run)) for _, run in groupby(messages, key=lambda m: m.role)]
    if not merged or merged[0].role != "user" or merged[-1].role != "user":
        raise ValueError("a conversation must start and end with a user message")
    return merged


def _merge_run(run: list[ChatMessage]) -> ChatMessage:
    """One message from a run of same-role messages; a single message is returned as is."""
    if len(run) == 1:
        return run[0]
    # model_copy keeps the role and skips re-validation; the total size is capped already.
    return run[0].model_copy(update={"content": "\n\n".join(m.content for m in run)})


class CompletionRequest(BaseModel):
    # extra="forbid": a client cannot smuggle in model, temperature or a system prompt.
    model_config = ConfigDict(extra="forbid")

    purpose: Purpose
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=MAX_MESSAGES)]

    @model_validator(mode="after")
    def check_size_and_shape(self) -> Self:
        if self.purpose in INTERNAL_PURPOSES:
            raise ValueError("this purpose has its own endpoint")
        if sum(len(message.content) for message in self.messages) > MAX_TOTAL_CHARS:
            raise ValueError(f"messages exceed {MAX_TOTAL_CHARS} characters in total")
        # Validation runs before the endpoint, so a bad shape costs no quota or budget.
        self.messages = normalize_messages(self.purpose, self.messages)
        return self


class CompletionResponse(BaseModel):
    content: str
