"""Request and response bodies for the AI proxy.

The limits bound what a single request can cost. They are generous compared to
what the apps send today (user input is capped at 4,000 characters client-side;
history is compacted to the last 20 entries).
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chef_backend.ai.purposes import Purpose

MAX_MESSAGES = 100
MAX_MESSAGE_CHARS = 100_000
MAX_TOTAL_CHARS = 200_000


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # No "system": system prompts are chosen by the server from the purpose.
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]


class CompletionRequest(BaseModel):
    # extra="forbid": a client cannot smuggle in model, temperature or a system prompt.
    model_config = ConfigDict(extra="forbid")

    purpose: Purpose
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=MAX_MESSAGES)]

    @model_validator(mode="after")
    def check_total_size(self) -> Self:
        if sum(len(message.content) for message in self.messages) > MAX_TOTAL_CHARS:
            raise ValueError(f"messages exceed {MAX_TOTAL_CHARS} characters in total")
        return self


class CompletionResponse(BaseModel):
    content: str
