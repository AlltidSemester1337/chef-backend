"""The tool-evaluation harness, offline: a scripted model instead of the Anthropic API."""

from dataclasses import dataclass, field
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message
from tool_eval_support import run


def message(*content: dict[str, Any]) -> Message:
    return Message.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "scripted",
            "content": list(content),
            "stop_reason": "tool_use" if content[-1]["type"] == "tool_use" else "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def tool_use(use_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": use_id, "name": name, "input": arguments}


@dataclass
class ScriptedMessages:
    replies: list[Message]
    requests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def create(self, **request: Any) -> Message:
        # A snapshot: the harness keeps appending to the same messages list.
        self.requests.append(request | {"messages": list(request["messages"])})
        return self.replies[len(self.requests) - 1]


@dataclass
class ScriptedClient:
    messages: ScriptedMessages


def test_tool_calls_run_against_the_server_until_the_model_answers() -> None:
    scripted = ScriptedClient(
        ScriptedMessages(
            [
                message(tool_use("u1", "search_recipes", query="vegetarian", scope="mine")),
                message(tool_use("u2", "get_recipe", recipe_id="secret")),
                message({"type": "text", "text": "You have Chana masala."}),
            ]
        )
    )

    transcript = run("Vegetarian?", client=cast(Anthropic, scripted))

    assert transcript.names() == ["search_recipes", "get_recipe"]
    assert [c.is_error for c in transcript.calls] == [False, True]
    assert transcript.answer == "You have Chana masala."
    requests = scripted.messages.requests
    # The model saw the real tools, the server's instructions and each tool result.
    assert "save_recipe_variant" in {t["name"] for t in requests[0]["tools"]}
    assert "Chef" in requests[0]["system"]
    [result] = requests[1]["messages"][-1]["content"]
    assert result["tool_use_id"] == "u1" and "Chana masala" in result["content"]
    [error] = requests[2]["messages"][-1]["content"]
    assert error["is_error"] is True and "recipe_not_found" in error["content"]


def test_a_saving_model_saves_into_the_fake_repository_only() -> None:
    scripted = ScriptedClient(
        ScriptedMessages(
            [
                message(
                    tool_use(
                        "u1",
                        "save_recipe_variant",
                        recipe_id="chana",
                        label="For two",
                        title="Chana masala for two",
                        ingredients=[{"name": "chickpeas", "quantity": "400", "unit": "g"}],
                        instructions=["Simmer."],
                    )
                ),
                message({"type": "text", "text": "Saved."}),
            ]
        )
    )

    transcript = run("Save it for two", client=cast(Anthropic, scripted))

    assert transcript.saved_variant_labels("chana") == ["For two"]
    assert transcript.saved_recipes() == []
