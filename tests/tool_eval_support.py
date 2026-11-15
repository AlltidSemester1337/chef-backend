"""Harness for the tool evaluations (test_tool_evals.py): Claude against the real MCP server.

The server runs in-process with synthetic recipes and fakes for saving, exactly as
in the MCP tests, so nothing reaches Firestore and nothing is really saved. Only
the model is real: each case sends one user message to the Anthropic API with the
server's own tools and instructions, runs every tool call against the server, and
records what happened in a Transcript for the case to check.

Costs money (API calls), so the evaluations only run when asked: see
test_tool_evals.py.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from fastapi.testclient import TestClient
from mcp_support import (
    BASE_URL,
    ME,
    OTHER,
    T0,
    FakeMcpTokenVerifier,
    FakeRecipeRepository,
    McpClient,
)
from test_mcp_save_recipe import RepositoryBackedStore
from test_recipe_saving import FakeBudget, FakeGenerator, FakeImageStore

from chef_backend.ai.dependencies import get_today
from chef_backend.main import create_app
from chef_backend.mcp_server.auth import (
    RECIPES_READ_SCOPE,
    RECIPES_WRITE_SCOPE,
    get_mcp_token_verifier,
)
from chef_backend.recipes.models import Recipe
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.saving import RecipeSaver, SaveLimits, get_recipe_saver
from chef_backend.recipes.service import RecipeService

MODEL = os.environ.get("TOOL_EVAL_MODEL", "claude-haiku-5-5")
MAX_TURNS = 8
TODAY = date(2026, 11, 8)


# --- The synthetic recipes the model can find -------------------------------------------


def _recipe(
    recipe_id: str,
    uid: str,
    title: str,
    tags: list[str],
    ingredients: list[str],
    *,
    shared: bool = False,
    day: int = 0,
) -> Recipe:
    return Recipe.model_validate(
        {
            "id": recipe_id,
            "uid": uid,
            "title": title,
            "summary": f"{title}, a {' '.join(tags[:2])} dish.",
            "servings": "4",
            "prepTime": "15 minutes",
            "cookingTime": "30 minutes",
            "difficulty": "EASY",
            "tags": tags,
            "isFavourite": shared,
            "ingredients": [{"name": name, "quantity": "200", "unit": "g"} for name in ingredients],
            "instructions": [f"Prepare the {ingredients[0]}.", "Cook everything together."],
            "updatedAt": T0 + timedelta(days=day),
        }
    )


def recipes() -> list[Recipe]:
    """The user's own recipes (one shared), community recipes and one private stranger's."""
    return [
        _recipe("chana", ME, "Chana masala", ["vegetarian", "indian"],
                ["chickpeas", "onion", "tomato"], day=1),
        _recipe("tikka", ME, "Chicken tikka masala", ["chicken", "indian"],
                ["chicken", "yoghurt", "tomato"], shared=True, day=2),
        _recipe("carbonara", ME, "Pasta carbonara", ["pasta", "italian", "quick"],
                ["spaghetti", "egg", "bacon", "onion"], day=3),
        _recipe("tacos", OTHER, "Halloumi tacos", ["vegetarian", "mexican", "quick"],
                ["halloumi", "tortilla", "cabbage"], shared=True, day=4),
        _recipe("lentils", OTHER, "Red lentil soup", ["vegan", "soup"],
                ["red lentils", "carrot", "onion"], shared=True, day=5),
        _recipe("secret", OTHER, "Secret stew", ["beef", "stew"],
                ["beef", "potato", "carrot"], day=6),
    ]  # fmt: skip


# --- What happened in one conversation ----------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    is_error: bool


@dataclass
class Transcript:
    calls: list[ToolCall] = field(default_factory=list[ToolCall])
    # The model's last text: its answer to the user (or its question back).
    answer: str = ""
    repository: FakeRecipeRepository = field(default_factory=FakeRecipeRepository)

    def calls_to(self, name: str) -> list[ToolCall]:
        return [call for call in self.calls if call.name == name]

    def called(self, name: str) -> bool:
        return bool(self.calls_to(name))

    def names(self) -> list[str]:
        return [call.name for call in self.calls]

    def saved_recipes(self) -> list[Recipe]:
        """Recipes the model saved in this conversation (the fixture ones excluded)."""
        fixture_ids = {r.id for r in recipes()}
        return [r for r in self.repository.recipes if r.id not in fixture_ids]

    def saved_variant_labels(self, recipe_id: str) -> list[str]:
        return [v.label for v in self.repository.saved_variants.get(recipe_id, [])]

    def __str__(self) -> str:
        """For assertion messages: what the model did and said."""
        lines = [
            f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
            + (" -> error" if call.is_error else "")
            for call in self.calls
        ]
        # The end matters most: that is where the model asks back or says what it did.
        answer = (
            self.answer
            if len(self.answer) <= 800
            else f"{self.answer[:300]} [...] {self.answer[-500:]}"
        )
        return "\n".join([*lines, f"answer: {answer}"])


# --- Running a conversation --------------------------------------------------------------


def _server(repository: FakeRecipeRepository) -> tuple[TestClient, McpClient]:
    verifier = FakeMcpTokenVerifier()
    saver = RecipeSaver(
        RepositoryBackedStore(repository),
        RecipeService(repository),
        FakeBudget(),
        FakeGenerator,
        FakeImageStore,
        SaveLimits(daily_saves=5, daily_images=5),
    )
    app = create_app()
    app.dependency_overrides.update(
        {
            get_mcp_token_verifier: lambda: verifier,
            get_recipe_service: lambda: RecipeService(repository),
            get_recipe_saver: lambda: saver,
            get_today: lambda: TODAY,
        }
    )
    http = TestClient(app, base_url=BASE_URL)
    token = verifier.issue(scopes=[RECIPES_READ_SCOPE, RECIPES_WRITE_SCOPE])
    return http, McpClient(http, token)


def _tools(mcp: McpClient) -> list[ToolParam]:
    tools: list[ToolParam] = [
        {"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]}
        for t in mcp.list_tools()
    ]
    # The tool list is the same in every call: cache it (cheaper and faster from turn 2).
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _instructions(mcp: McpClient) -> str:
    initialize = mcp.post(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "tool-evals", "version": "1"},
        },
    ).json()
    return initialize["result"].get("instructions", "")


def _result_text(result: dict[str, Any]) -> str:
    if "structuredContent" in result and not result["isError"]:
        return json.dumps(result["structuredContent"], ensure_ascii=False)
    return " ".join(block.get("text", "") for block in result["content"])


def run(prompt: str, client: Anthropic | None = None) -> Transcript:
    """One conversation: the user's `prompt`, then the model and the tools until it answers."""
    repository = FakeRecipeRepository(recipes=recipes())
    transcript = Transcript(repository=repository)
    http, mcp = _server(repository)
    client = client or Anthropic()  # ANTHROPIC_API_KEY from the environment
    with http:
        tools = _tools(mcp)
        # Like a Claude client: the server's instructions go into the system prompt.
        system = f"The Chef connector's instructions: {_instructions(mcp)}"
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        for _ in range(MAX_TURNS):
            response = client.messages.create(
                model=MODEL, max_tokens=2048, system=system, tools=tools, messages=messages
            )
            messages.append({"role": "assistant", "content": response.content})
            transcript.answer = "".join(b.text for b in response.content if b.type == "text")
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break
            results: list[ToolResultBlockParam] = []
            for use in tool_uses:
                arguments = cast(dict[str, Any], use.input)
                result = mcp.call(use.name, **arguments)
                transcript.calls.append(ToolCall(use.name, arguments, result["isError"]))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use.id,
                        "content": _result_text(result),
                        "is_error": result["isError"],
                    }
                )
            messages.append({"role": "user", "content": results})
    return transcript
