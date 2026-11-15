"""Tool evaluations: does Claude pick the right Chef tools and arguments?

Unlike the other tests these call a real model (see tool_eval_support.py), so they
cost money and are not deterministic: they are skipped unless asked for.

    RUN_TOOL_EVALS=1 ANTHROPIC_API_KEY=... uv run pytest tests/test_tool_evals.py -v

TOOL_EVAL_MODEL picks the model (default claude-sonnet-5). A failing case prints
what the model did (the transcript) so the tool or prompt descriptions can be fixed.

The synthetic recipes (tool_eval_support.recipes()):
  the user's own:  chana (Chana masala, vegetarian), tikka (Chicken tikka masala,
                   shared), carbonara (Pasta carbonara, quick)
  community:       tacos (Halloumi tacos, vegetarian, quick), lentils (Red lentil
                   soup, vegan)
  not visible:     secret (Secret stew, someone else's private recipe)
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from tool_eval_support import ToolCall, Transcript, run

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TOOL_EVALS") != "1", reason="calls the Anthropic API; RUN_TOOL_EVALS=1"
)


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    # Raises AssertionError (plain assert) when the model did the wrong thing.
    check: Callable[[Transcript], None]


# --- An example ----------------------------------------------------------------------------


def searches_own_vegetarian_recipes(t: Transcript) -> None:
    searches = t.calls_to("search_recipes")
    assert searches, "it should search"
    # Recipes are stored in English: a Swedish request must become English keywords.
    assert any("vegetarian" in call.arguments.get("query", "").lower() for call in searches)
    assert any(call.arguments.get("scope") == "mine" for call in searches)
    assert "Chana masala" in t.answer
    assert not t.saved_recipes(), "a question must not save anything"


def searches_others_tacos_recipes(t: Transcript) -> None:
    searches = t.calls_to("search_recipes")
    assert searches, "it should search"
    assert any("tacos" in call.arguments.get("query", "").lower() for call in searches)
    assert any(
        call.arguments.get("scope") == "others" or call.arguments.get("scope") == "community"
        for call in searches
    )
    assert not any(call.arguments.get("scope") == "mine" for call in searches)
    assert "tacos" in t.answer.casefold()
    assert not t.saved_recipes(), "a question / search must not save anything"


def create_and_save_no_onion_version_owned_recipe(t: Transcript) -> None:
    save = t.calls_to("save_recipe_variant")
    assert save, "it should create a new variant"
    assert not any("onion" in name for name in ingredient_names(save)), ingredient_names(save)
    assert all(call.arguments.get("recipe_id") == "carbonara" for call in save)
    assert not t.saved_recipes(), "it should not save a new recipe"
    assert t.saved_variant_labels("carbonara"), (
        "a create variant request should save a new variant for target recipe"
    )


def create_and_save_no_onion_version_new_recipe_others_recipe(t: Transcript) -> None:
    save = t.calls_to("save_recipe")
    assert save, "it should create a new recipe"
    assert not any("onion" in name for name in ingredient_names(save)), ingredient_names(save)
    assert not t.called("save_recipe_variant"), (
        "the description says variants are only for own recipes"
    )
    assert len(t.saved_recipes()) == 1


def create_and_no_save_no_onion_version_owned_recipe(t: Transcript) -> None:
    assert not {"save_recipe", "save_recipe_variant"} & set(t.names())


def searches_own_quick_simple_recipes(t: Transcript) -> None:
    searches = t.calls_to("search_recipes")
    assert searches, "it should search"
    assert any(call.arguments.get("scope") == "mine" for call in searches)
    assert "Chana masala".casefold() in t.answer.casefold()
    assert "Chicken tikka masala".casefold() in t.answer.casefold()
    assert "Pasta carbonara".casefold() in t.answer.casefold()
    assert "Halloumi tacos".casefold() not in t.answer.casefold()
    assert not t.saved_recipes(), "a question / search must not save anything"


def ingredient_names(calls: list[ToolCall]) -> list[str]:
    """Every ingredient name the model sent, lowercased, across all the calls."""
    return [
        ingredient.get("name", "").lower()
        for call in calls
        for ingredient in call.arguments.get("ingredients", [])
    ]


CASES = [
    Case(
        id="sv-search-own-vegetarian-recipes",
        prompt="Vilka vegetariska recept har jag sparat i Chef?",
        check=searches_own_vegetarian_recipes,
    ),
    Case(
        id="sv-search-others-tacos-recipes",
        prompt="Vilka tacos recept kan du rekommendera för mig från community?",
        check=searches_others_tacos_recipes,
    ),
    Case(
        id="sv-create-save-variant-of-owned-carbonara-recipe",
        prompt="Skapa och spara en variant utan lök för mitt Pasta carbonara recept.",
        check=create_and_save_no_onion_version_owned_recipe,
    ),
    Case(
        id="sv-create-save-new-recipe-of-others-lentil-recipe",
        prompt="Skapa och spara en recept utan lök för lins receptet i community.",
        check=create_and_save_no_onion_version_new_recipe_others_recipe,
    ),
    Case(
        id="sv-search-create-no-save-variant-of-owned-recipe",
        prompt="Kan du ge mig en variant utan lök för mitt Pasta carbonara recept?",
        check=create_and_no_save_no_onion_version_owned_recipe,
    ),
    Case(
        id="sv-search-simple/fast-recipes",
        prompt="Ge mig en lista med namnen på alla mina lättlagade recept.",
        check=searches_own_quick_simple_recipes,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_the_model_uses_the_tools_well(case: Case) -> None:
    transcript = run(case.prompt)
    try:
        case.check(transcript)
    except AssertionError as error:
        raise AssertionError(f"{error}\n--- transcript ---\n{transcript}") from None
