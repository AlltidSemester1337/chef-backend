"""Spec for normalize_messages: the message shape each purpose accepts.

The "accepted" fixtures replicate exactly what the Android 4.2 client sends
(see the request-shape table in the pass 7b handover). They must keep working
without an app change.
"""

from typing import Literal

import pytest
from pydantic import ValidationError

from chef_backend.ai.models import ChatMessage, CompletionRequest, normalize_messages
from chef_backend.ai.purposes import CONVERSATION_PURPOSES, Purpose

Role = Literal["user", "assistant"]
SINGLE_SHOT = sorted(set(Purpose) - CONVERSATION_PURPOSES)
CONVERSATIONS = sorted(CONVERSATION_PURPOSES)


def msgs(*turns: tuple[Role, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in turns]


def history(exchanges: int) -> list[tuple[Role, str]]:
    turns: list[tuple[Role, str]] = []
    for n in range(exchanges):
        turns.append(("user", f"question {n}"))
        turns.append(("assistant", f"answer {n}"))
    return turns


def roles(messages: list[ChatMessage]) -> list[str]:
    return [m.role for m in messages]


# --- What the 4.2 client sends: accepted unchanged ------------------------------

ACCEPTED: dict[str, tuple[Purpose, list[ChatMessage]]] = {
    "chat, first message": (Purpose.CHAT, msgs(("user", "Vad lagar jag ikväll?"))),
    "chat, preferences + history + recipe context + new": (
        Purpose.CHAT,
        msgs(
            ("user", "My preferences: vegetarian"),
            ("assistant", "Noted."),
            *history(10),
            ("user", "Recipe context: Chana masala ..."),
            ("assistant", "Got it."),
            ("user", "Kan jag frysa den?"),
        ),
    ),
    "chat, long session (89 messages)": (
        Purpose.CHAT,
        msgs(*history(44), ("user", "En fråga till")),
    ),
    "overlay_default, first message": (
        Purpose.OVERLAY_DEFAULT,
        msgs(("user", "Hur länge kokar man ägg?")),
    ),
    "overlay_default, follow-ups": (
        Purpose.OVERLAY_DEFAULT,
        msgs(*history(3), ("user", "Och löskokta?")),
    ),
    "overlay_recipe_context": (
        Purpose.OVERLAY_RECIPE_CONTEXT,
        msgs(
            ("user", "Recipe: Pasta carbonara ..."),
            ("assistant", "I have the recipe. What would you like to know?"),
            ("user", "Kan jag byta guanciale mot bacon?"),
        ),
    ),
    **{
        f"{purpose.value}, single message": (purpose, msgs(("user", "Some text")))
        for purpose in SINGLE_SHOT
    },
}


@pytest.mark.parametrize(("purpose", "messages"), ACCEPTED.values(), ids=ACCEPTED.keys())
def test_client_shapes_are_accepted_unchanged(
    purpose: Purpose, messages: list[ChatMessage]
) -> None:
    assert normalize_messages(purpose, messages) == messages


# --- Conversations: consecutive same-role turns are merged ---------------------


def test_two_user_turns_in_a_row_are_merged() -> None:
    # The client drops blank messages, so a blank stored reply leaves user, user.
    messages = msgs(
        ("user", "first"),
        ("user", "second"),
        ("assistant", "answer"),
        ("user", "new"),
    )

    result = normalize_messages(Purpose.CHAT, messages)

    assert result == msgs(("user", "first\n\nsecond"), ("assistant", "answer"), ("user", "new"))


def test_consecutive_assistant_turns_are_merged() -> None:
    messages = msgs(("user", "q"), ("assistant", "a1"), ("assistant", "a2"), ("user", "q2"))

    assert normalize_messages(Purpose.OVERLAY_DEFAULT, messages) == msgs(
        ("user", "q"), ("assistant", "a1\n\na2"), ("user", "q2")
    )


def test_runs_longer_than_two_are_merged_in_order() -> None:
    messages = msgs(("user", "a"), ("user", "b"), ("user", "c"))

    assert normalize_messages(Purpose.CHAT, messages) == msgs(("user", "a\n\nb\n\nc"))


def test_merging_does_not_change_the_input_list() -> None:
    messages = msgs(("user", "a"), ("user", "b"))
    original = list(messages)

    normalize_messages(Purpose.CHAT, messages)

    assert messages == original


@pytest.mark.parametrize("purpose", CONVERSATIONS)
def test_merged_result_always_alternates(purpose: Purpose) -> None:
    messages = msgs(
        ("user", "1"),
        ("user", "2"),
        ("assistant", "3"),
        ("assistant", "4"),
        ("user", "5"),
        ("assistant", "6"),
        ("user", "7"),
        ("user", "8"),
    )

    result = normalize_messages(purpose, messages)

    assert roles(result) == ["user", "assistant", "user", "assistant", "user"]


# --- Rejected shapes -----------------------------------------------------------

REJECTED: dict[str, tuple[Purpose, list[ChatMessage]]] = {
    **{
        f"{purpose.value}, two messages": (purpose, msgs(("user", "a"), ("user", "b")))
        for purpose in SINGLE_SHOT
    },
    **{
        f"{purpose.value}, assistant message": (purpose, msgs(("assistant", "a")))
        for purpose in SINGLE_SHOT
    },
    "derive_recipe_json, a conversation": (
        Purpose.DERIVE_RECIPE_JSON,
        msgs(("user", "a"), ("assistant", "b"), ("user", "c")),
    ),
    **{
        f"{purpose.value}, starts with assistant": (
            purpose,
            msgs(("assistant", "a"), ("user", "b")),
        )
        for purpose in CONVERSATIONS
    },
    **{
        f"{purpose.value}, ends with assistant": (purpose, msgs(("user", "a"), ("assistant", "b")))
        for purpose in CONVERSATIONS
    },
    "chat, only assistant turns": (Purpose.CHAT, msgs(("assistant", "a"), ("assistant", "b"))),
}


@pytest.mark.parametrize(("purpose", "messages"), REJECTED.values(), ids=REJECTED.keys())
def test_wrong_shapes_raise_value_error(purpose: Purpose, messages: list[ChatMessage]) -> None:
    with pytest.raises(ValueError):
        normalize_messages(purpose, messages)


# --- Wired into the request model ----------------------------------------------


def test_request_model_returns_normalized_messages() -> None:
    request = CompletionRequest.model_validate(
        {
            "purpose": "chat",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
        }
    )

    assert request.messages == msgs(("user", "first\n\nsecond"))


def test_request_model_rejects_wrong_shapes() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest.model_validate(
            {
                "purpose": "recipe_adjust",
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
            }
        )


@pytest.mark.parametrize(
    ("purpose", "messages"),
    [
        (Purpose.RECIPE_ADJUST, msgs(("user", "PRIVATE allergy notes"), ("user", "x"))),
        (Purpose.CHAT, msgs(("assistant", "PRIVATE allergy notes"))),
    ],
    ids=["single-shot", "conversation"],
)
def test_error_messages_do_not_contain_message_content(
    purpose: Purpose, messages: list[ChatMessage]
) -> None:
    with pytest.raises(ValueError) as raised:
        normalize_messages(purpose, messages)

    assert "PRIVATE" not in str(raised.value)
