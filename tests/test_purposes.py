from pathlib import Path

import pytest

from chef_backend.ai import dependencies
from chef_backend.ai.purposes import Purpose, build_purposes, load_public_prompt
from chef_backend.config import Settings


def test_every_purpose_has_a_config_and_a_prompt() -> None:
    purposes = build_purposes("private chat prompt")

    assert set(purposes) == set(Purpose)
    assert all(config.system_prompt for config in purposes.values())


def test_main_chat_uses_the_injected_private_prompt() -> None:
    assert build_purposes("private chat prompt")[Purpose.CHAT].system_prompt == (
        "private chat prompt"
    )


def test_main_chat_prompt_is_not_shipped_with_the_package() -> None:
    with pytest.raises(FileNotFoundError):
        load_public_prompt(Purpose.CHAT)


def test_only_interactive_purposes_count_toward_quota() -> None:
    counted = {p for p, c in build_purposes("x").items() if c.counts_toward_quota}

    assert counted == {
        Purpose.CHAT,
        Purpose.RECIPE_ADJUST,
        Purpose.OVERLAY_DEFAULT,
        Purpose.OVERLAY_RECIPE_CONTEXT,
    }


def test_json_purposes_use_json_mode() -> None:
    json_mode = {p for p, c in build_purposes("x").items() if c.json_mode}

    assert json_mode == {Purpose.DERIVE_RECIPE_JSON, Purpose.EXTRACT_PREFERENCES}


def test_get_purposes_reads_the_chat_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt_file = tmp_path / "chat.txt"
    prompt_file.write_text("  secret prompt\n")
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(chat_prompt_file=prompt_file)
    )
    dependencies.get_purposes.cache_clear()

    try:
        assert dependencies.get_purposes()[Purpose.CHAT].system_prompt == "secret prompt"
    finally:
        dependencies.get_purposes.cache_clear()


@pytest.mark.parametrize("content", [None, "   \n"], ids=["not configured", "empty file"])
def test_get_purposes_fails_closed_without_a_chat_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    prompt_file = None
    if content is not None:
        prompt_file = tmp_path / "chat.txt"
        prompt_file.write_text(content)
    monkeypatch.setattr(
        dependencies, "get_settings", lambda: Settings(chat_prompt_file=prompt_file)
    )
    dependencies.get_purposes.cache_clear()

    try:
        with pytest.raises(RuntimeError):
            dependencies.get_purposes()
    finally:
        dependencies.get_purposes.cache_clear()
