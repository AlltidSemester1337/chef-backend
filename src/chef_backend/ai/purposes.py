"""Server-side registry of AI purposes.

A client only names a purpose. Everything that shapes the model call (system
prompt, model, sampling parameters, JSON mode) and whether the call counts
against the user's quota is decided here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources

MISTRAL_SMALL_3_2 = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


class Purpose(StrEnum):
    CHAT = "chat"
    DERIVE_RECIPE_JSON = "derive_recipe_json"
    EXTRACT_PREFERENCES = "extract_preferences"
    COMPACT_HISTORY = "compact_history"
    RECIPE_ADJUST = "recipe_adjust"
    OVERLAY_DEFAULT = "overlay_default"
    OVERLAY_RECIPE_CONTEXT = "overlay_recipe_context"


@dataclass(frozen=True, slots=True)
class PurposeConfig:
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    model: str = MISTRAL_SMALL_3_2
    json_mode: bool = False
    # Interactive purposes count against the beta quota. Background purposes
    # (extraction, compaction) follow from an interaction that already counted.
    counts_toward_quota: bool = False


def load_public_prompt(purpose: Purpose) -> str:
    """Read a prompt shipped with the package (src/chef_backend/ai/prompts/)."""
    path = resources.files("chef_backend.ai").joinpath("prompts", f"{purpose.value}.txt")
    return path.read_text(encoding="utf-8").strip()


def build_purposes(chat_prompt: str) -> Mapping[Purpose, PurposeConfig]:
    """All purposes. The main chat prompt is private, so it is passed in rather than shipped."""
    return {
        Purpose.CHAT: PurposeConfig(
            system_prompt=chat_prompt,
            temperature=1.0,
            top_p=0.95,
            max_tokens=8192,
            counts_toward_quota=True,
        ),
        Purpose.DERIVE_RECIPE_JSON: PurposeConfig(
            system_prompt=load_public_prompt(Purpose.DERIVE_RECIPE_JSON),
            temperature=0.2,
            top_p=0.95,
            max_tokens=8192,
            json_mode=True,
        ),
        Purpose.EXTRACT_PREFERENCES: PurposeConfig(
            system_prompt=load_public_prompt(Purpose.EXTRACT_PREFERENCES),
            temperature=0.2,
            top_p=0.95,
            max_tokens=8192,
            json_mode=True,
        ),
        Purpose.COMPACT_HISTORY: PurposeConfig(
            system_prompt=load_public_prompt(Purpose.COMPACT_HISTORY),
            temperature=0.2,
            top_p=0.95,
            max_tokens=8192,
        ),
        Purpose.RECIPE_ADJUST: PurposeConfig(
            system_prompt=load_public_prompt(Purpose.RECIPE_ADJUST),
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            counts_toward_quota=True,
        ),
        Purpose.OVERLAY_DEFAULT: PurposeConfig(
            system_prompt=load_public_prompt(Purpose.OVERLAY_DEFAULT),
            temperature=1.0,
            top_p=0.95,
            max_tokens=2048,
            counts_toward_quota=True,
        ),
        Purpose.OVERLAY_RECIPE_CONTEXT: PurposeConfig(
            system_prompt=load_public_prompt(Purpose.OVERLAY_RECIPE_CONTEXT),
            temperature=1.0,
            top_p=0.95,
            max_tokens=2048,
            counts_toward_quota=True,
        ),
    }
