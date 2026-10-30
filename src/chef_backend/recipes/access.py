"""Who may see which recipe.

The repository reads with the Admin SDK, which bypasses the security rules, so every
read the backend serves goes through this policy. It is deliberately stricter than
the rules (which let any signed-in user read any recipe): a user sees their own
recipes and the ones shared with the community, nothing else.
"""

from chef_backend.recipes.models import Recipe


def is_visible_to(recipe: Recipe, uid: str) -> bool:
    """Own recipes and community recipes (`isFavourite`, including orphaned ones)."""
    return recipe.uid == uid or recipe.is_favourite
