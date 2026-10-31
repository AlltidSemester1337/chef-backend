from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from mcp.server.auth.provider import TokenVerifier as McpTokenVerifier

from chef_backend import __doc__ as description
from chef_backend.ai.dependencies import (
    get_budget_store,
    get_chat_client,
    get_purposes,
    get_quota_store,
)
from chef_backend.ai.routes import router as ai_router
from chef_backend.auth import AuthenticatedUser, get_current_user, get_token_verifier
from chef_backend.config import Settings, get_settings
from chef_backend.cooking_resources import get_preferences_reader
from chef_backend.cooking_resources import router as cooking_resources_router
from chef_backend.images import get_image_generator, get_image_store
from chef_backend.images import router as images_router
from chef_backend.mcp_server.auth import get_mcp_token_verifier
from chef_backend.mcp_server.login import router as mcp_login_router
from chef_backend.mcp_server.oauth import ChefOAuthProvider, get_oauth_provider
from chef_backend.mcp_server.server import build_mcp, build_oauth_routes
from chef_backend.recipes.routes import get_recipe_service
from chef_backend.recipes.routes import router as recipes_router
from chef_backend.recipes.service import RecipeService
from chef_backend.telemetry import shutdown_tracing
from chef_backend.tts import get_speech_synthesizer
from chef_backend.tts import router as tts_router


def check_production_config() -> None:
    """Build every lazily created collaborator once, so missing config fails at startup.

    Cloud Run then keeps serving the previous revision instead of switching to one
    that would only fail on the first authenticated request.
    """
    get_purposes()
    get_chat_client()
    get_quota_store()
    get_budget_store()
    get_speech_synthesizer()
    get_image_generator()
    get_image_store()
    get_preferences_reader()
    get_recipe_service()
    get_oauth_provider()
    get_token_verifier()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        if settings.environment == "prod":
            check_production_config()
        # A mounted app's own lifespan never runs, so the MCP session manager starts here.
        async with mcp.session_manager.run():
            yield
        shutdown_tracing()

    app = FastAPI(
        title="Chef backend", description=description or "", version="0.1.0", lifespan=lifespan
    )

    # Not /healthz: Cloud Run reserves some paths ending in "z".
    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness probe. Returns no configuration or secrets."""
        return {"status": "ok", "environment": settings.environment}

    @app.get("/v1/me", tags=["auth"])
    def me(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> AuthenticatedUser:
        """The authenticated caller. Lets clients smoke-test their ID and App Check tokens."""
        return user

    app.include_router(ai_router)
    app.include_router(tts_router)
    app.include_router(images_router)
    app.include_router(cooking_resources_router)
    app.include_router(recipes_router)

    def resolve_token_verifier() -> McpTokenVerifier:
        return app.dependency_overrides.get(get_mcp_token_verifier, get_mcp_token_verifier)()

    def resolve_recipe_service() -> RecipeService:
        return app.dependency_overrides.get(get_recipe_service, get_recipe_service)()

    def resolve_oauth_provider() -> ChefOAuthProvider:
        return app.dependency_overrides.get(get_oauth_provider, get_oauth_provider)()

    # The MCP authorization server: the SDK's OAuth endpoints and our login page.
    app.router.routes.extend(build_oauth_routes(settings.public_base_url, resolve_oauth_provider))
    app.include_router(mcp_login_router)

    # MCP is not FastAPI-routed, but resolving through dependency_overrides lets tests
    # swap its collaborators the same way as for the HTTP routes.
    mcp = build_mcp(settings.public_base_url, resolve_token_verifier, resolve_recipe_service)
    # The router's fallback for paths no route matches: /mcp and the OAuth metadata under
    # /.well-known must sit at the top level, and every FastAPI route (also ones added
    # later) still matches first. The MCP app answers 404 for anything else.
    app.router.default = mcp.streamable_http_app()
    app.state.mcp = mcp
    return app


app = create_app()
