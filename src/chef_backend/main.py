from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI

from chef_backend import __doc__ as description
from chef_backend.ai.dependencies import get_chat_client, get_purposes, get_quota_store
from chef_backend.ai.routes import router as ai_router
from chef_backend.auth import AuthenticatedUser, get_current_user, get_token_verifier
from chef_backend.config import Settings, get_settings


def check_production_config() -> None:
    """Build every lazily created collaborator once, so missing config fails at startup.

    Cloud Run then keeps serving the previous revision instead of switching to one
    that would only fail on the first authenticated request.
    """
    get_purposes()
    get_chat_client()
    get_quota_store()
    get_token_verifier()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        if settings.environment == "prod":
            check_production_config()
        yield

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
    return app


app = create_app()
