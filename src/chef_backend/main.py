from typing import Annotated

from fastapi import Depends, FastAPI

from chef_backend import __doc__ as description
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Chef backend", description=description or "", version="0.1.0")

    # Not /healthz: Cloud Run reserves some paths ending in "z".
    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness probe. Returns no configuration or secrets."""
        return {"status": "ok", "environment": settings.environment}

    @app.get("/v1/me", tags=["auth"])
    def me(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> AuthenticatedUser:
        """The authenticated caller. Lets clients smoke-test their ID and App Check tokens."""
        return user

    return app


app = create_app()
