from fastapi import FastAPI

from chef_backend import __doc__ as description
from chef_backend.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Chef backend", description=description or "", version="0.1.0")

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        """Liveness probe. Returns no configuration or secrets."""
        return {"status": "ok", "environment": settings.environment}

    return app


app = create_app()
