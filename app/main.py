from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings
from app.ws.inference import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="DMS Backend")
    if settings.environment == "local" and settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")
    return app


app = create_app()
