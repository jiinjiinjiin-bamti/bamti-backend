from fastapi import FastAPI

from app.api.routes import api_router
from app.ws.inference import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="DMS Backend")
    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")
    return app


app = create_app()
