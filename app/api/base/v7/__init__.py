from fastapi import APIRouter

from app.api.base.inference import router as inference_router
from app.api.base.v7 import websocket


router = APIRouter(prefix="/v7")
router.include_router(inference_router)
router.include_router(websocket.router)
