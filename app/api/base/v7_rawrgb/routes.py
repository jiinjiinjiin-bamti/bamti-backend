from fastapi import APIRouter

from app.api.base.inference import router as inference_router
from app.api.base.v7_rawrgb.websocket import router as websocket_router


router = APIRouter(prefix="/v7-rawrgb")
router.include_router(inference_router)
router.include_router(websocket_router)
