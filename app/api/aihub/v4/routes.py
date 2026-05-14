from fastapi import APIRouter

from app.api.aihub.inference import router as inference_router
from app.api.aihub.mobile import router as mobile_router
from app.api.aihub.websocket_v4 import router as websocket_router


router = APIRouter(prefix="/v4")
router.include_router(inference_router)
router.include_router(websocket_router)
router.include_router(mobile_router)
