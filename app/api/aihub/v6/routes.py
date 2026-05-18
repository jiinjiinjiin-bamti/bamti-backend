from fastapi import APIRouter

from app.api.aihub.inference import router as inference_router
from app.api.aihub.mobile_v6 import router as mobile_router
from app.api.aihub.websocket import router as websocket_router


router = APIRouter(prefix="/v6")
router.include_router(inference_router)
router.include_router(websocket_router)
router.include_router(mobile_router)
