from fastapi import APIRouter

from app.api.v4.debug_websocket import router as debug_websocket_router
from app.api.v4.inference import router as inference_router
from app.api.v4.mobile import router as mobile_router
from app.api.v4.websocket import router as websocket_router


router = APIRouter(prefix="/v4")
router.include_router(debug_websocket_router)
router.include_router(inference_router)
router.include_router(websocket_router)
router.include_router(mobile_router)
