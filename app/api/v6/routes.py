from fastapi import APIRouter

from app.api.v6.inference import router as inference_router
from app.api.v6.mobile import router as mobile_router
from app.api.v6.websocket import router as websocket_router


router = APIRouter(prefix="/v6")
router.include_router(inference_router)
router.include_router(websocket_router)
router.include_router(mobile_router)
