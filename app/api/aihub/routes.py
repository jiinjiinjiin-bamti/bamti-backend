from fastapi import APIRouter

from app.api.aihub.inference import router as inference_router
from app.api.aihub.mobile import router as mobile_router
from app.api.aihub.v4 import router as v4_router
from app.api.aihub.v6 import router as v6_router
from app.api.aihub.websocket import router as websocket_router


router = APIRouter(prefix="/aihub")
router.include_router(inference_router)
router.include_router(websocket_router)
router.include_router(mobile_router)
router.include_router(v4_router)
router.include_router(v6_router)
