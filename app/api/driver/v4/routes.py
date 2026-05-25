from fastapi import APIRouter

from app.api.driver.inference import router as inference_router
from app.api.driver.v4.mobile import router as mobile_router
from app.api.driver.v4.websocket import router as websocket_router


router = APIRouter(prefix="/v4")
router.include_router(inference_router)
router.include_router(websocket_router)
router.include_router(mobile_router)
