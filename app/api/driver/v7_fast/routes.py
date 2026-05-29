from fastapi import APIRouter

from app.api.driver.inference import router as inference_router
from app.api.driver.v7_fast.websocket import router as websocket_router


router = APIRouter(prefix="/v7-fast")
router.include_router(inference_router)
router.include_router(websocket_router)
