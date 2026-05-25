from fastapi import APIRouter

from app.api.driver.inference import router as inference_router
from app.api.driver.v7.mobile import router as mobile_router
from app.api.driver.v7 import pre_analysis_websocket, websocket


router = APIRouter(prefix="/v7")
router.include_router(inference_router)
router.include_router(websocket.router)
router.include_router(pre_analysis_websocket.router)
router.include_router(mobile_router)
