from fastapi import APIRouter

from app.api.v5.websocket import router as websocket_router


router = APIRouter(prefix="/v5")
router.include_router(websocket_router)
