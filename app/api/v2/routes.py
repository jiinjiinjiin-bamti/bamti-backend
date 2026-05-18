from fastapi import APIRouter

from app.api.v2.websocket import router as websocket_router


router = APIRouter(prefix="/v2")
router.include_router(websocket_router)
