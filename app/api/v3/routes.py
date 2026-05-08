from fastapi import APIRouter

from app.api.v3.websocket import router as websocket_router


router = APIRouter(prefix="/v3")
router.include_router(websocket_router)
