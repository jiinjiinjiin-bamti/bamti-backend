from fastapi import APIRouter

from app.api.base.v7 import router as v7_router


router = APIRouter(prefix="/base")
router.include_router(v7_router)
