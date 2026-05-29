from fastapi import APIRouter

from app.api.base.v7 import router as v7_router
from app.api.base.v7_rawrgb import router as v7_rawrgb_router


router = APIRouter(prefix="/base")
router.include_router(v7_router)
router.include_router(v7_rawrgb_router)
