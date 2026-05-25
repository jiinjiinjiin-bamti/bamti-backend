from fastapi import APIRouter

from app.api.driver.v4 import router as v4_router
from app.api.driver.v6 import router as v6_router


router = APIRouter(prefix="/driver")
router.include_router(v4_router)
router.include_router(v6_router)
