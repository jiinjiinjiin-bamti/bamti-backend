from fastapi import APIRouter

from app.api.driver.v1 import router as v1_router
from app.api.driver.v2 import router as v2_router
from app.api.driver.v3 import router as v3_router
from app.api.driver.v4 import router as v4_router
from app.api.driver.v4_1 import router as v4_1_router
from app.api.driver.v5 import router as v5_router
from app.api.driver.v6 import router as v6_router
from app.api.driver.v7 import router as v7_router


router = APIRouter(prefix="/driver")
router.include_router(v1_router)
router.include_router(v2_router)
router.include_router(v3_router)
router.include_router(v4_router)
router.include_router(v4_1_router)
router.include_router(v5_router)
router.include_router(v6_router)
router.include_router(v7_router)
