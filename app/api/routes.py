from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.v1 import router as v1_router
from app.api.v2 import router as v2_router
from app.api.v3 import router as v3_router
from app.api.v4 import router as v4_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(v1_router)
api_router.include_router(v2_router)
api_router.include_router(v3_router)
api_router.include_router(v4_router)
