from fastapi import APIRouter

from app.api.v1.inference import router as inference_router
from app.api.v1.telemetry import router as telemetry_router


router = APIRouter(prefix="/v1")
router.include_router(inference_router)
router.include_router(telemetry_router)
