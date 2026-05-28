from fastapi import APIRouter

from app.api.driver.v1.inference import router as inference_router


router = APIRouter(prefix="/v1")
router.include_router(inference_router)
