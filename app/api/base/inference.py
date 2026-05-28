from fastapi import APIRouter

from app.inference.manifest import get_model_manifest
from app.inference.schemas import ModelManifest


router = APIRouter(tags=["base-inference"])


@router.get("/detection-classes", response_model=ModelManifest)
async def get_detection_classes() -> ModelManifest:
    return get_model_manifest("base-torch")
