import logging
import time
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.inference.adapter import build_detection_scores
from app.inference.manifest import get_model_manifest, get_runner
from app.inference.schemas import DetectionScore, ModelManifest

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


class InferenceTelemetry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    processing_ms: float = Field(alias="processingMs", ge=0.0)
    processing_fps: float = Field(alias="processingFps", ge=0.0)


class InferenceFrameResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    frame_id: str = Field(alias="frameId")
    client_sent_at: str = Field(alias="clientSentAt")
    detections: list[DetectionScore]
    telemetry: InferenceTelemetry


class TelemetryRunAcceptedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: Literal["accepted"]
    created_at: datetime = Field(alias="createdAt")


@router.get("/detection-classes", response_model=ModelManifest)
async def get_detection_classes() -> ModelManifest:
    return get_model_manifest(settings.inference_runner)


@router.post("/inference/frame", response_model=InferenceFrameResponse)
async def infer_frame(
    frame: Annotated[UploadFile, File()],
    frame_id: Annotated[str, Form(alias="frameId")],
    client_sent_at: Annotated[str, Form(alias="clientSentAt")],
) -> InferenceFrameResponse:
    if frame.content_type != "image/jpeg":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image/jpeg frames are supported.",
        )

    frame_bytes = await frame.read(settings.max_frame_bytes + 1)
    if not frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Frame file must not be empty.",
        )
    if len(frame_bytes) > settings.max_frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Frame exceeds max_frame_bytes={settings.max_frame_bytes}.",
        )

    manifest = get_model_manifest(settings.inference_runner)
    runner = get_runner(settings.inference_runner)

    started_at = time.perf_counter()
    result = await runner.infer(frame_bytes)
    elapsed_seconds = time.perf_counter() - started_at
    processing_ms = round(elapsed_seconds * 1000.0, 3)
    processing_fps = round(1.0 / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0

    return InferenceFrameResponse(
        frame_id=frame_id,
        client_sent_at=client_sent_at,
        detections=build_detection_scores(result=result, manifest=manifest),
        telemetry=InferenceTelemetry(
            processing_ms=processing_ms,
            processing_fps=processing_fps,
        ),
    )


@router.post(
    "/telemetry/runs",
    response_model=TelemetryRunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_telemetry_run(
    payload: Annotated[dict[str, Any], Body()],
) -> TelemetryRunAcceptedResponse:
    logger.info(
        "telemetry_run_accepted",
        extra={"payload_keys": sorted(payload.keys())[:20]},
    )
    return TelemetryRunAcceptedResponse(
        id=f"temp-{uuid4().hex}",
        status="accepted",
        created_at=datetime.now(UTC),
    )
