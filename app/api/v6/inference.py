import time
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.v1.schemas import InferenceFrameResponse
from app.core.config import settings
from app.inference.manifest import get_model_manifest, get_runner
from app.inference.schemas import ModelManifest
from app.inference.score_averaging import RollingScoreAverager


router = APIRouter(tags=["v6-inference"])
score_averagers: dict[str, RollingScoreAverager] = {}


@router.get("/detection-classes", response_model=ModelManifest)
async def get_detection_classes() -> ModelManifest:
    return get_model_manifest(settings.inference_runner)


@router.post("/inference/frame", response_model=InferenceFrameResponse)
async def infer_frame(
    frame: Annotated[UploadFile, File()],
    frame_id: Annotated[str | None, Form(alias="frameId")] = None,
    client_sent_at: Annotated[str | None, Form(alias="clientSentAt")] = None,
    session_id: Annotated[str, Form(alias="sessionId")] = "default",
) -> InferenceFrameResponse:
    if frame.content_type != "image/jpeg":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image/jpeg frames are supported.",
        )

    server_received_at = time.time() * 1000
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

    runner = get_runner(settings.inference_runner)
    result = await runner.infer(frame_bytes)
    averager = score_averagers.setdefault(session_id, RollingScoreAverager(window_seconds=1.0))
    averaged_detections = averager.average(result.detections)
    server_responded_at = time.time() * 1000

    return InferenceFrameResponse(
        frame_id=frame_id,
        client_sent_at=client_sent_at,
        server_received_at=round(server_received_at, 3),
        server_responded_at=round(server_responded_at, 3),
        detections=averaged_detections,
        model=result.model,
        telemetry=result.telemetry,
    )
