from typing import Any, Literal

from fastapi import APIRouter, WebSocket
from pydantic import BaseModel, ConfigDict, Field

from app.api.driver.v7.risk import create_driver4_v7_risk_scorer
from app.api.v3.websocket import QueuedFrame, run_latest_pending_inference_stream


router = APIRouter(tags=["driver4-v7-rawrgb-inference"])


class RawRgbFrameMetaMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["frame_meta"]
    session_id: str = Field(alias="sessionId", min_length=1)
    frame_id: str = Field(alias="frameId", min_length=1)
    client_sent_at: str = Field(alias="clientSentAt", min_length=1)
    content_type: Literal["application/x-rgb24"] = Field(alias="contentType")
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    encoding_ms: float | None = Field(default=None, alias="encodingMs", ge=0.0)
    frame_time_seconds: float | None = Field(default=None, alias="frameTimeSeconds", ge=0.0)


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    risk_scorer = create_driver4_v7_risk_scorer()

    def session_metadata() -> dict[str, Any]:
        nonlocal risk_scorer
        risk_scorer = create_driver4_v7_risk_scorer()
        return {
            "modelProfile": "driver4",
            "apiVersion": "v7-rawrgb",
            "frameEncoding": "raw_rgb_224",
            "riskScoring": risk_scorer.metadata(),
        }

    async def infer_raw_rgb(runner: Any, frame: QueuedFrame) -> Any:
        return await runner.infer_raw_rgb(frame.frame_bytes, frame.meta.width, frame.meta.height)

    def risk_payload(result: Any, frame: QueuedFrame) -> dict[str, Any]:
        return {
            "riskScores": risk_scorer.update(
                {detection.variable_name: detection.score for detection in result.detections},
                now=frame.meta.frame_time_seconds,
            ),
        }

    await run_latest_pending_inference_stream(
        websocket,
        runner_name="driver4-torch",
        runtime_metadata={
            "modelProfile": "driver4",
            "apiVersion": "v7-rawrgb",
            "frameEncoding": "raw_rgb_224",
            "optimizedPayload": True,
        },
        session_metadata_factory=session_metadata,
        inference_factory=infer_raw_rgb,
        result_payload_factory=risk_payload,
        frame_meta_model=RawRgbFrameMetaMessage,
        include_model_in_results=False,
        include_queue_policy_in_results=False,
        include_queue_pending_in_results=False,
    )
