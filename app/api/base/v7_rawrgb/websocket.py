from typing import Any

from fastapi import APIRouter, WebSocket

from app.api.driver.v7.risk import create_driver4_v7_risk_scorer
from app.api.driver.v7_rawrgb.websocket import RawRgbFrameMetaMessage
from app.api.v3.websocket import QueuedFrame, run_latest_pending_inference_stream


router = APIRouter(tags=["base-v7-rawrgb-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    risk_scorer = create_driver4_v7_risk_scorer()

    def session_metadata() -> dict[str, Any]:
        nonlocal risk_scorer
        risk_scorer = create_driver4_v7_risk_scorer()
        return {
            "modelProfile": "base",
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
        runner_name="base-torch",
        runtime_metadata={
            "modelProfile": "base",
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
