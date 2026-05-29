from typing import Any

from fastapi import APIRouter, WebSocket

from app.api.driver.v7.risk import create_driver4_v7_risk_scorer
from app.api.v3.websocket import QueuedFrame, run_latest_pending_inference_stream
from app.api.v6.schemas import FrameMetaMessage


router = APIRouter(tags=["driver4-v7-fast-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    risk_scorer = create_driver4_v7_risk_scorer()

    def session_metadata() -> dict[str, Any]:
        nonlocal risk_scorer
        risk_scorer = create_driver4_v7_risk_scorer()
        return {
            "modelProfile": "driver4",
            "apiVersion": "v7-fast",
            "riskScoring": risk_scorer.metadata(),
        }

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
            "apiVersion": "v7-fast",
            "optimizedPayload": True,
        },
        session_metadata_factory=session_metadata,
        result_payload_factory=risk_payload,
        frame_meta_model=FrameMetaMessage,
        include_model_in_results=False,
        include_queue_policy_in_results=False,
        include_queue_pending_in_results=False,
    )
