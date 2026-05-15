from fastapi import APIRouter, WebSocket

from app.api.v3.websocket import run_latest_pending_inference_stream


router = APIRouter(prefix="/debug", tags=["v4-debug-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await run_latest_pending_inference_stream(
        websocket,
        runner_name="bamti-torch-debug-raw",
        runtime_metadata={"apiVersion": "v4-debug"},
        include_debug_raw_detections=True,
    )
