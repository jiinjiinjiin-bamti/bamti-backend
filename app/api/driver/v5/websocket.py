from fastapi import APIRouter, WebSocket

from app.api.v3.websocket import run_latest_pending_inference_stream


router = APIRouter(tags=["driver4-v5-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await run_latest_pending_inference_stream(
        websocket,
        runner_name="driver4-torch-compiled",
        runtime_metadata={
            "modelProfile": "driver4",
            "apiVersion": "v5",
            "queuePolicy": "latest_pending_only",
            "runtime": "torch_compile",
        },
    )
