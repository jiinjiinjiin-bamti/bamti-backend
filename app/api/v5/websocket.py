from fastapi import APIRouter, WebSocket

from app.api.v3.websocket import run_latest_pending_inference_stream


router = APIRouter(tags=["v5-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await run_latest_pending_inference_stream(
        websocket,
        runner_name="bamti-torch-compiled",
        runtime_metadata={
            "runtime": "torch_compile",
            "queuePolicy": "latest_pending_only",
        },
    )
