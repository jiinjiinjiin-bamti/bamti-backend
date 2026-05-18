from fastapi import APIRouter, WebSocket

from app.api.v3.websocket import run_latest_pending_inference_stream


router = APIRouter(tags=["aihub-v4-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await run_latest_pending_inference_stream(
        websocket,
        runner_name="aihub-torch",
        runtime_metadata={"modelProfile": "aihub", "apiVersion": "v4"},
    )
