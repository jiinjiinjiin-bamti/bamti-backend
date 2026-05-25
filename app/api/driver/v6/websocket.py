from fastapi import APIRouter, WebSocket

from app.api.aihub.websocket import run_smoothed_inference_stream


router = APIRouter(tags=["driver4-v6-inference"])


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await run_smoothed_inference_stream(websocket, runner_name="driver4-torch", model_profile="Driver4")
