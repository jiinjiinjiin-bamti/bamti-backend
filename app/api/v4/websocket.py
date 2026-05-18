from app.api.v3.websocket import inference_stream
from fastapi import APIRouter


router = APIRouter(tags=["v4-inference"])
router.add_api_websocket_route("/inference/stream", inference_stream)
