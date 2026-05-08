import json
import time
from json import JSONDecodeError

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.api.v2.schemas import FrameMetaMessage, SessionEndMessage, SessionStartMessage
from app.core.config import settings
from app.inference.manifest import get_runner


router = APIRouter(tags=["v2-inference"])


def _server_time_ms() -> float:
    return round(time.time() * 1000, 3)


async def _send_error(websocket: WebSocket, code: str, message: str, frame_id: str | None = None) -> None:
    payload = {
        "type": "error",
        "code": code,
        "message": message,
    }
    if frame_id is not None:
        payload["frameId"] = frame_id
    await websocket.send_json(payload)


def _parse_json_message(raw_message: str) -> dict:
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("WebSocket text messages must be JSON objects.")
    return payload


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id: str | None = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = _parse_json_message(raw_message)
            except (JSONDecodeError, ValueError) as exc:
                await _send_error(websocket, "invalid_json", str(exc))
                continue

            message_type = payload.get("type")

            if message_type == "session_start":
                try:
                    session_start = SessionStartMessage.model_validate(payload)
                except ValidationError as exc:
                    await _send_error(websocket, "invalid_session_start", exc.errors()[0]["msg"])
                    continue

                session_id = session_start.session_id
                await websocket.send_json(
                    {
                        "type": "session_started",
                        "sessionId": session_id,
                        "serverTime": _server_time_ms(),
                        "transport": "websocket",
                    },
                )
                continue

            if message_type == "session_end":
                try:
                    session_end = SessionEndMessage.model_validate(payload)
                except ValidationError as exc:
                    await _send_error(websocket, "invalid_session_end", exc.errors()[0]["msg"])
                    continue

                await websocket.send_json(
                    {
                        "type": "session_ended",
                        "sessionId": session_end.session_id,
                        "serverTime": _server_time_ms(),
                    },
                )
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                return

            if message_type != "frame_meta":
                await _send_error(websocket, "unsupported_message_type", f"Unsupported message type: {message_type}")
                continue

            if session_id is None:
                await _send_error(websocket, "session_not_started", "Send session_start before frame_meta.")
                continue

            try:
                frame_meta = FrameMetaMessage.model_validate(payload)
            except ValidationError as exc:
                await _send_error(websocket, "invalid_frame_meta", exc.errors()[0]["msg"])
                continue

            if frame_meta.session_id != session_id:
                await _send_error(websocket, "session_mismatch", "frame_meta sessionId does not match active session.", frame_meta.frame_id)
                continue

            server_received_at = _server_time_ms()
            frame_bytes = await websocket.receive_bytes()
            if not frame_bytes:
                await _send_error(websocket, "empty_frame", "Frame binary message must not be empty.", frame_meta.frame_id)
                continue
            if len(frame_bytes) > settings.max_frame_bytes:
                await _send_error(
                    websocket,
                    "frame_too_large",
                    f"Frame exceeds max_frame_bytes={settings.max_frame_bytes}.",
                    frame_meta.frame_id,
                )
                continue

            runner = get_runner(settings.inference_runner)
            result = await runner.infer(frame_bytes)
            server_responded_at = _server_time_ms()

            await websocket.send_json(
                {
                    "type": "inference_result",
                    "sessionId": session_id,
                    "frameId": frame_meta.frame_id,
                    "clientSentAt": frame_meta.client_sent_at,
                    "serverReceivedAt": server_received_at,
                    "serverRespondedAt": server_responded_at,
                    "detections": [detection.model_dump(by_alias=True) for detection in result.detections],
                    "model": result.model.model_dump(by_alias=True),
                    "telemetry": result.telemetry.model_dump(by_alias=True),
                },
            )
    except WebSocketDisconnect:
        return
    finally:
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
