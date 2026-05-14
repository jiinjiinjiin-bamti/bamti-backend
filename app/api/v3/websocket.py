import asyncio
import json
import logging
import time
from dataclasses import dataclass
from json import JSONDecodeError

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.api.v2.schemas import FrameMetaMessage, SessionEndMessage, SessionStartMessage
from app.core.config import settings
from app.inference.manifest import get_runner


router = APIRouter(tags=["v3-inference"])
logger = logging.getLogger(__name__)


@dataclass
class QueuedFrame:
    meta: FrameMetaMessage
    frame_bytes: bytes
    server_received_at: float


def _server_time_ms() -> float:
    return round(time.time() * 1000, 3)


def _parse_json_message(raw_message: str) -> dict:
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("WebSocket text messages must be JSON objects.")
    return payload


async def run_latest_pending_inference_stream(
    websocket: WebSocket,
    runner_name: str | None = None,
    runtime_metadata: dict | None = None,
) -> None:
    await websocket.accept()

    session_id: str | None = None
    pending_frame: QueuedFrame | None = None
    dropped_frames = 0
    frame_available = asyncio.Event()
    queue_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    websocket_close_sent = False

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def close_websocket(code: int = status.WS_1000_NORMAL_CLOSURE) -> None:
        nonlocal websocket_close_sent

        if websocket_close_sent:
            return
        websocket_close_sent = True
        try:
            await websocket.close(code=code)
        except RuntimeError as exc:
            if "Unexpected ASGI message 'websocket.close'" not in str(exc):
                raise

    async def send_error(code: str, message: str, frame_id: str | None = None) -> None:
        payload = {
            "type": "error",
            "code": code,
            "message": message,
        }
        if frame_id is not None:
            payload["frameId"] = frame_id
        await send_json(payload)

    async def process_latest_frames() -> None:
        nonlocal pending_frame

        runner = get_runner(runner_name or settings.inference_runner)

        while not stop_event.is_set():
            await frame_available.wait()
            if stop_event.is_set():
                return

            async with queue_lock:
                frame = pending_frame
                pending_frame = None
                if pending_frame is None:
                    frame_available.clear()

            if frame is None:
                continue

            try:
                result = await runner.infer(frame.frame_bytes)
            except Exception:
                logger.exception("Latest-pending inference failed for session %s frame %s", session_id, frame.meta.frame_id)
                stop_event.set()
                await send_error("inference_failed", "WebSocket inference failed. Check backend model/runtime logs.", frame.meta.frame_id)
                await close_websocket(code=status.WS_1011_INTERNAL_ERROR)
                return
            server_responded_at = _server_time_ms()

            await send_json(
                {
                    "type": "inference_result",
                    "sessionId": session_id,
                    "frameId": frame.meta.frame_id,
                    "clientSentAt": frame.meta.client_sent_at,
                    "serverReceivedAt": frame.server_received_at,
                    "serverRespondedAt": server_responded_at,
                    "detections": [detection.model_dump(by_alias=True) for detection in result.detections],
                    "model": result.model.model_dump(by_alias=True),
                    "queue": {
                        "policy": "latest_pending_only",
                        "droppedFrames": dropped_frames,
                        "pendingFrames": 1 if pending_frame is not None else 0,
                    },
                    "telemetry": result.telemetry.model_dump(by_alias=True),
                },
            )

    processor_task = asyncio.create_task(process_latest_frames())

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = _parse_json_message(raw_message)
            except (JSONDecodeError, ValueError) as exc:
                await send_error("invalid_json", str(exc))
                continue

            message_type = payload.get("type")

            if message_type == "session_start":
                try:
                    session_start = SessionStartMessage.model_validate(payload)
                except ValidationError as exc:
                    await send_error("invalid_session_start", exc.errors()[0]["msg"])
                    continue

                session_id = session_start.session_id
                await send_json(
                    {
                        "type": "session_started",
                        "sessionId": session_id,
                        "serverTime": _server_time_ms(),
                        "transport": "websocket",
                        "queuePolicy": "latest_pending_only",
                        **(runtime_metadata or {}),
                    },
                )
                continue

            if message_type == "session_end":
                try:
                    session_end = SessionEndMessage.model_validate(payload)
                except ValidationError as exc:
                    await send_error("invalid_session_end", exc.errors()[0]["msg"])
                    continue

                stop_event.set()
                frame_available.set()
                await send_json(
                    {
                        "type": "session_ended",
                        "sessionId": session_end.session_id,
                        "serverTime": _server_time_ms(),
                        "droppedFrames": dropped_frames,
                    },
                )
                await close_websocket(code=status.WS_1000_NORMAL_CLOSURE)
                return

            if message_type != "frame_meta":
                await send_error("unsupported_message_type", f"Unsupported message type: {message_type}")
                continue

            if session_id is None:
                await send_error("session_not_started", "Send session_start before frame_meta.")
                continue

            try:
                frame_meta = FrameMetaMessage.model_validate(payload)
            except ValidationError as exc:
                await send_error("invalid_frame_meta", exc.errors()[0]["msg"])
                continue

            if frame_meta.session_id != session_id:
                await send_error("session_mismatch", "frame_meta sessionId does not match active session.", frame_meta.frame_id)
                continue

            server_received_at = _server_time_ms()
            frame_bytes = await websocket.receive_bytes()
            if not frame_bytes:
                await send_error("empty_frame", "Frame binary message must not be empty.", frame_meta.frame_id)
                continue
            if len(frame_bytes) > settings.max_frame_bytes:
                await send_error(
                    "frame_too_large",
                    f"Frame exceeds max_frame_bytes={settings.max_frame_bytes}.",
                    frame_meta.frame_id,
                )
                continue

            dropped_frame_id: str | None = None
            async with queue_lock:
                if pending_frame is not None:
                    dropped_frame_id = pending_frame.meta.frame_id
                    dropped_frames += 1
                pending_frame = QueuedFrame(
                    meta=frame_meta,
                    frame_bytes=frame_bytes,
                    server_received_at=server_received_at,
                )
                frame_available.set()

            if dropped_frame_id is not None:
                await send_json(
                    {
                        "type": "frame_dropped",
                        "sessionId": session_id,
                        "frameId": dropped_frame_id,
                        "droppedFrames": dropped_frames,
                        "reason": "replaced_by_latest",
                    },
                )
    except WebSocketDisconnect:
        return
    finally:
        stop_event.set()
        frame_available.set()
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await close_websocket()


@router.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket) -> None:
    await run_latest_pending_inference_stream(websocket)
