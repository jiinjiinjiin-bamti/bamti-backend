import asyncio
import json
import logging
import time
from dataclasses import dataclass
from json import JSONDecodeError

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.api.driver.v7.risk import create_driver4_v7_risk_scorer
from app.api.v6.schemas import FrameMetaMessage, SessionEndMessage, SessionStartMessage
from app.core.config import settings
from app.inference.manifest import get_runner


router = APIRouter(tags=["driver4-v7-pre-analysis"])
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


@router.websocket("/pre-analysis/stream")
async def pre_analysis_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    session_id: str | None = None
    frame_queue: asyncio.Queue[QueuedFrame] = asyncio.Queue()
    send_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    risk_scorer = create_driver4_v7_risk_scorer()
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
        payload = {"type": "error", "code": code, "message": message}
        if frame_id is not None:
            payload["frameId"] = frame_id
        await send_json(payload)

    async def process_frames_in_order() -> None:
        runner = get_runner("driver4-torch")
        while not stop_event.is_set():
            frame = await frame_queue.get()
            try:
                result = await runner.infer(frame.frame_bytes)
            except Exception:
                logger.exception("Driver4 v7 pre-analysis failed for session %s frame %s", session_id, frame.meta.frame_id)
                stop_event.set()
                await send_error("inference_failed", "Driver4 v7 pre-analysis failed. Check backend model/runtime logs.", frame.meta.frame_id)
                await close_websocket(code=status.WS_1011_INTERNAL_ERROR)
                return
            finally:
                frame_queue.task_done()

            risk_scores = risk_scorer.update(
                {detection.variable_name: detection.score for detection in result.detections},
                now=frame.meta.frame_time_seconds,
            )
            await send_json(
                {
                    "type": "inference_result",
                    "sessionId": session_id,
                    "frameId": frame.meta.frame_id,
                    "clientSentAt": frame.meta.client_sent_at,
                    "serverReceivedAt": frame.server_received_at,
                    "serverRespondedAt": _server_time_ms(),
                    "detections": [detection.model_dump(by_alias=True) for detection in result.detections],
                    "riskScores": risk_scores,
                    "riskScoring": risk_scorer.metadata(),
                    "model": result.model.model_dump(by_alias=True),
                    "queue": {
                        "policy": "fifo_no_drop",
                        "droppedFrames": 0,
                        "pendingFrames": frame_queue.qsize(),
                    },
                    "telemetry": result.telemetry.model_dump(by_alias=True),
                },
            )

    processor_task = asyncio.create_task(process_frames_in_order())

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
                risk_scorer = create_driver4_v7_risk_scorer()
                await send_json(
                    {
                        "type": "session_started",
                        "sessionId": session_id,
                        "serverTime": _server_time_ms(),
                        "transport": "websocket",
                        "queuePolicy": "fifo_no_drop",
                        "riskScoring": risk_scorer.metadata(),
                    },
                )
                continue

            if message_type == "session_end":
                try:
                    session_end = SessionEndMessage.model_validate(payload)
                except ValidationError as exc:
                    await send_error("invalid_session_end", exc.errors()[0]["msg"])
                    continue

                await frame_queue.join()
                stop_event.set()
                await send_json(
                    {
                        "type": "session_ended",
                        "sessionId": session_end.session_id,
                        "serverTime": _server_time_ms(),
                        "droppedFrames": 0,
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
                await send_error("frame_too_large", f"Frame exceeds max_frame_bytes={settings.max_frame_bytes}.", frame_meta.frame_id)
                continue

            await frame_queue.put(QueuedFrame(meta=frame_meta, frame_bytes=frame_bytes, server_received_at=server_received_at))
    except WebSocketDisconnect:
        return
    finally:
        stop_event.set()
        processor_task.cancel()
