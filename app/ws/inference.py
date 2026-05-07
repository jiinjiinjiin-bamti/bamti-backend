import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketState

from app.alerts.engine import AlertEngine
from app.core.config import settings
from app.inference.manifest import get_runner
from app.inference.schemas import FrameMeta, QueuedFrame
from app.ws.lifecycle import session_lifecycle
from app.ws.manager import WebSocketSessionManager

router = APIRouter()
manager = WebSocketSessionManager(queue_size=settings.frame_queue_size)
logger = logging.getLogger(__name__)


class SessionStartMessage(BaseModel):
    type: Literal["session_start"]
    session_id: str
    driver_id: str | None = None
    started_at: datetime | None = None


class SessionEndMessage(BaseModel):
    type: Literal["session_end"]
    session_id: str
    ended_at: datetime | None = None


class PingMessage(BaseModel):
    type: Literal["ping"]


async def _send_results(
    websocket: WebSocket,
    queue: asyncio.Queue[QueuedFrame],
    runner_name: str,
) -> None:
    runner = get_runner(runner_name)
    alert_engine = AlertEngine()

    while True:
        frame = await queue.get()
        try:
            result = await runner.infer(frame.jpeg_bytes)
            alerts = alert_engine.evaluate(result)
            await websocket.send_json(
                {
                    "type": "inference_result",
                    "session_id": frame.session_id,
                    "frame_id": frame.meta.frame_id,
                    "captured_at": frame.meta.captured_at.isoformat(),
                    "result": result.model_dump(),
                    "alerts": [alert.model_dump() for alert in alerts],
                }
            )
        finally:
            queue.task_done()


def _message_type(payload: dict[str, Any]) -> str:
    message_type = payload.get("type")
    if not isinstance(message_type, str):
        raise ValueError("Message must include a string 'type'.")
    return message_type


async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    if websocket.application_state == WebSocketState.CONNECTED:
        await websocket.send_json(
            {
                "type": "error",
                "code": code,
                "message": message,
            }
        )


def _validate_frame_meta(meta: FrameMeta) -> str | None:
    if meta.content_type != "image/jpeg":
        return "Only image/jpeg frames are supported."
    return None


def _validate_frame_bytes(frame: bytes) -> tuple[str, str] | None:
    if not frame:
        return ("empty_frame", "Binary frame must not be empty.")
    if len(frame) > settings.max_frame_bytes:
        return (
            "frame_too_large",
            f"Binary frame exceeds max_frame_bytes={settings.max_frame_bytes}.",
        )
    return None


async def _drain_queue(websocket: WebSocket, queue: asyncio.Queue[QueuedFrame]) -> bool:
    try:
        await asyncio.wait_for(
            queue.join(),
            timeout=settings.websocket_drain_timeout_seconds,
        )
        return True
    except TimeoutError:
        await _send_error(
            websocket,
            "queue_drain_timeout",
            "Timed out while waiting for queued frames to finish.",
        )
        return False


async def _start_persisted_session(start: SessionStartMessage) -> bool:
    try:
        return await session_lifecycle.start_session(
            session_id=start.session_id,
            driver_id=start.driver_id,
            started_at=start.started_at,
        )
    except Exception:
        logger.exception(
            "websocket_session_start_unhandled_error",
            extra={"session_id": start.session_id},
        )
        return False


async def _end_persisted_session(
    session_id: str,
    ended_at: datetime | None,
    close_reason: str,
) -> None:
    try:
        persisted = await session_lifecycle.end_session(
            session_id=session_id,
            ended_at=ended_at,
            close_reason=close_reason,
        )
        if not persisted:
            logger.warning(
                "websocket_session_end_not_persisted",
                extra={"session_id": session_id, "close_reason": close_reason},
            )
    except Exception:
        logger.exception(
            "websocket_session_end_unhandled_error",
            extra={"session_id": session_id, "close_reason": close_reason},
        )


@router.websocket("/inference")
async def inference_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    session_id: str | None = None
    queue: asyncio.Queue[QueuedFrame] | None = None
    worker: asyncio.Task[None] | None = None
    pending_meta: FrameMeta | None = None
    persisted_session_open = False
    cleanup_reason = "client_disconnect"

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.websocket_idle_timeout_seconds,
                )
            except TimeoutError:
                await _send_error(
                    websocket,
                    "idle_timeout",
                    "No frame or control message received before idle timeout.",
                )
                await websocket.close(code=1000)
                cleanup_reason = "idle_timeout"
                break

            if message.get("type") == "websocket.disconnect":
                cleanup_reason = "client_disconnect"
                break

            if "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                message_type = _message_type(payload)

                if message_type == "ping":
                    PingMessage.model_validate(payload)
                    await websocket.send_json({"type": "pong"})
                    continue

                if message_type == "session_start":
                    start = SessionStartMessage.model_validate(payload)
                    persisted_session_open = await _start_persisted_session(start)
                    if not persisted_session_open:
                        await _send_error(
                            websocket,
                            "session_persistence_failed",
                            "Failed to persist session_start.",
                        )
                        await websocket.close(code=1011)
                        cleanup_reason = "session_start_persistence_failed"
                        break
                    session_id = start.session_id
                    queue = manager.create_queue(session_id)
                    worker = asyncio.create_task(
                        _send_results(websocket, queue, settings.inference_runner)
                    )
                    await websocket.send_json(
                        {"type": "session_started", "session_id": session_id}
                    )
                    continue

                if message_type == "frame_meta":
                    if session_id is None:
                        await _send_error(
                            websocket,
                            "session_not_started",
                            "Send session_start before frame_meta.",
                        )
                        continue
                    if pending_meta is not None:
                        pending_meta = None
                        await _send_error(
                            websocket,
                            "frame_already_pending",
                            "Send the binary frame for the previous frame_meta before sending another frame_meta.",
                        )
                        continue
                    meta = FrameMeta.model_validate(payload)
                    validation_error = _validate_frame_meta(meta)
                    if validation_error is not None:
                        pending_meta = None
                        await _send_error(
                            websocket,
                            "unsupported_content_type",
                            validation_error,
                        )
                        continue
                    pending_meta = meta
                    continue

                if message_type == "session_end":
                    end = SessionEndMessage.model_validate(payload)
                    end_session_id = session_id or end.session_id
                    cleanup_reason = "session_end"
                    if queue is not None:
                        drained = await _drain_queue(websocket, queue)
                        if not drained:
                            cleanup_reason = "queue_drain_timeout"
                    if persisted_session_open:
                        await _end_persisted_session(
                            session_id=end_session_id,
                            ended_at=end.ended_at,
                            close_reason=cleanup_reason,
                        )
                        persisted_session_open = False
                    await websocket.send_json(
                        {"type": "session_ended", "session_id": end_session_id}
                    )
                    break

                await _send_error(
                    websocket,
                    "unsupported_message",
                    f"Unsupported message type: {message_type}",
                )
                continue

            if "bytes" in message and message["bytes"] is not None:
                if session_id is None or queue is None or pending_meta is None:
                    await _send_error(
                        websocket,
                        "frame_meta_required",
                        "Send frame_meta before a binary frame.",
                    )
                    continue
                frame_bytes = message["bytes"]
                validation_error = _validate_frame_bytes(frame_bytes)
                if validation_error is not None:
                    code, error_message = validation_error
                    pending_meta = None
                    await _send_error(websocket, code, error_message)
                    continue
                manager.put_latest(
                    queue,
                    QueuedFrame(
                        session_id=session_id,
                        meta=pending_meta,
                        jpeg_bytes=frame_bytes,
                    ),
                )
                pending_meta = None

    except WebSocketDisconnect:
        cleanup_reason = "client_disconnect"
        pass
    except (ValidationError, ValueError) as exc:
        cleanup_reason = "invalid_message"
        await _send_error(websocket, "invalid_message", str(exc))
    finally:
        if persisted_session_open and session_id is not None:
            await _end_persisted_session(
                session_id=session_id,
                ended_at=None,
                close_reason=cleanup_reason,
            )
        if worker is not None:
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if session_id is not None:
            manager.remove_queue(session_id)
