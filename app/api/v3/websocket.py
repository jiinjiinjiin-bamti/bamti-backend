import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Awaitable, Callable

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


def _websocket_diagnostics_enabled() -> bool:
    return os.getenv("WEBSOCKET_DIAGNOSTICS", "").lower() in {"1", "true", "yes", "on"}


async def run_latest_pending_inference_stream(
    websocket: WebSocket,
    runner_name: str | None = None,
    runtime_metadata: dict | None = None,
    include_debug_raw_detections: bool = False,
    session_metadata_factory: Callable[[], dict[str, Any]] | None = None,
    inference_factory: Callable[[Any, QueuedFrame], Awaitable[Any]] | None = None,
    result_payload_factory: Callable[[Any, QueuedFrame], dict[str, Any]] | None = None,
    frame_meta_model: type[Any] = FrameMetaMessage,
    include_model_in_results: bool = True,
    include_queue_policy_in_results: bool = True,
    include_queue_pending_in_results: bool = True,
) -> None:
    await websocket.accept()

    session_id: str | None = None
    pending_frame: QueuedFrame | None = None
    dropped_frames = 0
    diagnostics_enabled = _websocket_diagnostics_enabled()
    diagnostics = {
        "receive_text": 0,
        "frame_meta": 0,
        "frame_bytes": 0,
        "queued": 0,
        "dropped": 0,
        "infer_start": 0,
        "infer_done": 0,
        "send_start": 0,
        "send_done": 0,
        "send_error": 0,
    }
    diagnostic_last_values: dict[str, float | str | None] = {
        "frame_id": None,
        "infer_ms": None,
        "send_ms": None,
    }
    frame_available = asyncio.Event()
    queue_lock = asyncio.Lock()
    send_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    websocket_close_sent = False

    async def send_json(payload: dict) -> None:
        async with send_lock:
            diagnostics["send_start"] += 1
            send_started_at = time.perf_counter()
            try:
                await websocket.send_json(payload)
            except Exception:
                diagnostics["send_error"] += 1
                raise
            finally:
                diagnostic_last_values["send_ms"] = round((time.perf_counter() - send_started_at) * 1000, 3)
            diagnostics["send_done"] += 1

    def log_diagnostics(reason: str) -> None:
        if not diagnostics_enabled:
            return

        logger.info(
            "ws_diag reason=%s session=%s receive_text=%s frame_meta=%s frame_bytes=%s queued=%s "
            "dropped=%s infer_start=%s infer_done=%s send_start=%s send_done=%s send_error=%s "
            "pending=%s last_frame=%s last_infer_ms=%s last_send_ms=%s",
            reason,
            session_id,
            diagnostics["receive_text"],
            diagnostics["frame_meta"],
            diagnostics["frame_bytes"],
            diagnostics["queued"],
            diagnostics["dropped"],
            diagnostics["infer_start"],
            diagnostics["infer_done"],
            diagnostics["send_start"],
            diagnostics["send_done"],
            diagnostics["send_error"],
            1 if pending_frame is not None else 0,
            diagnostic_last_values["frame_id"],
            diagnostic_last_values["infer_ms"],
            diagnostic_last_values["send_ms"],
        )

    async def log_diagnostics_periodically() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(5)
            log_diagnostics("periodic")

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

            diagnostics["infer_start"] += 1
            diagnostic_last_values["frame_id"] = frame.meta.frame_id
            inference_started_at = time.perf_counter()
            try:
                result = await (
                    inference_factory(runner, frame)
                    if inference_factory is not None
                    else runner.infer(frame.frame_bytes)
                )
            except Exception:
                logger.exception("Latest-pending inference failed for session %s frame %s", session_id, frame.meta.frame_id)
                stop_event.set()
                await send_error("inference_failed", "WebSocket inference failed. Check backend model/runtime logs.", frame.meta.frame_id)
                await close_websocket(code=status.WS_1011_INTERNAL_ERROR)
                return
            diagnostics["infer_done"] += 1
            diagnostic_last_values["infer_ms"] = round((time.perf_counter() - inference_started_at) * 1000, 3)
            server_responded_at = _server_time_ms()

            queue_payload = {
                "droppedFrames": dropped_frames,
            }
            if include_queue_policy_in_results:
                queue_payload["policy"] = "latest_pending_only"
            if include_queue_pending_in_results:
                queue_payload["pendingFrames"] = 1 if pending_frame is not None else 0

            response_payload = {
                "type": "inference_result",
                "sessionId": session_id,
                "frameId": frame.meta.frame_id,
                "clientSentAt": frame.meta.client_sent_at,
                "serverReceivedAt": frame.server_received_at,
                "serverRespondedAt": server_responded_at,
                "detections": [detection.model_dump(by_alias=True) for detection in result.detections],
                "queue": queue_payload,
                "telemetry": result.telemetry.model_dump(by_alias=True),
            }
            if include_model_in_results:
                response_payload["model"] = result.model.model_dump(by_alias=True)
            if result_payload_factory is not None:
                response_payload.update(result_payload_factory(result, frame))
            if include_debug_raw_detections and result.debug_raw_detections:
                response_payload["debugRawDetections"] = [
                    detection.model_dump(by_alias=True)
                    for detection in result.debug_raw_detections
                ]

            await send_json(response_payload)

    processor_task = asyncio.create_task(process_latest_frames())
    diagnostics_task = asyncio.create_task(log_diagnostics_periodically()) if diagnostics_enabled else None

    try:
        while True:
            raw_message = await websocket.receive_text()
            diagnostics["receive_text"] += 1
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
                session_metadata = session_metadata_factory() if session_metadata_factory is not None else {}
                await send_json(
                    {
                        "type": "session_started",
                        "sessionId": session_id,
                        "serverTime": _server_time_ms(),
                        "transport": "websocket",
                        "queuePolicy": "latest_pending_only",
                        **(runtime_metadata or {}),
                        **session_metadata,
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
                frame_meta = frame_meta_model.model_validate(payload)
            except ValidationError as exc:
                await send_error("invalid_frame_meta", exc.errors()[0]["msg"])
                continue
            diagnostics["frame_meta"] += 1
            diagnostic_last_values["frame_id"] = frame_meta.frame_id

            if frame_meta.session_id != session_id:
                await send_error("session_mismatch", "frame_meta sessionId does not match active session.", frame_meta.frame_id)
                continue

            server_received_at = _server_time_ms()
            frame_bytes = await websocket.receive_bytes()
            diagnostics["frame_bytes"] += 1
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
                    diagnostics["dropped"] += 1
                pending_frame = QueuedFrame(
                    meta=frame_meta,
                    frame_bytes=frame_bytes,
                    server_received_at=server_received_at,
                )
                diagnostics["queued"] += 1
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
        log_diagnostics("closing")
        stop_event.set()
        frame_available.set()
        if diagnostics_task is not None:
            diagnostics_task.cancel()
            try:
                await diagnostics_task
            except asyncio.CancelledError:
                pass
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
