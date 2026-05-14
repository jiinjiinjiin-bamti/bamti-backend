import json
import logging
import time
from json import JSONDecodeError
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.api.aihub.mobile.schemas import (
    DashboardControlMessage,
    JsonObject,
    MobileFrameMetaMessage,
    MobileSessionCreateRequest,
    MobileSessionResponse,
    PhoneTelemetryMessage,
)
from app.api.aihub.mobile.session_manager import MobileConnection, MobileSession, mobile_session_manager
from app.core.config import settings
from app.inference.manifest import get_runner
from app.inference.score_averaging import RollingScoreAverager


router = APIRouter(prefix="/mobile", tags=["aihub-v6-mobile"])
logger = logging.getLogger(__name__)


def _server_time_ms() -> float:
    return round(time.time() * 1000, 3)


def _session_response(session: MobileSession) -> MobileSessionResponse:
    return MobileSessionResponse(
        session_id=session.session_id,
        camera_url=session.camera_url,
        created_at=session.created_at,
        expires_at=session.expires_at,
        phone_connected=session.phone_connected,
        dashboard_connected=session.dashboard_connected,
        streaming=session.streaming,
        status=session.status,
    )


def _parse_json_message(raw_message: str) -> JsonObject:
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("WebSocket text messages must be JSON objects.")
    return payload


async def _require_session(session_id: str) -> MobileSession:
    session = await mobile_session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mobile session was not found or expired.")
    return session


async def _require_ws_session(websocket: WebSocket, session_id: str) -> MobileSession | None:
    session = await mobile_session_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    return session


async def _send_ws_error(connection: MobileConnection, code: str, message: str, frame_id: str | None = None) -> None:
    payload = {
        "type": "error",
        "code": code,
        "message": message,
    }
    if frame_id is not None:
        payload["frameId"] = frame_id
    await mobile_session_manager.send_json(connection, payload)


@router.post("/sessions", response_model=MobileSessionResponse)
async def create_mobile_session(payload: MobileSessionCreateRequest) -> MobileSessionResponse:
    session = await mobile_session_manager.create_session(payload.camera_url_base)
    return _session_response(session)


@router.get("/sessions/{session_id}", response_model=MobileSessionResponse)
async def get_mobile_session(session_id: str) -> MobileSessionResponse:
    session = await _require_session(session_id)
    return _session_response(session)


@router.delete("/sessions/{session_id}", response_model=MobileSessionResponse)
async def delete_mobile_session(session_id: str) -> MobileSessionResponse:
    session = await mobile_session_manager.delete_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mobile session was not found.")
    return _session_response(session)


@router.websocket("/sessions/{session_id}/phone-frame-stream")
async def phone_frame_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = await _require_ws_session(websocket, session_id)
    if session is None:
        return

    connection = await mobile_session_manager.set_connection(session, "phone_frame", websocket)
    runner = get_runner("aihub-torch")
    score_averager = RollingScoreAverager(window_seconds=1.0)

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = _parse_json_message(raw_message)
            except (JSONDecodeError, ValueError) as exc:
                await _send_ws_error(connection, "invalid_json", str(exc))
                continue

            message_type = payload.get("type")
            if message_type == "phone_telemetry":
                try:
                    telemetry = PhoneTelemetryMessage.model_validate(payload).model_dump(by_alias=True, exclude_none=True)
                except ValidationError as exc:
                    await _send_ws_error(connection, "invalid_phone_telemetry", exc.errors()[0]["msg"])
                    continue
                await mobile_session_manager.update_phone_telemetry(session, telemetry)
                continue

            if message_type != "frame_meta":
                await _send_ws_error(connection, "unsupported_message_type", f"Unsupported message type: {message_type}")
                continue

            try:
                frame_meta = MobileFrameMetaMessage.model_validate(payload)
            except ValidationError as exc:
                await _send_ws_error(connection, "invalid_frame_meta", exc.errors()[0]["msg"])
                continue

            server_received_at = _server_time_ms()
            frame_bytes = await websocket.receive_bytes()
            if not frame_bytes:
                await _send_ws_error(connection, "empty_frame", "Frame binary message must not be empty.", frame_meta.frame_id)
                continue
            if len(frame_bytes) > settings.max_frame_bytes:
                await _send_ws_error(
                    connection,
                    "frame_too_large",
                    f"Frame exceeds max_frame_bytes={settings.max_frame_bytes}.",
                    frame_meta.frame_id,
                )
                continue
            if not session.streaming:
                continue

            try:
                result = await runner.infer(frame_bytes)
            except Exception:
                logger.exception("AIHub mobile v6 inference failed for session %s frame %s", session_id, frame_meta.frame_id)
                await mobile_session_manager.set_streaming(session, False)
                await _send_ws_error(connection, "inference_failed", "Mobile inference failed. Check backend model/runtime logs.", frame_meta.frame_id)
                await mobile_session_manager.send_dashboard_event(
                    session,
                    {
                        "type": "error",
                        "code": "inference_failed",
                        "message": "Mobile inference failed. Check backend model/runtime logs.",
                        "frameId": frame_meta.frame_id,
                    },
                )
                continue
            averaged_detections = score_averager.average(result.detections)
            server_responded_at = _server_time_ms()

            await mobile_session_manager.send_dashboard_event(
                session,
                {
                    "type": "inference_result",
                    "sessionId": session_id,
                    "frameId": frame_meta.frame_id,
                    "clientSentAt": frame_meta.client_sent_at,
                    "serverReceivedAt": server_received_at,
                    "serverRespondedAt": server_responded_at,
                    "frame": {
                        "width": frame_meta.width,
                        "height": frame_meta.height,
                        "encodingMs": frame_meta.encoding_ms,
                        "frameSizeBytes": frame_meta.frame_size_bytes or len(frame_bytes),
                    },
                    "detections": [detection.model_dump(by_alias=True) for detection in averaged_detections],
                    "model": result.model.model_dump(by_alias=True),
                    "scoreSmoothing": {
                        "enabled": True,
                        "windowMs": 1000,
                    },
                    "telemetry": result.telemetry.model_dump(by_alias=True),
                    "phoneTelemetry": session.phone_telemetry,
                },
            )
    except WebSocketDisconnect:
        return
    finally:
        await mobile_session_manager.clear_connection(session, "phone_frame", connection)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()


@router.websocket("/sessions/{session_id}/dashboard-channel")
async def dashboard_channel(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = await _require_ws_session(websocket, session_id)
    if session is None:
        return

    connection = await mobile_session_manager.set_connection(session, "dashboard_channel", websocket)

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = _parse_json_message(raw_message)
                control = DashboardControlMessage.model_validate(payload)
            except (JSONDecodeError, ValueError, ValidationError) as exc:
                message = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                await _send_ws_error(connection, "invalid_dashboard_control", message)
                continue

            await mobile_session_manager.set_streaming(session, control.type == "start_stream")
    except WebSocketDisconnect:
        return
    finally:
        await mobile_session_manager.clear_connection(session, "dashboard_channel", connection)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()


@router.websocket("/sessions/{session_id}/signaling")
async def mobile_signaling(
    websocket: WebSocket,
    session_id: str,
    role: Literal["phone", "dashboard"] = Query(...),
) -> None:
    await websocket.accept()
    session = await _require_ws_session(websocket, session_id)
    if session is None:
        return

    channel = "phone_signaling" if role == "phone" else "dashboard_signaling"
    connection = await mobile_session_manager.set_connection(session, channel, websocket)

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = _parse_json_message(raw_message)
            except (JSONDecodeError, ValueError) as exc:
                await _send_ws_error(connection, "invalid_json", str(exc))
                continue
            await mobile_session_manager.relay_signaling(session, channel, payload)
    except WebSocketDisconnect:
        return
    finally:
        await mobile_session_manager.clear_connection(session, channel, connection)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
