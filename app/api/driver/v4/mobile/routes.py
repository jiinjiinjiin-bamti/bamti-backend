from typing import Literal

from fastapi import APIRouter, Query, WebSocket

from app.api.aihub.mobile.routes import (
    MobileSessionCreateRequest,
    MobileSessionResponse,
    create_mobile_session,
    dashboard_channel,
    delete_mobile_session,
    get_mobile_session,
    mobile_signaling,
    run_phone_frame_stream,
)


router = APIRouter(prefix="/mobile", tags=["driver4-v4-mobile"])
router.add_api_route("/sessions", create_mobile_session, methods=["POST"], response_model=MobileSessionResponse)
router.add_api_route("/sessions/{session_id}", get_mobile_session, methods=["GET"], response_model=MobileSessionResponse)
router.add_api_route("/sessions/{session_id}", delete_mobile_session, methods=["DELETE"], response_model=MobileSessionResponse)
router.add_api_websocket_route("/sessions/{session_id}/dashboard-channel", dashboard_channel)


@router.websocket("/sessions/{session_id}/phone-frame-stream")
async def phone_frame_stream(websocket: WebSocket, session_id: str) -> None:
    await run_phone_frame_stream(websocket, session_id, runner_name="driver4-torch", model_profile="Driver4")


@router.websocket("/sessions/{session_id}/signaling")
async def signaling(
    websocket: WebSocket,
    session_id: str,
    role: Literal["phone", "dashboard"] = Query(...),
) -> None:
    await mobile_signaling(websocket, session_id, role)
