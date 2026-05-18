import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketState


session_ttl = timedelta(minutes=15)


@dataclass
class MobileConnection:
    websocket: WebSocket
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class MobileSession:
    session_id: str
    camera_url: str
    created_at: datetime
    expires_at: datetime
    phone_frame: MobileConnection | None = None
    dashboard_channel: MobileConnection | None = None
    phone_signaling: MobileConnection | None = None
    dashboard_signaling: MobileConnection | None = None
    phone_telemetry: dict | None = None
    streaming: bool = False

    @property
    def phone_connected(self) -> bool:
        return self.phone_frame is not None

    @property
    def dashboard_connected(self) -> bool:
        return self.dashboard_channel is not None

    @property
    def status(self) -> str:
        if datetime.now(UTC) >= self.expires_at:
            return "expired"
        if self.streaming:
            return "streaming"
        if self.phone_connected and self.dashboard_connected:
            return "connected"
        return "waiting"


class MobileSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, MobileSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, camera_url_base: str | None) -> MobileSession:
        now = datetime.now(UTC)
        session_id = f"mobile-{uuid4().hex[:12]}"
        base = (camera_url_base or "").rstrip("/")
        if "{sessionId}" in base:
            camera_url = base.replace("{sessionId}", session_id)
        else:
            camera_url = f"{base}?sessionId={session_id}" if base else f"/camera?sessionId={session_id}"
        session = MobileSession(
            session_id=session_id,
            camera_url=camera_url,
            created_at=now,
            expires_at=now + session_ttl,
        )

        async with self._lock:
            self._sessions[session_id] = session

        return session

    async def get_session(self, session_id: str) -> MobileSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if datetime.now(UTC) >= session.expires_at:
                await self._close_session_connections(session)
                self._sessions.pop(session_id, None)
                return None
            return session

    async def delete_session(self, session_id: str) -> MobileSession | None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return None
            await self._close_session_connections(session)
            return session

    async def set_connection(self, session: MobileSession, channel: str, websocket: WebSocket) -> MobileConnection:
        connection = MobileConnection(websocket=websocket)
        async with self._lock:
            old_connection = getattr(session, channel)
            setattr(session, channel, connection)

        if old_connection is not None and old_connection.websocket is not websocket:
            await self.send_json(old_connection, {"type": "duplicate_connection", "channel": channel})
            await self.close_connection(old_connection)

        await self.broadcast_session_state(session)
        if channel.endswith("signaling"):
            await self.notify_signaling_peer_connected(session, channel)
        return connection

    async def clear_connection(self, session: MobileSession, channel: str, connection: MobileConnection) -> None:
        async with self._lock:
            if getattr(session, channel) is connection:
                setattr(session, channel, None)
                if channel == "phone_frame" or channel == "dashboard_channel":
                    session.streaming = False

        await self.broadcast_session_state(session)
        if channel == "phone_frame":
            await self.send_dashboard_event(session, {"type": "phone_disconnected", "sessionId": session.session_id})
        if channel == "dashboard_channel":
            await self.send_phone_control(session, {"type": "stream_state", "streaming": False})

    async def set_streaming(self, session: MobileSession, streaming: bool) -> None:
        async with self._lock:
            session.streaming = streaming and session.phone_connected and session.dashboard_connected

        payload = {
            "type": "stream_state",
            "sessionId": session.session_id,
            "streaming": session.streaming,
        }
        await self.send_dashboard_event(session, payload)
        await self.send_phone_control(session, payload)

    async def update_phone_telemetry(self, session: MobileSession, telemetry: dict) -> None:
        async with self._lock:
            session.phone_telemetry = telemetry
        await self.send_dashboard_event(
            session,
            {
                "type": "phone_telemetry",
                "sessionId": session.session_id,
                "telemetry": telemetry,
            },
        )

    async def send_dashboard_event(self, session: MobileSession, payload: dict) -> None:
        if session.dashboard_channel is not None:
            await self.send_json(session.dashboard_channel, payload)

    async def send_phone_control(self, session: MobileSession, payload: dict) -> None:
        if session.phone_frame is not None:
            await self.send_json(session.phone_frame, payload)

    async def relay_signaling(self, session: MobileSession, from_channel: str, payload: dict) -> None:
        target = session.dashboard_signaling if from_channel == "phone_signaling" else session.phone_signaling
        if target is None:
            return

        await self.send_json(
            target,
            {
                **payload,
                "sessionId": session.session_id,
                "from": "phone" if from_channel == "phone_signaling" else "dashboard",
            },
        )

    async def notify_signaling_peer_connected(self, session: MobileSession, connected_channel: str) -> None:
        peer_payload = {
            "type": "peer_connected",
            "sessionId": session.session_id,
            "peer": "phone" if connected_channel == "phone_signaling" else "dashboard",
        }
        if connected_channel == "phone_signaling" and session.dashboard_signaling is not None:
            await self.send_json(session.dashboard_signaling, peer_payload)
        if connected_channel == "dashboard_signaling" and session.phone_signaling is not None:
            await self.send_json(session.phone_signaling, peer_payload)
        if session.phone_signaling is not None and session.dashboard_signaling is not None:
            await self.send_json(
                getattr(session, connected_channel),
                {
                    "type": "peer_connected",
                    "sessionId": session.session_id,
                    "peer": "dashboard" if connected_channel == "phone_signaling" else "phone",
                },
            )

    async def broadcast_session_state(self, session: MobileSession) -> None:
        payload = self.session_state_payload(session)
        if session.dashboard_channel is not None:
            await self.send_json(session.dashboard_channel, payload)
        if session.phone_frame is not None:
            await self.send_json(session.phone_frame, payload)

    def session_state_payload(self, session: MobileSession) -> dict:
        return {
            "type": "session_state",
            "sessionId": session.session_id,
            "phoneConnected": session.phone_connected,
            "dashboardConnected": session.dashboard_connected,
            "streaming": session.streaming,
            "status": session.status,
            "createdAt": session.created_at.isoformat(),
            "expiresAt": session.expires_at.isoformat(),
            "phoneTelemetry": session.phone_telemetry,
        }

    async def send_json(self, connection: MobileConnection, payload: dict) -> None:
        if connection.websocket.application_state == WebSocketState.DISCONNECTED:
            return
        async with connection.lock:
            await connection.websocket.send_json(payload)

    async def close_connection(self, connection: MobileConnection) -> None:
        if connection.websocket.application_state != WebSocketState.DISCONNECTED:
            await connection.websocket.close()

    async def _close_session_connections(self, session: MobileSession) -> None:
        connections = [
            session.phone_frame,
            session.dashboard_channel,
            session.phone_signaling,
            session.dashboard_signaling,
        ]
        for connection in connections:
            if connection is not None:
                await self.close_connection(connection)


mobile_session_manager = MobileSessionManager()
