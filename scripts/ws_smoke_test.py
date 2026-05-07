import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.storage.models import (
    DistractionEvent,
    DrivingSession,
    SessionStatus,
    SessionSummary,
)

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect  # type: ignore[no-redef]


class SmokeTestError(RuntimeError):
    pass


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


async def _receive_json(websocket: Any, timeout_seconds: float) -> dict[str, Any]:
    message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
    if not isinstance(message, str):
        raise SmokeTestError(f"Expected JSON text message, got {type(message).__name__}.")
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise SmokeTestError("Expected JSON object message.")
    return payload


async def _exercise_websocket(
    ws_url: str,
    session_id: str,
    driver_id: str,
    timeout_seconds: float,
) -> None:
    started_at = datetime.now(UTC).replace(microsecond=0)
    captured_at = started_at + timedelta(seconds=1)
    ended_at = started_at + timedelta(seconds=2)

    async with connect(ws_url, open_timeout=timeout_seconds) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "session_start",
                    "session_id": session_id,
                    "driver_id": driver_id,
                    "started_at": _isoformat(started_at),
                }
            )
        )
        started = await _receive_json(websocket, timeout_seconds)
        if started != {"type": "session_started", "session_id": session_id}:
            raise SmokeTestError(f"Unexpected session_start response: {started}")

        await websocket.send(
            json.dumps(
                {
                    "type": "frame_meta",
                    "frame_id": f"{session_id}-frame-1",
                    "captured_at": _isoformat(captured_at),
                    "content_type": "image/jpeg",
                }
            )
        )
        await websocket.send(b"\xff\xd8\xff\xe0mock-jpeg\xff\xd9")

        inference_result = await _receive_json(websocket, timeout_seconds)
        if inference_result.get("type") != "inference_result":
            raise SmokeTestError(f"Unexpected inference response: {inference_result}")
        if inference_result.get("session_id") != session_id:
            raise SmokeTestError(f"Unexpected inference session_id: {inference_result}")

        await websocket.send(
            json.dumps(
                {
                    "type": "session_end",
                    "session_id": session_id,
                    "ended_at": _isoformat(ended_at),
                }
            )
        )
        ended = await _receive_json(websocket, timeout_seconds)
        if ended != {"type": "session_ended", "session_id": session_id}:
            raise SmokeTestError(f"Unexpected session_end response: {ended}")


async def _load_persisted_rows(
    database_url: str,
    session_id: str,
) -> tuple[DrivingSession | None, SessionSummary | None, int]:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with session_factory() as db:
            session_result = await db.execute(
                select(DrivingSession).where(DrivingSession.id == session_id)
            )
            summary_result = await db.execute(
                select(SessionSummary).where(SessionSummary.session_id == session_id)
            )
            event_count_result = await db.execute(
                select(func.count())
                .select_from(DistractionEvent)
                .where(DistractionEvent.session_id == session_id)
            )
            return (
                session_result.scalar_one_or_none(),
                summary_result.scalar_one_or_none(),
                int(event_count_result.scalar_one()),
            )
    finally:
        await engine.dispose()


async def _assert_persisted(
    database_url: str,
    session_id: str,
    timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    session: DrivingSession | None = None
    summary: SessionSummary | None = None
    event_count = 0

    while True:
        session, summary, event_count = await _load_persisted_rows(
            database_url=database_url,
            session_id=session_id,
        )
        if session is not None and summary is not None:
            break
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.5)

    errors: list[str] = []
    if session is None:
        errors.append("driving_session row was not created.")
    elif session.status != SessionStatus.ENDED:
        errors.append(f"driving_session status is {session.status!r}, expected 'ended'.")

    if summary is None:
        errors.append("session_summary row was not created.")
    else:
        if summary.total_events != 0:
            errors.append(f"session_summary.total_events={summary.total_events}, expected 0.")
        if summary.distraction_seconds != 0.0:
            errors.append(
                "session_summary.distraction_seconds="
                f"{summary.distraction_seconds}, expected 0.0."
            )

    if event_count != 0:
        errors.append(f"distraction_event count is {event_count}, expected 0 for MockRunner.")

    if errors:
        raise SmokeTestError(" ".join(errors))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise /ws/inference and verify MySQL session persistence."
    )
    parser.add_argument(
        "--ws-url",
        default=os.getenv("WS_SMOKE_WS_URL", "ws://localhost:8000/ws/inference"),
        help="WebSocket URL to test.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("WS_SMOKE_DATABASE_URL", settings.database_url),
        help="SQLAlchemy async database URL used for verification.",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("WS_SMOKE_SESSION_ID", f"smoke-{uuid4().hex}"),
        help="Session id to create.",
    )
    parser.add_argument(
        "--driver-id",
        default=os.getenv("WS_SMOKE_DRIVER_ID", "smoke-driver"),
        help="Driver id to send in session_start.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("WS_SMOKE_TIMEOUT_SECONDS", "10")),
        help="Timeout for WebSocket messages and DB verification.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    await _exercise_websocket(
        ws_url=args.ws_url,
        session_id=args.session_id,
        driver_id=args.driver_id,
        timeout_seconds=args.timeout_seconds,
    )
    await _assert_persisted(
        database_url=args.database_url,
        session_id=args.session_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        "Smoke test passed: WebSocket session persisted to MySQL "
        f"with driving_session and session_summary rows for {args.session_id}."
    )


if __name__ == "__main__":
    asyncio.run(_main())
