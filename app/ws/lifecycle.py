import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.storage.database import async_session_factory
from app.storage.repositories import DrivingSessionRepository, SessionSummaryRepository

logger = logging.getLogger(__name__)


class WebSocketSessionLifecycle:
    """Persists WebSocket session lifecycle changes."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
    ) -> None:
        self._session_factory = session_factory

    async def start_session(
        self,
        session_id: str,
        driver_id: str | None,
        started_at: datetime | None,
    ) -> bool:
        async with self._session_factory() as db:
            try:
                repository = DrivingSessionRepository(db)
                await repository.create(
                    session_id=session_id,
                    driver_id=driver_id,
                    started_at=started_at,
                )
                await db.commit()
                logger.info(
                    "websocket_session_started",
                    extra={"session_id": session_id, "driver_id": driver_id},
                )
            except SQLAlchemyError:
                await db.rollback()
                logger.exception(
                    "websocket_session_start_persistence_failed",
                    extra={"session_id": session_id, "driver_id": driver_id},
                )
                return False
            return True

    async def end_session(
        self,
        session_id: str,
        ended_at: datetime | None,
        close_reason: str,
    ) -> bool:
        async with self._session_factory() as db:
            try:
                session_repository = DrivingSessionRepository(db)
                summary_repository = SessionSummaryRepository(db)

                session = await session_repository.end(
                    session_id=session_id,
                    ended_at=ended_at,
                )
                if session is None:
                    logger.warning(
                        "websocket_session_end_missing_session",
                        extra={
                            "session_id": session_id,
                            "close_reason": close_reason,
                        },
                    )
                    await db.rollback()
                    return False

                await summary_repository.upsert(
                    session_id=session_id,
                    total_events=0,
                    distraction_seconds=0.0,
                )
                await db.commit()
                logger.info(
                    "websocket_session_ended",
                    extra={"session_id": session_id, "close_reason": close_reason},
                )
            except SQLAlchemyError:
                await db.rollback()
                logger.exception(
                    "websocket_session_end_persistence_failed",
                    extra={"session_id": session_id, "close_reason": close_reason},
                )
                return False
            return True


session_lifecycle = WebSocketSessionLifecycle()
