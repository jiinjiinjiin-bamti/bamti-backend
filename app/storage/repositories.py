from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.models import (
    DistractionEvent,
    DrivingSession,
    SessionStatus,
    SessionSummary,
)


class DrivingSessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        session_id: str,
        driver_id: str | None,
        started_at: datetime | None = None,
    ) -> DrivingSession:
        session = DrivingSession(
            id=session_id,
            driver_id=driver_id,
            started_at=started_at or datetime.now(UTC),
            status=SessionStatus.ACTIVE,
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def get(self, session_id: str) -> DrivingSession | None:
        result = await self.db.execute(
            select(DrivingSession).where(DrivingSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def end(
        self,
        session_id: str,
        ended_at: datetime | None = None,
    ) -> DrivingSession | None:
        session = await self.get(session_id)
        if session is None:
            return None
        session.status = SessionStatus.ENDED
        session.ended_at = ended_at or datetime.now(UTC)
        await self.db.flush()
        return session


class DistractionEventRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        session_id: str,
        label: str,
        severity: str,
        confidence: float,
        message: str,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> DistractionEvent:
        event = DistractionEvent(
            session_id=session_id,
            label=label,
            severity=severity,
            confidence=confidence,
            message=message,
            started_at=started_at or datetime.now(UTC),
            ended_at=ended_at,
        )
        self.db.add(event)
        await self.db.flush()
        return event

    async def list_by_session(self, session_id: str) -> list[DistractionEvent]:
        result = await self.db.execute(
            select(DistractionEvent).where(DistractionEvent.session_id == session_id)
        )
        return list(result.scalars().all())


class SessionSummaryRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert(
        self,
        session_id: str,
        total_events: int,
        distraction_seconds: float,
    ) -> SessionSummary:
        result = await self.db.execute(
            select(SessionSummary).where(SessionSummary.session_id == session_id)
        )
        summary = result.scalar_one_or_none()
        if summary is None:
            summary = SessionSummary(
                session_id=session_id,
                total_events=total_events,
                distraction_seconds=distraction_seconds,
            )
            self.db.add(summary)
        else:
            summary.total_events = total_events
            summary.distraction_seconds = distraction_seconds
        await self.db.flush()
        return summary
