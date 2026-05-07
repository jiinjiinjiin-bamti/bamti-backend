"""Shared pytest fixtures."""

from dataclasses import dataclass
from datetime import datetime

import pytest


@dataclass
class RecordedSessionStart:
    session_id: str
    driver_id: str | None
    started_at: datetime | None


@dataclass
class RecordedSessionEnd:
    session_id: str
    ended_at: datetime | None
    close_reason: str


@dataclass
class RecordedSessionSummary:
    session_id: str
    total_events: int
    distraction_seconds: float


class RecordingSessionLifecycle:
    def __init__(self) -> None:
        self.starts: list[RecordedSessionStart] = []
        self.ends: list[RecordedSessionEnd] = []
        self.summaries: list[RecordedSessionSummary] = []
        self.raw_frames: list[bytes] = []
        self.per_frame_results: list[dict[str, object]] = []

    async def start_session(
        self,
        session_id: str,
        driver_id: str | None,
        started_at: datetime | None,
    ) -> bool:
        self.starts.append(
            RecordedSessionStart(
                session_id=session_id,
                driver_id=driver_id,
                started_at=started_at,
            )
        )
        return True

    async def end_session(
        self,
        session_id: str,
        ended_at: datetime | None,
        close_reason: str,
    ) -> bool:
        self.ends.append(
            RecordedSessionEnd(
                session_id=session_id,
                ended_at=ended_at,
                close_reason=close_reason,
            )
        )
        self.summaries.append(
            RecordedSessionSummary(
                session_id=session_id,
                total_events=0,
                distraction_seconds=0.0,
            )
        )
        return True


@pytest.fixture(autouse=True)
def recording_session_lifecycle(monkeypatch: pytest.MonkeyPatch) -> RecordingSessionLifecycle:
    lifecycle = RecordingSessionLifecycle()
    monkeypatch.setattr("app.ws.inference.session_lifecycle", lifecycle)
    return lifecycle
