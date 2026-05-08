import json
import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Body

from app.api.v1.schemas import TelemetryPayload, TelemetryRunListItem, TelemetryRunListResponse, TelemetryRunSavedResponse
from app.core.config import settings


router = APIRouter(prefix="/telemetry", tags=["v1-telemetry"])
kst = timezone(timedelta(hours=9))


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return slug[:80] or "performance-run"


def _runs_dir():
    settings.telemetry_runs_dir.mkdir(parents=True, exist_ok=True)
    return settings.telemetry_runs_dir


@router.post("/runs", response_model=TelemetryRunSavedResponse)
async def save_telemetry_run(payload: TelemetryPayload = Body(...)) -> TelemetryRunSavedResponse:
    now = datetime.now(kst)
    created_at = now.isoformat()
    label = _safe_slug(str(payload.get("label") or "performance-run"))
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%SKST")
    run_id = f"{timestamp}_{label}"
    file_path = _runs_dir() / f"{run_id}.json"

    document = {
        "runId": run_id,
        "createdAt": created_at,
        **payload,
    }
    file_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    return TelemetryRunSavedResponse(
        run_id=run_id,
        file_name=file_path.name,
        path=str(file_path),
        saved=True,
    )


@router.get("/runs", response_model=TelemetryRunListResponse)
async def list_telemetry_runs() -> TelemetryRunListResponse:
    runs = [
        TelemetryRunListItem(
            file_name=file_path.name,
            path=str(file_path),
            size_bytes=file_path.stat().st_size,
            modified_at=datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc),
        )
        for file_path in sorted(_runs_dir().glob("*.json"), reverse=True)
    ]

    return TelemetryRunListResponse(runs=runs)
