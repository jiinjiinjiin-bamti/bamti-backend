from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.inference.schemas import DetectionScore, InferenceTelemetry, ModelRuntimeInfo


class InferenceFrameResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    frame_id: str | None = Field(alias="frameId")
    client_sent_at: str | None = Field(alias="clientSentAt")
    server_received_at: float = Field(alias="serverReceivedAt", ge=0.0)
    server_responded_at: float = Field(alias="serverRespondedAt", ge=0.0)
    detections: list[DetectionScore]
    model: ModelRuntimeInfo
    telemetry: InferenceTelemetry


class TelemetryRunSavedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    file_name: str = Field(alias="fileName")
    path: str
    saved: Literal[True]


class TelemetryRunListItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_name: str = Field(alias="fileName")
    path: str
    size_bytes: int = Field(alias="sizeBytes", ge=0)
    modified_at: datetime = Field(alias="modifiedAt")


class TelemetryRunListResponse(BaseModel):
    runs: list[TelemetryRunListItem]


TelemetryPayload = dict[str, Any]
