from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MobileSessionCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    camera_url_base: str | None = Field(default=None, alias="cameraUrlBase")


class MobileSessionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    camera_url: str = Field(alias="cameraUrl")
    created_at: datetime = Field(alias="createdAt")
    expires_at: datetime = Field(alias="expiresAt")
    phone_connected: bool = Field(alias="phoneConnected")
    dashboard_connected: bool = Field(alias="dashboardConnected")
    streaming: bool
    status: str


class MobileFrameMetaMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["frame_meta"]
    frame_id: str = Field(alias="frameId", min_length=1)
    client_sent_at: str = Field(alias="clientSentAt", min_length=1)
    content_type: Literal["image/jpeg"] = Field(alias="contentType")
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    encoding_ms: float | None = Field(default=None, alias="encodingMs", ge=0.0)
    frame_size_bytes: int | None = Field(default=None, alias="frameSizeBytes", ge=0)


class PhoneTelemetryMessage(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["phone_telemetry"]
    target_transmission_fps: float | None = Field(default=None, alias="targetTransmissionFps", ge=0.0)
    phone_transmission_fps: float | None = Field(default=None, alias="phoneTransmissionFps", ge=0.0)
    phone_encoding_latency_ms: float | None = Field(default=None, alias="phoneEncodingLatencyMs", ge=0.0)
    average_frame_size_bytes: float | None = Field(default=None, alias="averageFrameSizeBytes", ge=0.0)
    phone_upload_bitrate_kbps: float | None = Field(default=None, alias="phoneUploadBitrateKbps", ge=0.0)
    dropped_frames: float | None = Field(default=None, alias="droppedFrames", ge=0.0)
    frame_skip_rate: float | None = Field(default=None, alias="frameSkipRate", ge=0.0)
    camera_facing: str | None = Field(default=None, alias="cameraFacing")
    capture_width: int | None = Field(default=None, alias="captureWidth", ge=1)
    capture_height: int | None = Field(default=None, alias="captureHeight", ge=1)
    preview_width: int | None = Field(default=None, alias="previewWidth", ge=1)
    preview_height: int | None = Field(default=None, alias="previewHeight", ge=1)


class DashboardControlMessage(BaseModel):
    type: Literal["start_stream", "stop_stream"]


JsonObject = dict[str, Any]
