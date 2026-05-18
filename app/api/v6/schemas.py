from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionStartMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["session_start"]
    session_id: str = Field(alias="sessionId", min_length=1)
    client_started_at: float | None = Field(default=None, alias="clientStartedAt")
    target_transmission_fps: float | None = Field(default=None, alias="targetTransmissionFps", ge=0.0)
    transport: Literal["websocket"] = "websocket"


class FrameMetaMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["frame_meta"]
    session_id: str = Field(alias="sessionId", min_length=1)
    frame_id: str = Field(alias="frameId", min_length=1)
    client_sent_at: str = Field(alias="clientSentAt", min_length=1)
    content_type: Literal["image/jpeg"] = Field(alias="contentType")
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    encoding_ms: float | None = Field(default=None, alias="encodingMs", ge=0.0)


class SessionEndMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["session_end"]
    session_id: str = Field(alias="sessionId", min_length=1)
