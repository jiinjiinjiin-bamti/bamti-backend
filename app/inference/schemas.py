from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectionClass(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    variable_name: str = Field(alias="variableName")
    class_id: str = Field(alias="classId")
    display_name: str = Field(alias="displayName")
    description: str
    threshold: float = Field(ge=0.0, le=1.0)


class DetectionScore(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    variable_name: str = Field(alias="variableName")
    class_id: str = Field(alias="classId")
    display_name: str = Field(alias="displayName")
    score: float = Field(ge=0.0, le=1.0)


class ModelManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_version: str = Field(alias="modelVersion")
    classes: tuple[DetectionClass, ...]


class InferenceResult(BaseModel):
    is_distracted: bool
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class FrameMeta(BaseModel):
    type: Literal["frame_meta"]
    frame_id: str
    captured_at: datetime
    content_type: str


class QueuedFrame(BaseModel):
    session_id: str
    meta: FrameMeta
    jpeg_bytes: bytes

    model_config = {"arbitrary_types_allowed": True}
