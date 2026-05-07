from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
