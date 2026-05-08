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


class ModelRuntimeInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    architecture: str
    class_names: list[str] = Field(alias="classNames")
    device: str
    input_size: int = Field(alias="inputSize")
    score_activation: str = Field(alias="scoreActivation")


class InferenceTelemetry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    processing_fps: float = Field(alias="processingFps", ge=0.0)
    preprocess_ms: float = Field(alias="preprocessMs", ge=0.0)
    inference_ms: float = Field(alias="inferenceMs", ge=0.0)
    postprocess_ms: float = Field(alias="postprocessMs", ge=0.0)
    server_total_ms: float = Field(alias="serverTotalMs", ge=0.0)


class InferenceResult(BaseModel):
    detections: list[DetectionScore]
    model: ModelRuntimeInfo
    telemetry: InferenceTelemetry
