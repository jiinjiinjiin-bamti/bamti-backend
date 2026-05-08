from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


backend_root = Path(__file__).resolve().parents[2]
workspace_root = backend_root.parent


class Settings(BaseSettings):
    app_name: str = Field(default="DMS Backend", validation_alias=AliasChoices("APP_NAME", "BAMTI_APP_NAME"))
    app_version: str = Field(default="0.1.0", validation_alias=AliasChoices("APP_VERSION", "BAMTI_APP_VERSION"))
    environment: str = Field(default="local", validation_alias=AliasChoices("ENVIRONMENT", "BAMTI_ENVIRONMENT"))
    inference_runner: str = Field(default="bamti-torch", validation_alias=AliasChoices("INFERENCE_RUNNER", "BAMTI_INFERENCE_RUNNER"))
    model_path: Path = Field(
        default=workspace_root / "model" / "final_model.pth",
        validation_alias=AliasChoices("MODEL_PATH", "BAMTI_MODEL_PATH"),
    )
    model_device: str = Field(default="cpu", validation_alias=AliasChoices("MODEL_DEVICE", "BAMTI_MODEL_DEVICE"))
    model_input_size: int = Field(default=224, validation_alias=AliasChoices("MODEL_INPUT_SIZE", "BAMTI_MODEL_INPUT_SIZE"))
    model_score_activation: str = Field(
        default="softmax",
        validation_alias=AliasChoices("MODEL_SCORE_ACTIVATION", "BAMTI_MODEL_SCORE_ACTIVATION"),
    )
    torch_num_threads: int = Field(default=5, validation_alias=AliasChoices("TORCH_NUM_THREADS", "BAMTI_TORCH_NUM_THREADS"))
    telemetry_runs_dir: Path = Field(
        default=backend_root / "telemetry_runs",
        validation_alias=AliasChoices("TELEMETRY_RUNS_DIR", "BAMTI_TELEMETRY_RUNS_DIR"),
    )
    max_frame_bytes: int = Field(default=1_048_576, validation_alias=AliasChoices("MAX_FRAME_BYTES", "BAMTI_MAX_FRAME_BYTES"))
    cors_allowed_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    database_url: str = "mysql+aiomysql://dms_user:dms_password@localhost:3306/dms"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
