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
        default=workspace_root / "model" / "exp04_pseudo_ir_aug_DayBest",
        validation_alias=AliasChoices("MODEL_PATH", "BAMTI_MODEL_PATH"),
    )
    aihub_model_path: Path = Field(
        default=workspace_root / "model" / "final_model.pth",
        validation_alias=AliasChoices("AIHUB_MODEL_PATH", "BAMTI_AIHUB_MODEL_PATH"),
    )
    driver4_model_path: Path = Field(
        default=workspace_root / "model" / "final_model_4cls.pth",
        validation_alias=AliasChoices("DRIVER4_MODEL_PATH", "BAMTI_DRIVER4_MODEL_PATH"),
    )
    model_device: str = Field(default="cpu", validation_alias=AliasChoices("MODEL_DEVICE", "BAMTI_MODEL_DEVICE"))
    model_input_size: int = Field(default=224, validation_alias=AliasChoices("MODEL_INPUT_SIZE", "BAMTI_MODEL_INPUT_SIZE"))
    model_score_activation: str = Field(
        default="softmax",
        validation_alias=AliasChoices("MODEL_SCORE_ACTIVATION", "BAMTI_MODEL_SCORE_ACTIVATION"),
    )
    torch_num_threads: int = Field(default=5, validation_alias=AliasChoices("TORCH_NUM_THREADS", "BAMTI_TORCH_NUM_THREADS"))
    torch_compile_backend: str = Field(
        default="inductor",
        validation_alias=AliasChoices("TORCH_COMPILE_BACKEND", "BAMTI_TORCH_COMPILE_BACKEND"),
    )
    torch_compile_mode: str = Field(
        default="reduce-overhead",
        validation_alias=AliasChoices("TORCH_COMPILE_MODE", "BAMTI_TORCH_COMPILE_MODE"),
    )
    driver4_v7_activation_threshold: float = Field(
        default=0.5,
        validation_alias=AliasChoices("DRIVER4_V7_ACTIVATION_THRESHOLD", "BAMTI_DRIVER4_V7_ACTIVATION_THRESHOLD"),
    )
    driver4_v7_decay: float = Field(default=0.95, validation_alias=AliasChoices("DRIVER4_V7_DECAY", "BAMTI_DRIVER4_V7_DECAY"))
    driver4_v7_recovery_decay: float = Field(
        default=0.85,
        validation_alias=AliasChoices("DRIVER4_V7_RECOVERY_DECAY", "BAMTI_DRIVER4_V7_RECOVERY_DECAY"),
    )
    driver4_v7_score_scale: float = Field(
        default=10.0,
        validation_alias=AliasChoices("DRIVER4_V7_SCORE_SCALE", "BAMTI_DRIVER4_V7_SCORE_SCALE"),
    )
    driver4_v7_weight_body_touching: float = Field(
        default=1.0,
        validation_alias=AliasChoices("DRIVER4_V7_WEIGHT_BODY_TOUCHING", "BAMTI_DRIVER4_V7_WEIGHT_BODY_TOUCHING"),
    )
    driver4_v7_weight_distraction: float = Field(
        default=0.8,
        validation_alias=AliasChoices("DRIVER4_V7_WEIGHT_DISTRACTION", "BAMTI_DRIVER4_V7_WEIGHT_DISTRACTION"),
    )
    driver4_v7_weight_phone_operation: float = Field(
        default=1.3,
        validation_alias=AliasChoices("DRIVER4_V7_WEIGHT_PHONE_OPERATION", "BAMTI_DRIVER4_V7_WEIGHT_PHONE_OPERATION"),
    )
    driver4_v7_weight_steering_operation: float = Field(
        default=1.0,
        validation_alias=AliasChoices("DRIVER4_V7_WEIGHT_STEERING_OPERATION", "BAMTI_DRIVER4_V7_WEIGHT_STEERING_OPERATION"),
    )
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
        "https://bamti.stableh.com",
    )
    database_url: str = "mysql+aiomysql://dms_user:dms_password@localhost:3306/dms"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
