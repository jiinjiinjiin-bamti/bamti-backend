from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DMS Backend"
    environment: str = "local"
    inference_runner: str = "mock"
    frame_queue_size: int = 4
    websocket_idle_timeout_seconds: float = 30.0
    websocket_drain_timeout_seconds: float = 5.0
    max_frame_bytes: int = 1_048_576
    cors_allowed_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    )
    database_url: str = (
        "mysql+asyncmy://dms_user:dms_password@localhost:3306/dms"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
