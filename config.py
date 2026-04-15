"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Core settings for the AR Survey & Inspection backend."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Database ---
    DATABASE_URL: str = "postgresql://arsurvey:arsurvey@localhost:5432/ar_survey"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- S3 / MinIO ---
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_BUCKET: str = "ar-survey-clips"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"

    # --- Auth ---
    API_KEY: str = "dev-api-key-change-me"

    # --- AI Worker ---
    MODEL_DEVICE: str = Field(default="cpu", description="cpu or cuda")
    SAM_MODEL: str = Field(default="vit_b", description="SAM model variant")
    FRAME_SAMPLE_RATE: int = Field(default=1, description="Frames per second to sample")


settings = Settings()
