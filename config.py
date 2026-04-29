"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


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

    # --- Base URL (used for dev-mode upload URLs returned to clients) ---
    BASE_URL: str = "http://localhost:8000"

    # --- Auth ---
    API_KEY: str = "dev-api-key-change-in-prod"
    JWT_SECRET: str = "super-secret-jwt-key-change-in-prod"
    ENVIRONMENT: str = Field(default="development", description="development or production")

    # --- AI Worker ---
    MODEL_DEVICE: str = Field(default="cpu", description="cpu or cuda")
    SAM_MODEL: str = Field(default="vit_b", description="SAM model variant")
    SAM2_MODEL: str = Field(
        default="facebook/sam2.1-hiera-tiny",
        description="SAM2 model ID (tiny/small/base-plus/large)",
    )
    USE_SAM2: bool = Field(default=True, description="Enable SAM2 tracking pipeline")
    FRAME_SAMPLE_RATE: int = Field(default=1, description="Frames per second to sample")

    # --- Production safety ---
    ALLOWED_ORIGINS: str = Field(default="", description="Comma-separated production CORS origins")
    MAX_UPLOAD_BYTES: int = Field(default=250 * 1024 * 1024, description="Maximum uploaded clip size")
    ALLOWED_UPLOAD_CONTENT_TYPES: str = Field(default="video/mp4,video/quicktime,video/x-m4v")

    @model_validator(mode="after")
    def validate_production_settings(self):
        if self.ENVIRONMENT.lower() != "production":
            return self

        weak_values = {
            "API_KEY": self.API_KEY in {"dev-api-key-change-in-prod", "dev-api-key-change-me", ""},
            "JWT_SECRET": self.JWT_SECRET in {"super-secret-jwt-key-change-in-prod", "", "change-me"}
            or len(self.JWT_SECRET) < 32,
            "S3_ACCESS_KEY": self.S3_ACCESS_KEY in {"minioadmin", "", "change-me"},
            "S3_SECRET_KEY": self.S3_SECRET_KEY in {"minioadmin", "", "change-me"},
        }
        bad = [name for name, is_bad in weak_values.items() if is_bad]
        if bad:
            raise ValueError(f"Unsafe production settings: {', '.join(bad)} must be set to strong non-default values")
        if not self.ALLOWED_ORIGINS.strip():
            raise ValueError("ALLOWED_ORIGINS must be set in production")
        return self

    @property
    def allowed_origins_list(self) -> list[str]:
        if self.ENVIRONMENT.lower() == "development" and not self.ALLOWED_ORIGINS.strip():
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def allowed_upload_content_types_set(self) -> set[str]:
        return {item.strip().lower() for item in self.ALLOWED_UPLOAD_CONTENT_TYPES.split(",") if item.strip()}


settings = Settings()
