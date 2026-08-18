"""Application settings loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the API."""

    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")

    app_name: str = "Sign Language Recognition API"
    environment: str = "development"
    log_level: str = "INFO"
    model_path: Path = PROJECT_DIR / "model" / "efficientnet_asl_weights.pth"
    labels_path: Path | None = None
    model_architecture: str = "efficientnet_b0"
    image_size: int = Field(default=224, ge=32, le=2048)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=20_000_000, ge=1)
    allowed_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""
    return Settings()
