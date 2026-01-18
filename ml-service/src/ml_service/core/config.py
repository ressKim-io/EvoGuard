"""Configuration management for ml-service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Model settings
    model_name: str = Field(default="mock", description="Model name to use")
    model_path: str | None = Field(default=None, description="Path to model weights")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Classification confidence threshold"
    )

    # MLflow settings
    mlflow_tracking_uri: str | None = Field(default=None, description="MLflow tracking URI")
    mlflow_model_alias: str = Field(default="champion", description="MLflow model alias to load")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json or text)")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
