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

    # Feature Store settings
    feature_store_db_url: str | None = Field(
        default=None,
        description="PostgreSQL URL for Feature Store (postgresql+asyncpg://...)",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json or text)")

    # Feature Store TTL settings (seconds)
    text_features_ttl_seconds: int = Field(
        default=86400, description="TTL for text features (24 hours)"
    )
    battle_features_ttl_seconds: int = Field(
        default=3600, description="TTL for battle features (1 hour)"
    )
    user_features_ttl_seconds: int = Field(
        default=21600, description="TTL for user features (6 hours)"
    )
    default_features_ttl_seconds: int = Field(
        default=86400, description="Default TTL for features (24 hours)"
    )

    # Monitoring settings
    low_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold below which predictions are considered low confidence",
    )
    prediction_buffer_size: int = Field(
        default=1000, ge=100, description="Size of the in-memory prediction buffer"
    )
    prediction_sample_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Rate at which to sample predictions for detailed logging",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
