"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request model for text classification."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    request_id: str | None = Field(default=None, description="Optional request ID for tracking")


class ClassifyBatchRequest(BaseModel):
    """Request model for batch text classification."""

    texts: list[str] = Field(
        ..., min_length=1, max_length=100, description="List of texts to classify"
    )
    request_id: str | None = Field(default=None, description="Optional request ID for tracking")


class ClassificationResult(BaseModel):
    """Single classification result."""

    text: str = Field(..., description="Original text")
    is_toxic: bool = Field(..., description="Whether the text is classified as toxic")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    label: str = Field(..., description="Classification label (toxic/non-toxic)")


class ClassifyResponse(BaseModel):
    """Response model for single text classification."""

    success: bool = Field(default=True, description="Whether the request was successful")
    result: ClassificationResult = Field(..., description="Classification result")
    model_version: str = Field(..., description="Model version used for classification")
    request_id: str | None = Field(default=None, description="Request ID if provided")


class ClassifyBatchResponse(BaseModel):
    """Response model for batch text classification."""

    success: bool = Field(default=True, description="Whether the request was successful")
    results: list[ClassificationResult] = Field(..., description="List of classification results")
    model_version: str = Field(..., description="Model version used for classification")
    request_id: str | None = Field(default=None, description="Request ID if provided")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_version: str = Field(..., description="Current model version")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
