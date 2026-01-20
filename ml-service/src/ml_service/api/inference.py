"""Inference API for toxic text classification.

Production-ready API endpoint for text moderation.

Usage:
    # Start server
    uvicorn ml_service.api.inference:app --host 0.0.0.0 --port 8000

    # Call API
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "hello world"}'
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global classifier instance
_classifier = None


def get_classifier():
    """Get or load the classifier."""
    global _classifier
    if _classifier is None:
        from ml_service.pipeline.classifier_adapter import TrainedClassifierAdapter

        # Use original balanced model for production
        # (adversarial-retrained has class imbalance issue)
        model_paths = [
            Path("models/toxic-classifier"),
            Path("models/adversarial-retrained"),
        ]

        for path in model_paths:
            if path.exists():
                logger.info(f"Loading model from {path}")
                _classifier = TrainedClassifierAdapter(path)
                _classifier.load()
                break

        if _classifier is None:
            raise RuntimeError("No trained model found")

    return _classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading classifier model...")
    try:
        get_classifier()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="EvoGuard Toxic Text Classifier",
    description="API for detecting toxic/inappropriate text content",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class PredictRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    """Prediction response."""

    toxic: bool = Field(..., description="Whether text is toxic")
    confidence: float = Field(..., description="Confidence score (0-1)")
    label: str = Field(..., description="Label name: toxic or non-toxic")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    texts: list[str] = Field(..., description="List of texts to classify", min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    results: list[PredictResponse]
    total: int
    toxic_count: int
    latency_ms: float


class ModerateRequest(BaseModel):
    """Content moderation request."""

    text: str = Field(..., description="Text to moderate")
    threshold: float = Field(default=0.5, description="Confidence threshold for toxic classification")


class ModerateResponse(BaseModel):
    """Content moderation response."""

    allowed: bool = Field(..., description="Whether content is allowed")
    toxic: bool
    confidence: float
    reason: str | None = Field(None, description="Reason if blocked")


# Endpoints
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness check - verifies model is loaded."""
    try:
        classifier = get_classifier()
        # Quick test
        classifier.predict(["test"])
        return {"ready": True, "model": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict if text is toxic.

    Returns:
        PredictResponse with toxic flag and confidence.
    """
    start_time = time.perf_counter()

    try:
        classifier = get_classifier()
        results = classifier.predict([request.text])
        result = results[0]

        latency_ms = (time.perf_counter() - start_time) * 1000

        return PredictResponse(
            toxic=result["label"] == 1,
            confidence=result["confidence"],
            label=result["label_name"],
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Batch prediction for multiple texts.

    More efficient than multiple single predictions.
    """
    start_time = time.perf_counter()

    try:
        classifier = get_classifier()
        results = classifier.predict(request.texts)

        latency_ms = (time.perf_counter() - start_time) * 1000

        responses = []
        toxic_count = 0

        for result in results:
            is_toxic = result["label"] == 1
            if is_toxic:
                toxic_count += 1

            responses.append(PredictResponse(
                toxic=is_toxic,
                confidence=result["confidence"],
                label=result["label_name"],
                latency_ms=0,  # Individual latency not tracked in batch
            ))

        return BatchPredictResponse(
            results=responses,
            total=len(responses),
            toxic_count=toxic_count,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/moderate", response_model=ModerateResponse)
async def moderate_content(request: ModerateRequest) -> ModerateResponse:
    """Content moderation endpoint.

    Use this for filtering user-generated content.
    Returns whether content should be allowed or blocked.
    """
    try:
        classifier = get_classifier()
        results = classifier.predict([request.text])
        result = results[0]

        is_toxic = result["label"] == 1
        confidence = result["confidence"]

        # Apply threshold
        should_block = is_toxic and confidence >= request.threshold

        return ModerateResponse(
            allowed=not should_block,
            toxic=is_toxic,
            confidence=confidence,
            reason="Toxic content detected" if should_block else None,
        )

    except Exception as e:
        logger.error(f"Moderation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
