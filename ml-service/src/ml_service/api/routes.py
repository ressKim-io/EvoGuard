"""API routes for ml-service."""

from fastapi import APIRouter, HTTPException, status
from prometheus_client import Counter, Histogram

from ml_service.api.schemas import (
    ClassificationResult,
    ClassifyBatchRequest,
    ClassifyBatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
)
from ml_service.services.inference import get_inference_service

# Prometheus metrics
CLASSIFY_REQUESTS = Counter(
    "ml_classify_requests_total",
    "Total number of classification requests",
    ["endpoint", "status"],
)
CLASSIFY_LATENCY = Histogram(
    "ml_classify_latency_seconds",
    "Classification request latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
TOXIC_CLASSIFICATIONS = Counter(
    "ml_toxic_classifications_total",
    "Total number of texts classified as toxic",
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Check service health status."""
    service = get_inference_service()
    return HealthResponse(
        status="healthy" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_version=service.model_version,
    )


@router.get("/ready", response_model=HealthResponse, tags=["health"])
async def readiness_check() -> HealthResponse:
    """Check if service is ready to accept requests."""
    service = get_inference_service()
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=service.model_version,
    )


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["classification"],
)
async def classify_text(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a single text for toxic content.

    Args:
        request: The classification request containing the text.

    Returns:
        ClassifyResponse with classification result.
    """
    service = get_inference_service()

    if not service.is_ready:
        CLASSIFY_REQUESTS.labels(endpoint="classify", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    with CLASSIFY_LATENCY.labels(endpoint="classify").time():
        result = service.classify(request.text)

    CLASSIFY_REQUESTS.labels(endpoint="classify", status="success").inc()
    if result.is_toxic:
        TOXIC_CLASSIFICATIONS.inc()

    return ClassifyResponse(
        success=True,
        result=ClassificationResult(
            text=request.text,
            is_toxic=result.is_toxic,
            confidence=result.confidence,
            label=result.label,
        ),
        model_version=service.model_version,
        request_id=request.request_id,
    )


@router.post(
    "/classify/batch",
    response_model=ClassifyBatchResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["classification"],
)
async def classify_batch(request: ClassifyBatchRequest) -> ClassifyBatchResponse:
    """Classify multiple texts for toxic content.

    Args:
        request: The batch classification request containing texts.

    Returns:
        ClassifyBatchResponse with classification results.
    """
    service = get_inference_service()

    if not service.is_ready:
        CLASSIFY_REQUESTS.labels(endpoint="classify_batch", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    with CLASSIFY_LATENCY.labels(endpoint="classify_batch").time():
        results = service.classify_batch(request.texts)

    CLASSIFY_REQUESTS.labels(endpoint="classify_batch", status="success").inc()

    classification_results = []
    for text, result in zip(request.texts, results, strict=True):
        if result.is_toxic:
            TOXIC_CLASSIFICATIONS.inc()
        classification_results.append(
            ClassificationResult(
                text=text,
                is_toxic=result.is_toxic,
                confidence=result.confidence,
                label=result.label,
            )
        )

    return ClassifyBatchResponse(
        success=True,
        results=classification_results,
        model_version=service.model_version,
        request_id=request.request_id,
    )
