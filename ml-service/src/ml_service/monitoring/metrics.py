"""Prometheus metrics for EvoGuard ML Service.

Defines all metrics for monitoring ML model performance,
adversarial pipeline, and system health.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response

# =============================================================================
# ML Model Metrics
# =============================================================================

PREDICTION_COUNTER = Counter(
    'model_prediction_total',
    'Total number of predictions',
    ['model_version', 'label']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Prediction confidence score distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

MODEL_F1_SCORE = Gauge(
    'model_f1_score',
    'Current model F1 score'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy'
)

MODEL_INFO = Info(
    'model',
    'Model information'
)

# =============================================================================
# Adversarial Pipeline Metrics
# =============================================================================

EVASION_RATE = Gauge(
    'adversarial_evasion_rate',
    'Current adversarial evasion rate'
)

ATTACK_COUNTER = Counter(
    'adversarial_attack_total',
    'Total number of adversarial attacks',
    ['strategy']
)

ATTACK_SUCCESS_COUNTER = Counter(
    'adversarial_attack_success_total',
    'Total number of successful adversarial attacks (evasions)',
    ['strategy']
)

COEVOLUTION_CYCLE = Counter(
    'coevolution_cycle_total',
    'Total number of co-evolution cycles',
    ['action']  # retrain_defender, evolve_attacker, balanced
)

DEFENDER_RETRAIN = Counter(
    'defender_retrain_total',
    'Total number of defender retraining events'
)

ATTACKER_EVOLUTION = Counter(
    'attacker_evolution_total',
    'Total number of attacker evolution events'
)

COEVOLUTION_CYCLE_DURATION = Histogram(
    'coevolution_cycle_duration_seconds',
    'Duration of co-evolution cycles',
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800]
)

# =============================================================================
# Data Drift Metrics
# =============================================================================

DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Data drift score (KL divergence or similar)'
)

FEATURE_DRIFT = Gauge(
    'feature_drift_detected',
    'Feature drift detection flag',
    ['feature_name']
)

INPUT_TEXT_LENGTH = Histogram(
    'input_text_length',
    'Distribution of input text lengths',
    buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
)

# =============================================================================
# System Metrics
# =============================================================================

GPU_MEMORY_USED = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes'
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage'
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time to load the model'
)


# =============================================================================
# Helper Functions
# =============================================================================

def setup_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics endpoint for FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


def record_prediction(
    label: int,
    confidence: float,
    latency: float,
    model_version: str = "v1"
) -> None:
    """Record a prediction event.
    
    Args:
        label: Predicted label (0 or 1)
        confidence: Prediction confidence
        latency: Prediction latency in seconds
        model_version: Model version string
    """
    label_name = "toxic" if label == 1 else "non-toxic"
    PREDICTION_COUNTER.labels(model_version=model_version, label=label_name).inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CONFIDENCE.observe(confidence)


def record_attack(strategy: str, success: bool) -> None:
    """Record an adversarial attack attempt.
    
    Args:
        strategy: Name of the attack strategy
        success: Whether the attack successfully evaded detection
    """
    ATTACK_COUNTER.labels(strategy=strategy).inc()
    if success:
        ATTACK_SUCCESS_COUNTER.labels(strategy=strategy).inc()


def record_coevolution_cycle(
    action: str,
    duration: float,
    evasion_rate: float
) -> None:
    """Record a co-evolution cycle.
    
    Args:
        action: Action taken (retrain_defender, evolve_attacker, balanced)
        duration: Cycle duration in seconds
        evasion_rate: Evasion rate for this cycle
    """
    COEVOLUTION_CYCLE.labels(action=action).inc()
    COEVOLUTION_CYCLE_DURATION.observe(duration)
    EVASION_RATE.set(evasion_rate)
    
    if action == "retrain_defender":
        DEFENDER_RETRAIN.inc()
    elif action == "evolve_attacker":
        ATTACKER_EVOLUTION.inc()


def update_model_metrics(f1_score: float, accuracy: float) -> None:
    """Update model performance metrics.
    
    Args:
        f1_score: Model F1 score
        accuracy: Model accuracy
    """
    MODEL_F1_SCORE.set(f1_score)
    MODEL_ACCURACY.set(accuracy)


def update_evasion_rate(rate: float) -> None:
    """Update the current evasion rate.
    
    Args:
        rate: Evasion rate (0.0 to 1.0)
    """
    EVASION_RATE.set(rate)


def update_data_drift(score: float) -> None:
    """Update data drift score.
    
    Args:
        score: Drift score value
    """
    DATA_DRIFT_SCORE.set(score)


@contextmanager
def track_prediction_latency():
    """Context manager to track prediction latency."""
    start = time.perf_counter()
    yield
    PREDICTION_LATENCY.observe(time.perf_counter() - start)


def track_latency(metric: Histogram):
    """Decorator to track function latency.
    
    Args:
        metric: Histogram metric to record latency
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                metric.observe(time.perf_counter() - start)
        return wrapper
    return decorator
