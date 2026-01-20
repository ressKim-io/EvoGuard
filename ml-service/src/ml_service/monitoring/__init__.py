"""Monitoring module for EvoGuard ML Service.

Provides Prometheus metrics and monitoring utilities.
"""

from ml_service.monitoring.metrics import (
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    PREDICTION_CONFIDENCE,
    EVASION_RATE,
    ATTACK_COUNTER,
    ATTACK_SUCCESS_COUNTER,
    COEVOLUTION_CYCLE,
    DEFENDER_RETRAIN,
    ATTACKER_EVOLUTION,
    MODEL_F1_SCORE,
    MODEL_ACCURACY,
    DATA_DRIFT_SCORE,
    setup_metrics,
    record_prediction,
    record_attack,
    record_coevolution_cycle,
    update_model_metrics,
    update_evasion_rate,
)

__all__ = [
    "PREDICTION_COUNTER",
    "PREDICTION_LATENCY",
    "PREDICTION_CONFIDENCE",
    "EVASION_RATE",
    "ATTACK_COUNTER",
    "ATTACK_SUCCESS_COUNTER",
    "COEVOLUTION_CYCLE",
    "DEFENDER_RETRAIN",
    "ATTACKER_EVOLUTION",
    "MODEL_F1_SCORE",
    "MODEL_ACCURACY",
    "DATA_DRIFT_SCORE",
    "setup_metrics",
    "record_prediction",
    "record_attack",
    "record_coevolution_cycle",
    "update_model_metrics",
    "update_evasion_rate",
]
