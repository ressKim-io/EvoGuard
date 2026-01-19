"""Adversarial MLOps Pipeline.

This module provides an automated pipeline for:
- Running adversarial attacks against the model
- Evaluating model quality through quality gates
- Collecting failed samples for retraining
- Augmenting training data with adversarial examples
- Promoting models through Champion/Challenger evaluation

Usage:
    from ml_service.pipeline import AdversarialPipelineOrchestrator, PipelineConfig

    config = PipelineConfig()
    orchestrator = AdversarialPipelineOrchestrator(config)
    result = await orchestrator.run_cycle()
"""

from ml_service.pipeline.config import PipelineConfig, QualityGateConfig
from ml_service.pipeline.orchestrator import (
    AdversarialPipelineOrchestrator,
    CycleResult,
    PipelineState,
)
from ml_service.pipeline.attack_runner import AttackRunner, AttackBatchResult
from ml_service.pipeline.quality_gate import QualityGate, QualityGateDecision, DecisionType
from ml_service.pipeline.sample_collector import FailedSampleCollector, FailedSample
from ml_service.pipeline.data_augmentor import TrainingDataAugmentor, AugmentedDataset
from ml_service.pipeline.model_promoter import ModelPromoter, PromotionDecision

__all__ = [
    # Config
    "PipelineConfig",
    "QualityGateConfig",
    # Orchestrator
    "AdversarialPipelineOrchestrator",
    "CycleResult",
    "PipelineState",
    # Attack
    "AttackRunner",
    "AttackBatchResult",
    # Quality Gate
    "QualityGate",
    "QualityGateDecision",
    "DecisionType",
    # Sample Collector
    "FailedSampleCollector",
    "FailedSample",
    # Data Augmentor
    "TrainingDataAugmentor",
    "AugmentedDataset",
    # Model Promoter
    "ModelPromoter",
    "PromotionDecision",
]
