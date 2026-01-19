"""Training module for QLoRA fine-tuning.

This module provides:
- TrainingConfig: Configuration for training
- DataProcessor: Data preprocessing pipeline (requires training deps)
- QLoRATrainer: QLoRA fine-tuning trainer (requires training deps)
- MLflowTracker: MLflow experiment tracking (requires training deps)
- JigsawDatasetLoader: Load Jigsaw datasets from HuggingFace
"""

from ml_service.training.config import DataConfig, LoRAConfig, TrainingConfig

__all__ = [
    "DataConfig",
    "DataProcessor",
    "JigsawDatasetLoader",
    "LoRAConfig",
    "MLflowTracker",
    "QLoRATrainer",
    "TextDataset",
    "TrainingConfig",
    "get_sample_data",
]

# These require optional training dependencies
try:
    from ml_service.training.data import DataProcessor, TextDataset
    from ml_service.training.datasets import JigsawDatasetLoader, get_sample_data
    from ml_service.training.mlflow_utils import MLflowTracker
    from ml_service.training.trainer import QLoRATrainer
except ImportError:
    DataProcessor = None  # type: ignore[assignment, misc]
    TextDataset = None  # type: ignore[assignment, misc]
    JigsawDatasetLoader = None  # type: ignore[assignment, misc]
    get_sample_data = None  # type: ignore[assignment, misc]
    MLflowTracker = None  # type: ignore[assignment, misc]
    QLoRATrainer = None  # type: ignore[assignment, misc]
