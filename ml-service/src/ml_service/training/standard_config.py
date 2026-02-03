"""Standard training configuration for fair model comparison.

All training scripts should import and use these settings to ensure
consistent and comparable results.

Usage:
    from ml_service.training.standard_config import STANDARD_CONFIG, get_data_paths, evaluate_model
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


@dataclass
class StandardConfig:
    """Standard training configuration."""

    # Model
    base_model: str = "beomi/KcELECTRA-base-v2022"
    max_length: int = 256
    num_labels: int = 2

    # Training
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Loss
    loss_function: str = "FocalLoss"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Optimizer
    optimizer: str = "AdamW"
    scheduler: str = "linear_warmup"
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    # Data
    dataset_version: str = "korean_standard_v1"
    data_dir: str = "data/korean"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_model": self.base_model,
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_clip": self.gradient_clip,
            "loss_function": self.loss_function,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "use_amp": self.use_amp,
            "seed": self.seed,
            "dataset_version": self.dataset_version,
        }


# Global standard configuration
STANDARD_CONFIG = StandardConfig()


@dataclass
class PMFModelConfig:
    """Configuration for PMF ensemble models."""

    name: str
    pretrained: str
    output_dir: str
    description: str


# PMF Ensemble models - all use STANDARD_CONFIG hyperparameters
PMF_MODELS = [
    PMFModelConfig(
        name="kcelectra",
        pretrained="beomi/KcELECTRA-base-v2022",
        output_dir="models/pmf/kcelectra",
        description="Comment-specialized Korean ELECTRA (baseline)",
    ),
    PMFModelConfig(
        name="klue-bert",
        pretrained="klue/bert-base",
        output_dir="models/pmf/klue-bert",
        description="General-purpose Korean BERT from KLUE",
    ),
    PMFModelConfig(
        name="koelectra-v3",
        pretrained="monologg/koelectra-base-v3-discriminator",
        output_dir="models/pmf/koelectra-v3",
        description="KoELECTRA v3 discriminator",
    ),
]


@dataclass
class CoevolutionConfig:
    """Configuration for coevolution training."""

    retrain_epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    attack_batch_size: int = 150
    attack_variants: int = 10


COEVOLUTION_CONFIG = CoevolutionConfig()


def get_data_paths(
    data_dir: str | Path | None = None,
    version: str = "korean_standard_v1",
) -> dict[str, Path]:
    """Get standardized data file paths.

    Args:
        data_dir: Data directory path. If None, uses default.
        version: Dataset version prefix.

    Returns:
        Dictionary with train, valid, test paths.
    """
    if data_dir is None:
        # Default path relative to ml-service
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "korean"
    else:
        data_dir = Path(data_dir)

    return {
        "train": data_dir / f"{version}_train.csv",
        "valid": data_dir / f"{version}_valid.csv",
        "test": data_dir / f"{version}_test.csv",
        "metadata": data_dir / f"{version}_metadata.json",
    }


def evaluate_model(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
) -> dict[str, Any]:
    """Standard evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (optional).

    Returns:
        Dictionary of evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        # Primary metric (model selection criterion)
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        # Secondary metrics
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Confusion matrix
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return metrics


def is_better_model(
    new_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    primary_metric: str = "f1_weighted",
    secondary_metric: str = "fn",
) -> bool:
    """Compare models: primary metric first, then secondary.

    Args:
        new_metrics: Metrics from new model.
        best_metrics: Metrics from current best model.
        primary_metric: Primary comparison metric (higher is better).
        secondary_metric: Secondary metric for tie-breaking (lower is better for fn/fp).

    Returns:
        True if new model is better.
    """
    if not best_metrics:
        return True

    new_primary = new_metrics.get(primary_metric, 0)
    best_primary = best_metrics.get(primary_metric, 0)

    if new_primary > best_primary:
        return True
    elif new_primary == best_primary:
        # Tie-breaker: lower FN is better
        new_secondary = new_metrics.get(secondary_metric, float("inf"))
        best_secondary = best_metrics.get(secondary_metric, float("inf"))
        return new_secondary < best_secondary

    return False


def set_seed(seed: int = STANDARD_CONFIG.seed) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model, config: StandardConfig | None = None):
    """Get standard optimizer."""
    import torch

    if config is None:
        config = STANDARD_CONFIG

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def get_scheduler(optimizer, num_training_steps: int, config: StandardConfig | None = None):
    """Get standard learning rate scheduler."""
    from transformers import get_linear_schedule_with_warmup

    if config is None:
        config = STANDARD_CONFIG

    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def get_loss_function(config: StandardConfig | None = None):
    """Get standard loss function."""
    from ml_service.training.losses import FocalLoss

    if config is None:
        config = STANDARD_CONFIG

    return FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
