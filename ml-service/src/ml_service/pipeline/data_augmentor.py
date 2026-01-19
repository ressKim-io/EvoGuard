"""Data augmentor for adversarial pipeline.

This module augments training data with failed samples and generates
additional variants for improved model robustness.
"""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ml_service.pipeline.config import RetrainConfig, StorageConfig
from ml_service.pipeline.sample_collector import FailedSample

logger = logging.getLogger(__name__)

# Add attacker module to path if needed
ATTACKER_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "attacker"
if str(ATTACKER_PATH) not in sys.path:
    sys.path.insert(0, str(ATTACKER_PATH.parent))

if TYPE_CHECKING:
    from datasets import DatasetDict


@dataclass
class AugmentedDataset:
    """Result of data augmentation."""

    original_count: int
    augmented_count: int
    total_count: int
    samples: list[dict[str, Any]]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(self.samples)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_count": self.original_count,
            "augmented_count": self.augmented_count,
            "total_count": self.total_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class TrainingDataAugmentor:
    """Augments training data with failed samples for retraining.

    Takes failed samples from the sample collector and generates
    augmented training data by:
    1. Including the original toxic text with correct label
    2. Including the variant that evaded detection with correct label
    3. Generating additional variants for each failed sample

    Example:
        >>> augmentor = TrainingDataAugmentor(retrain_config, storage_config)
        >>> result = augmentor.augment(failed_samples)
        >>> df = result.to_dataframe()
    """

    def __init__(
        self,
        retrain_config: RetrainConfig,
        storage_config: StorageConfig,
    ) -> None:
        """Initialize data augmentor.

        Args:
            retrain_config: Retraining configuration.
            storage_config: Storage configuration.
        """
        self.retrain_config = retrain_config
        self.storage_config = storage_config
        self._orchestrator = None
        storage_config.augmented_data_dir.mkdir(parents=True, exist_ok=True)

    def _get_orchestrator(self) -> Any:
        """Get attack orchestrator for additional augmentation."""
        if self._orchestrator is None:
            try:
                from attacker.orchestrator import (
                    AttackOrchestrator,
                    OrchestratorConfig,
                    SelectionMode,
                )

                config = OrchestratorConfig(
                    selection_mode=SelectionMode.ALL,
                    include_llm=False,  # Rule-based only for augmentation
                )
                self._orchestrator = AttackOrchestrator(config)
            except ImportError:
                logger.warning("Attacker module not available, using simple augmentation")
                self._orchestrator = None
        return self._orchestrator

    def augment(
        self,
        failed_samples: list[FailedSample],
        original_data: pd.DataFrame | None = None,
    ) -> AugmentedDataset:
        """Augment training data with failed samples.

        Args:
            failed_samples: List of failed samples to augment from.
            original_data: Optional original training data to merge with.

        Returns:
            AugmentedDataset with all samples.
        """
        timestamp = datetime.now(UTC)
        augmented_samples: list[dict[str, Any]] = []

        logger.info(f"Augmenting data with {len(failed_samples)} failed samples")

        # Load original data if configured and not provided
        if original_data is None and self.retrain_config.merge_with_original:
            if self.retrain_config.original_data_path:
                original_data = self._load_original_data()

        original_count = len(original_data) if original_data is not None else 0

        # Process failed samples
        for sample in failed_samples:
            # Add original toxic text
            augmented_samples.append({
                "text": sample.original_text,
                "label": 1,  # toxic
                "source": "original",
                "strategy": "none",
            })

            # Add the variant that evaded detection
            augmented_samples.append({
                "text": sample.variant_text,
                "label": 1,  # toxic (correct label)
                "source": "evasion",
                "strategy": sample.strategy_name,
            })

            # Generate additional variants
            additional = self._generate_variants(
                sample.original_text,
                num_variants=self.retrain_config.augmentation_multiplier - 1,
            )
            augmented_samples.extend(additional)

        # Limit augmented samples if needed
        if len(augmented_samples) > self.retrain_config.max_augmented_samples:
            random.shuffle(augmented_samples)
            augmented_samples = augmented_samples[:self.retrain_config.max_augmented_samples]
            logger.info(f"Limited augmented samples to {len(augmented_samples)}")

        augmented_count = len(augmented_samples)

        # Merge with original data if available
        if original_data is not None and self.retrain_config.merge_with_original:
            original_samples = original_data.to_dict("records")
            # Standardize format
            for sample in original_samples:
                sample["source"] = "original_dataset"
                sample["strategy"] = "none"
            augmented_samples = original_samples + augmented_samples

        result = AugmentedDataset(
            original_count=original_count,
            augmented_count=augmented_count,
            total_count=len(augmented_samples),
            samples=augmented_samples,
            timestamp=timestamp,
            metadata={
                "num_failed_samples": len(failed_samples),
                "augmentation_multiplier": self.retrain_config.augmentation_multiplier,
                "merged_with_original": self.retrain_config.merge_with_original,
            },
        )

        logger.info(
            f"Augmentation complete: {augmented_count} augmented samples, "
            f"{result.total_count} total"
        )

        return result

    def _load_original_data(self) -> pd.DataFrame | None:
        """Load original training data."""
        if self.retrain_config.original_data_path is None:
            return None

        path = self.retrain_config.original_data_path
        if not path.exists():
            logger.warning(f"Original data file not found: {path}")
            return None

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} original training samples")
        return df

    def _generate_variants(
        self,
        text: str,
        num_variants: int,
    ) -> list[dict[str, Any]]:
        """Generate additional variants for a text.

        Args:
            text: Text to generate variants for.
            num_variants: Number of variants to generate.

        Returns:
            List of variant samples.
        """
        if num_variants <= 0:
            return []

        orchestrator = self._get_orchestrator()

        if orchestrator is None:
            # Simple augmentation without orchestrator
            return self._simple_augment(text, num_variants)

        try:
            result = orchestrator.generate(text, num_variants=num_variants)
            variants = []
            for evasion_result in result.results:
                variants.append({
                    "text": evasion_result.evasion,
                    "label": 1,  # toxic
                    "source": "augmented",
                    "strategy": evasion_result.strategy_name,
                })
            return variants
        except Exception as e:
            logger.warning(f"Variant generation failed: {e}, using simple augmentation")
            return self._simple_augment(text, num_variants)

    def _simple_augment(
        self,
        text: str,
        num_variants: int,
    ) -> list[dict[str, Any]]:
        """Simple augmentation without attack strategies.

        Applies basic transformations like case changes and spacing.

        Args:
            text: Text to augment.
            num_variants: Number of variants.

        Returns:
            List of augmented samples.
        """
        variants = []
        augmentations = [
            lambda t: t.lower(),
            lambda t: t.upper(),
            lambda t: t.title(),
            lambda t: " ".join(t.split()),  # Normalize whitespace
            lambda t: t.replace(" ", "  "),  # Double spaces
            lambda t: "".join(c + " " if i % 2 else c for i, c in enumerate(t)),
        ]

        random.shuffle(augmentations)
        for i, aug_fn in enumerate(augmentations[:num_variants]):
            try:
                augmented_text = aug_fn(text)
                if augmented_text != text:
                    variants.append({
                        "text": augmented_text,
                        "label": 1,
                        "source": "simple_augmented",
                        "strategy": f"simple_{i}",
                    })
            except Exception:
                continue

        return variants

    def save(
        self,
        dataset: AugmentedDataset,
        cycle_id: str | None = None,
    ) -> Path:
        """Save augmented dataset to file.

        Args:
            dataset: Augmented dataset to save.
            cycle_id: Optional cycle identifier.

        Returns:
            Path to saved file.
        """
        timestamp = datetime.now(UTC)
        filename = f"augmented_{cycle_id or timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.storage_config.augmented_data_dir / filename

        df = dataset.to_dataframe()
        df.to_csv(filepath, index=False)

        logger.info(f"Saved augmented dataset to {filepath}")
        return filepath

    def prepare_for_training(
        self,
        dataset: AugmentedDataset,
        tokenizer_name: str = "bert-base-uncased",
    ) -> "DatasetDict":
        """Prepare augmented data for model training.

        Args:
            dataset: Augmented dataset.
            tokenizer_name: Tokenizer to use.

        Returns:
            DatasetDict ready for training.
        """
        try:
            from ml_service.training.data import DataProcessor
            from ml_service.training.config import TrainingConfig
        except ImportError as e:
            raise ImportError(
                "Training dependencies not available. "
                "Install with: uv pip install --group training"
            ) from e

        # Create config and processor
        config = TrainingConfig(model_name=tokenizer_name)
        processor = DataProcessor(config)

        # Convert to DataFrame
        df = dataset.to_dataframe()

        # Prepare datasets
        datasets = processor.prepare_datasets(df)
        tokenized = processor.tokenize_datasets(datasets)

        return tokenized

    def get_statistics(self, dataset: AugmentedDataset) -> dict[str, Any]:
        """Get statistics about augmented dataset.

        Args:
            dataset: Augmented dataset.

        Returns:
            Dictionary of statistics.
        """
        df = dataset.to_dataframe()

        stats = {
            "total_samples": len(df),
            "original_count": dataset.original_count,
            "augmented_count": dataset.augmented_count,
            "source_distribution": df["source"].value_counts().to_dict() if "source" in df else {},
            "strategy_distribution": df["strategy"].value_counts().to_dict() if "strategy" in df else {},
            "label_distribution": df["label"].value_counts().to_dict() if "label" in df else {},
        }

        return stats
