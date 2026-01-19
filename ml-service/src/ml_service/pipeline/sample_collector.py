"""Failed sample collector for adversarial pipeline.

This module collects and stores samples that the model misclassified,
which are valuable for retraining to improve model robustness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml_service.pipeline.config import StorageConfig
from ml_service.pipeline.attack_runner import AttackBatchResult, AttackResult

logger = logging.getLogger(__name__)


@dataclass
class FailedSample:
    """A sample that the model failed to classify correctly."""

    original_text: str
    variant_text: str
    original_label: int
    predicted_label: int
    strategy_name: str
    confidence: float
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "variant_text": self.variant_text,
            "original_label": self.original_label,
            "predicted_label": self.predicted_label,
            "strategy_name": self.strategy_name,
            "confidence": self.confidence,
            "collected_at": self.collected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedSample":
        """Create from dictionary."""
        return cls(
            original_text=data["original_text"],
            variant_text=data["variant_text"],
            original_label=data["original_label"],
            predicted_label=data["predicted_label"],
            strategy_name=data["strategy_name"],
            confidence=data["confidence"],
            collected_at=datetime.fromisoformat(data["collected_at"]),
        )

    @classmethod
    def from_attack_result(cls, result: AttackResult) -> "FailedSample":
        """Create from AttackResult."""
        return cls(
            original_text=result.original_text,
            variant_text=result.variant_text,
            original_label=result.original_label,
            predicted_label=result.model_prediction,
            strategy_name=result.strategy_name,
            confidence=result.model_confidence,
        )


@dataclass
class CollectionResult:
    """Result of sample collection."""

    total_collected: int
    total_from_batch: int
    samples: list[FailedSample]
    storage_path: Path | None
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_collected": self.total_collected,
            "total_from_batch": self.total_from_batch,
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "timestamp": self.timestamp.isoformat(),
        }


class FailedSampleCollector:
    """Collects and manages failed samples for retraining.

    Stores samples that evaded the model's detection, organizing them
    for later use in data augmentation and retraining.

    Example:
        >>> collector = FailedSampleCollector(storage_config)
        >>> result = collector.collect(attack_batch_result)
        >>> print(f"Collected {result.total_collected} failed samples")
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize sample collector.

        Args:
            config: Storage configuration.
        """
        self.config = config
        self._samples: list[FailedSample] = []
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        self.config.failed_samples_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, attack_result: AttackBatchResult) -> CollectionResult:
        """Collect failed samples from attack batch result.

        Args:
            attack_result: Result from attack batch.

        Returns:
            CollectionResult with collection statistics.
        """
        timestamp = datetime.now(UTC)
        failed = attack_result.get_failed_samples()

        collected_samples = []
        for result in failed:
            sample = FailedSample.from_attack_result(result)
            collected_samples.append(sample)
            self._samples.append(sample)

        logger.info(
            f"Collected {len(collected_samples)} failed samples "
            f"(total in memory: {len(self._samples)})"
        )

        return CollectionResult(
            total_collected=len(collected_samples),
            total_from_batch=attack_result.total_variants,
            samples=collected_samples,
            storage_path=None,
            timestamp=timestamp,
        )

    def save(self, cycle_id: str | None = None) -> Path:
        """Save collected samples to file.

        Args:
            cycle_id: Optional cycle identifier for filename.

        Returns:
            Path to saved file.
        """
        if not self._samples:
            logger.warning("No samples to save")
            return self.config.failed_samples_dir

        timestamp = datetime.now(UTC)
        filename = f"failed_samples_{cycle_id or timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.config.failed_samples_dir / filename

        data = {
            "metadata": {
                "saved_at": timestamp.isoformat(),
                "total_samples": len(self._samples),
                "cycle_id": cycle_id,
            },
            "samples": [s.to_dict() for s in self._samples],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self._samples)} samples to {filepath}")
        return filepath

    def load(self, filepath: Path | str) -> list[FailedSample]:
        """Load samples from file.

        Args:
            filepath: Path to samples file.

        Returns:
            List of loaded samples.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Samples file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        samples = [FailedSample.from_dict(s) for s in data["samples"]]
        logger.info(f"Loaded {len(samples)} samples from {filepath}")
        return samples

    def load_all(self) -> list[FailedSample]:
        """Load all samples from storage directory.

        Returns:
            List of all saved samples.
        """
        all_samples = []
        for filepath in self.config.failed_samples_dir.glob("failed_samples_*.json"):
            samples = self.load(filepath)
            all_samples.extend(samples)

        logger.info(f"Loaded {len(all_samples)} total samples from storage")
        return all_samples

    def get_samples(self) -> list[FailedSample]:
        """Get currently collected samples in memory.

        Returns:
            List of samples in memory.
        """
        return self._samples.copy()

    def clear(self) -> int:
        """Clear samples from memory.

        Returns:
            Number of samples cleared.
        """
        count = len(self._samples)
        self._samples = []
        logger.info(f"Cleared {count} samples from memory")
        return count

    def to_dataframe(self, include_stored: bool = False) -> pd.DataFrame:
        """Convert samples to DataFrame.

        Args:
            include_stored: Whether to include samples from storage.

        Returns:
            DataFrame with samples.
        """
        if include_stored:
            samples = self.load_all()
            samples.extend(self._samples)
        else:
            samples = self._samples

        if not samples:
            return pd.DataFrame(columns=[
                "original_text", "variant_text", "original_label",
                "predicted_label", "strategy_name", "confidence", "collected_at"
            ])

        return pd.DataFrame([s.to_dict() for s in samples])

    def get_strategy_distribution(self) -> dict[str, int]:
        """Get distribution of samples by attack strategy.

        Returns:
            Dictionary of strategy -> count.
        """
        distribution: dict[str, int] = {}
        for sample in self._samples:
            distribution[sample.strategy_name] = distribution.get(sample.strategy_name, 0) + 1
        return distribution

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about collected samples.

        Returns:
            Dictionary of statistics.
        """
        if not self._samples:
            return {
                "total_samples": 0,
                "unique_originals": 0,
                "strategy_distribution": {},
                "avg_confidence": 0.0,
            }

        unique_originals = len(set(s.original_text for s in self._samples))
        avg_confidence = sum(s.confidence for s in self._samples) / len(self._samples)

        return {
            "total_samples": len(self._samples),
            "unique_originals": unique_originals,
            "strategy_distribution": self.get_strategy_distribution(),
            "avg_confidence": avg_confidence,
        }

    def cleanup_old_files(self, keep_last: int = 10) -> int:
        """Remove old sample files, keeping only the most recent.

        Args:
            keep_last: Number of files to keep.

        Returns:
            Number of files removed.
        """
        files = sorted(
            self.config.failed_samples_dir.glob("failed_samples_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        removed = 0
        for filepath in files[keep_last:]:
            filepath.unlink()
            removed += 1

        if removed > 0:
            logger.info(f"Removed {removed} old sample files")

        return removed
