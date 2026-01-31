"""Hard Negative Mining for Focused Training.

어려운 샘플(hard negatives)을 수집하고 우선순위를 매겨 집중 학습을 지원합니다.

Components:
- BoundarySampleCollector: confidence 0.4~0.6 경계 샘플 수집
- ErrorSampleCollector: FP/FN 에러 샘플 수집
- HardNegativeMiner: 메인 클래스 (통합)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HardSample:
    """Hard negative sample for training."""

    text: str
    label: int
    predicted_label: int
    confidence: float
    sample_type: str  # "false_negative", "false_positive", "boundary"
    weight: float = 1.0
    source: str = ""  # 출처 (attack, evaluation 등)
    original_text: str | None = None
    strategy_name: str | None = None
    collected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "sample_type": self.sample_type,
            "weight": self.weight,
            "source": self.source,
            "original_text": self.original_text,
            "strategy_name": self.strategy_name,
            "collected_at": self.collected_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HardSample":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            label=data["label"],
            predicted_label=data.get("predicted_label", data["label"]),
            confidence=data.get("confidence", 0.5),
            sample_type=data.get("sample_type", "unknown"),
            weight=data.get("weight", 1.0),
            source=data.get("source", ""),
            original_text=data.get("original_text"),
            strategy_name=data.get("strategy_name"),
            collected_at=data.get("collected_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class HardNegativeMinerConfig:
    """Configuration for Hard Negative Miner."""

    # 경계 샘플 수집 설정
    boundary_low: float = 0.4
    boundary_high: float = 0.6

    # 가중치 설정
    false_negative_weight: float = 2.0  # 독성→정상 (가장 위험)
    false_positive_weight: float = 1.5  # 정상→독성
    boundary_weight: float = 1.0  # 경계 케이스

    # 수집 제한
    max_samples: int = 10000
    max_samples_per_type: int = 5000

    # 재학습 트리거
    retrain_threshold: int = 100  # 이 이상 샘플 쌓이면 재학습 트리거
    batch_size: int = 64  # 학습 배치 크기

    # 저장 경로
    storage_path: Path = field(default_factory=lambda: Path("data/korean/hard_samples.json"))

    # 중복 제거
    deduplicate: bool = True


class BoundarySampleCollector:
    """Collect samples with confidence between boundary_low and boundary_high.

    모델이 확신하지 못하는 경계 케이스를 수집합니다.
    """

    def __init__(
        self,
        boundary_low: float = 0.4,
        boundary_high: float = 0.6,
    ) -> None:
        self.boundary_low = boundary_low
        self.boundary_high = boundary_high
        self._collected: list[HardSample] = []

    def collect(
        self,
        text: str,
        label: int,
        predicted_label: int,
        confidence: float,
        source: str = "",
        **kwargs: Any,
    ) -> HardSample | None:
        """Collect a sample if it's in the boundary zone.

        Args:
            text: 텍스트
            label: 실제 라벨
            predicted_label: 예측 라벨
            confidence: 모델 신뢰도
            source: 샘플 출처
            **kwargs: 추가 정보

        Returns:
            HardSample if collected, None otherwise
        """
        if self.boundary_low <= confidence <= self.boundary_high:
            sample = HardSample(
                text=text,
                label=label,
                predicted_label=predicted_label,
                confidence=confidence,
                sample_type="boundary",
                weight=1.0,
                source=source,
                original_text=kwargs.get("original_text"),
                strategy_name=kwargs.get("strategy_name"),
            )
            self._collected.append(sample)
            return sample
        return None

    def get_samples(self) -> list[HardSample]:
        """Get all collected boundary samples."""
        return self._collected.copy()

    def clear(self) -> None:
        """Clear collected samples."""
        self._collected.clear()

    def __len__(self) -> int:
        return len(self._collected)


class ErrorSampleCollector:
    """Collect FP (False Positive) and FN (False Negative) samples.

    - FN (독성→정상): 가장 위험 - 혐오 표현을 놓침
    - FP (정상→독성): 덜 위험하지만 사용자 경험 저하
    """

    def __init__(
        self,
        false_negative_weight: float = 2.0,
        false_positive_weight: float = 1.5,
    ) -> None:
        self.false_negative_weight = false_negative_weight
        self.false_positive_weight = false_positive_weight
        self._false_negatives: list[HardSample] = []
        self._false_positives: list[HardSample] = []

    def collect(
        self,
        text: str,
        label: int,
        predicted_label: int,
        confidence: float,
        source: str = "",
        **kwargs: Any,
    ) -> HardSample | None:
        """Collect a sample if it's an error.

        Args:
            text: 텍스트
            label: 실제 라벨 (1=toxic, 0=clean)
            predicted_label: 예측 라벨
            confidence: 모델 신뢰도
            source: 샘플 출처
            **kwargs: 추가 정보

        Returns:
            HardSample if collected, None otherwise
        """
        if label == predicted_label:
            return None

        if label == 1 and predicted_label == 0:
            # False Negative: 독성인데 정상으로 예측
            sample = HardSample(
                text=text,
                label=label,
                predicted_label=predicted_label,
                confidence=confidence,
                sample_type="false_negative",
                weight=self.false_negative_weight,
                source=source,
                original_text=kwargs.get("original_text"),
                strategy_name=kwargs.get("strategy_name"),
            )
            self._false_negatives.append(sample)
            return sample

        elif label == 0 and predicted_label == 1:
            # False Positive: 정상인데 독성으로 예측
            sample = HardSample(
                text=text,
                label=label,
                predicted_label=predicted_label,
                confidence=confidence,
                sample_type="false_positive",
                weight=self.false_positive_weight,
                source=source,
                original_text=kwargs.get("original_text"),
                strategy_name=kwargs.get("strategy_name"),
            )
            self._false_positives.append(sample)
            return sample

        return None

    def collect_fn(
        self,
        text: str,
        confidence: float = 0.5,
        source: str = "",
        **kwargs: Any,
    ) -> HardSample:
        """Explicitly collect a False Negative sample.

        Args:
            text: 독성 텍스트 (정상으로 잘못 예측됨)
            confidence: 모델 신뢰도
            source: 샘플 출처
            **kwargs: 추가 정보

        Returns:
            HardSample
        """
        sample = HardSample(
            text=text,
            label=1,
            predicted_label=0,
            confidence=confidence,
            sample_type="false_negative",
            weight=self.false_negative_weight,
            source=source,
            original_text=kwargs.get("original_text"),
            strategy_name=kwargs.get("strategy_name"),
        )
        self._false_negatives.append(sample)
        return sample

    def collect_fp(
        self,
        text: str,
        confidence: float = 0.5,
        source: str = "",
        **kwargs: Any,
    ) -> HardSample:
        """Explicitly collect a False Positive sample.

        Args:
            text: 정상 텍스트 (독성으로 잘못 예측됨)
            confidence: 모델 신뢰도
            source: 샘플 출처
            **kwargs: 추가 정보

        Returns:
            HardSample
        """
        sample = HardSample(
            text=text,
            label=0,
            predicted_label=1,
            confidence=confidence,
            sample_type="false_positive",
            weight=self.false_positive_weight,
            source=source,
            original_text=kwargs.get("original_text"),
            strategy_name=kwargs.get("strategy_name"),
        )
        self._false_positives.append(sample)
        return sample

    def get_false_negatives(self) -> list[HardSample]:
        """Get all False Negative samples."""
        return self._false_negatives.copy()

    def get_false_positives(self) -> list[HardSample]:
        """Get all False Positive samples."""
        return self._false_positives.copy()

    def get_all_samples(self) -> list[HardSample]:
        """Get all error samples."""
        return self._false_negatives + self._false_positives

    def clear(self) -> None:
        """Clear all collected samples."""
        self._false_negatives.clear()
        self._false_positives.clear()

    def __len__(self) -> int:
        return len(self._false_negatives) + len(self._false_positives)


class HardNegativeMiner:
    """Hard Negative Mining for focused training.

    어려운 샘플을 수집하고 우선순위를 매겨 방어자 학습을 강화합니다.

    Sample Types and Weights:
    - false_negative (weight=2.0): 독성을 정상으로 예측 (가장 위험)
    - false_positive (weight=1.5): 정상을 독성으로 예측
    - boundary (weight=1.0): 경계 케이스 (confidence 0.4~0.6)
    """

    def __init__(
        self,
        config: HardNegativeMinerConfig | None = None,
    ) -> None:
        """Initialize Hard Negative Miner.

        Args:
            config: 설정 (없으면 기본값)
        """
        self.config = config or HardNegativeMinerConfig()

        # 수집기 초기화
        self._boundary_collector = BoundarySampleCollector(
            boundary_low=self.config.boundary_low,
            boundary_high=self.config.boundary_high,
        )
        self._error_collector = ErrorSampleCollector(
            false_negative_weight=self.config.false_negative_weight,
            false_positive_weight=self.config.false_positive_weight,
        )

        # 통합 저장소
        self._samples: list[HardSample] = []
        self._seen_texts: set[str] = set()

        # 통계
        self._stats = {
            "total_collected": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "boundary": 0,
            "deduplicated": 0,
        }

        # 로드
        self._load_samples()

    def _load_samples(self) -> None:
        """Load previously collected samples."""
        if self.config.storage_path.exists():
            try:
                with open(self.config.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data.get("samples", []):
                    sample = HardSample.from_dict(item)
                    self._samples.append(sample)
                    if self.config.deduplicate:
                        self._seen_texts.add(sample.text)

                self._stats = data.get("stats", self._stats)
                logger.info(f"[HNM] Loaded {len(self._samples)} samples")

            except Exception as e:
                logger.warning(f"[HNM] Failed to load samples: {e}")

    def _save_samples(self) -> None:
        """Save collected samples."""
        self.config.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "samples": [s.to_dict() for s in self._samples],
            "stats": self._stats,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        with open(self.config.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def mine_from_attack(
        self,
        attack_results: list[dict[str, Any]],
    ) -> int:
        """Mine hard samples from attack results.

        Args:
            attack_results: 공격 결과 리스트
                - variant_text: 변형된 텍스트
                - original_text: 원본 텍스트
                - original_label: 원래 라벨
                - model_prediction: 모델 예측
                - model_confidence: 모델 신뢰도
                - is_evasion: 탐지 우회 여부
                - strategy_name: 사용된 전략

        Returns:
            Number of samples collected
        """
        collected = 0

        for result in attack_results:
            text = result.get("variant_text", "")
            if not text:
                continue

            # 중복 체크
            if self.config.deduplicate and text in self._seen_texts:
                self._stats["deduplicated"] += 1
                continue

            label = result.get("original_label", 1)
            predicted = result.get("model_prediction", label)
            confidence = result.get("model_confidence", 0.5)
            is_evasion = result.get("is_evasion", False)

            kwargs = {
                "original_text": result.get("original_text"),
                "strategy_name": result.get("strategy_name"),
            }

            sample = None

            # Evasion = False Negative (독성인데 정상으로 예측)
            if is_evasion and label == 1:
                sample = self._error_collector.collect(
                    text=text,
                    label=label,
                    predicted_label=predicted,
                    confidence=confidence,
                    source="attack",
                    **kwargs,
                )
                if sample:
                    self._stats["false_negatives"] += 1

            # 경계 케이스
            elif self.config.boundary_low <= confidence <= self.config.boundary_high:
                sample = self._boundary_collector.collect(
                    text=text,
                    label=label,
                    predicted_label=predicted,
                    confidence=confidence,
                    source="attack",
                    **kwargs,
                )
                if sample:
                    self._stats["boundary"] += 1

            if sample:
                self._add_sample(sample)
                collected += 1

        if collected > 0:
            self._save_samples()
            logger.info(f"[HNM] Collected {collected} samples from attack")

        return collected

    def mine_from_evaluation(
        self,
        texts: list[str],
        labels: list[int],
        predictions: list[dict[str, Any]],
    ) -> int:
        """Mine hard samples from evaluation results.

        Args:
            texts: 텍스트 리스트
            labels: 실제 라벨 리스트
            predictions: 예측 결과 리스트 (label, confidence)

        Returns:
            Number of samples collected
        """
        collected = 0

        for text, label, pred in zip(texts, labels, predictions):
            # 중복 체크
            if self.config.deduplicate and text in self._seen_texts:
                self._stats["deduplicated"] += 1
                continue

            predicted = pred.get("label", label)
            confidence = pred.get("confidence", 0.5)

            sample = None

            # 에러 수집
            if label != predicted:
                sample = self._error_collector.collect(
                    text=text,
                    label=label,
                    predicted_label=predicted,
                    confidence=confidence,
                    source="evaluation",
                )
                if sample:
                    if sample.sample_type == "false_negative":
                        self._stats["false_negatives"] += 1
                    else:
                        self._stats["false_positives"] += 1

            # 경계 케이스
            elif self.config.boundary_low <= confidence <= self.config.boundary_high:
                sample = self._boundary_collector.collect(
                    text=text,
                    label=label,
                    predicted_label=predicted,
                    confidence=confidence,
                    source="evaluation",
                )
                if sample:
                    self._stats["boundary"] += 1

            if sample:
                self._add_sample(sample)
                collected += 1

        if collected > 0:
            self._save_samples()
            logger.info(f"[HNM] Collected {collected} samples from evaluation")

        return collected

    def _add_sample(self, sample: HardSample) -> None:
        """Add a sample to the collection."""
        # 용량 제한
        if len(self._samples) >= self.config.max_samples:
            # 가장 오래된 낮은 가중치 샘플 제거
            self._samples.sort(key=lambda s: (s.weight, s.collected_at))
            removed = self._samples.pop(0)
            self._seen_texts.discard(removed.text)

        self._samples.append(sample)
        if self.config.deduplicate:
            self._seen_texts.add(sample.text)
        self._stats["total_collected"] += 1

    def get_priority_batch(
        self,
        batch_size: int | None = None,
        include_types: list[str] | None = None,
    ) -> list[HardSample]:
        """Get a batch of high-priority samples.

        Args:
            batch_size: 배치 크기 (없으면 config 사용)
            include_types: 포함할 샘플 타입 (없으면 전체)

        Returns:
            List of HardSample sorted by priority (weight * recency)
        """
        batch_size = batch_size or self.config.batch_size

        # 타입 필터
        if include_types:
            samples = [s for s in self._samples if s.sample_type in include_types]
        else:
            samples = self._samples.copy()

        # 우선순위 정렬 (가중치 높고 최근 것 우선)
        samples.sort(key=lambda s: (-s.weight, s.collected_at), reverse=False)

        return samples[:batch_size]

    def should_trigger_retrain(self) -> bool:
        """Check if retraining should be triggered.

        Returns:
            True if accumulated samples >= threshold
        """
        return len(self._samples) >= self.config.retrain_threshold

    def get_training_data(
        self,
        max_samples: int | None = None,
        clear_after: bool = True,
    ) -> tuple[list[str], list[int], list[float]]:
        """Get training data from collected samples.

        Args:
            max_samples: 최대 샘플 수 (없으면 전체)
            clear_after: 반환 후 샘플 삭제 여부

        Returns:
            (texts, labels, weights) tuple
        """
        samples = self.get_priority_batch(
            batch_size=max_samples or len(self._samples)
        )

        texts = [s.text for s in samples]
        labels = [s.label for s in samples]
        weights = [s.weight for s in samples]

        if clear_after:
            self.clear()

        return texts, labels, weights

    def get_statistics(self) -> dict[str, Any]:
        """Get mining statistics.

        Returns:
            Statistics dictionary
        """
        # 타입별 분포
        type_counts = defaultdict(int)
        for s in self._samples:
            type_counts[s.sample_type] += 1

        # 소스별 분포
        source_counts = defaultdict(int)
        for s in self._samples:
            source_counts[s.source] += 1

        return {
            "total_samples": len(self._samples),
            "by_type": dict(type_counts),
            "by_source": dict(source_counts),
            "stats": self._stats.copy(),
            "retrain_ready": self.should_trigger_retrain(),
            "threshold": self.config.retrain_threshold,
        }

    def clear(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()
        self._seen_texts.clear()
        self._boundary_collector.clear()
        self._error_collector.clear()

        # 저장
        self._save_samples()
        logger.info("[HNM] Cleared all samples")

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return (
            f"HardNegativeMiner("
            f"samples={len(self._samples)}, "
            f"fn={self._stats['false_negatives']}, "
            f"fp={self._stats['false_positives']}, "
            f"boundary={self._stats['boundary']})"
        )


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Hard Negative Miner Demo")
    print("=" * 60)

    miner = HardNegativeMiner()
    print(f"\nInitialized: {miner}")

    # 공격 결과 시뮬레이션
    attack_results = [
        {
            "variant_text": "ㅅㅂ 뭐야",
            "original_text": "시발 뭐야",
            "original_label": 1,
            "model_prediction": 0,  # FN
            "model_confidence": 0.6,
            "is_evasion": True,
            "strategy_name": "chosung",
        },
        {
            "variant_text": "안녕하세요",
            "original_text": "안녕하세요",
            "original_label": 0,
            "model_prediction": 1,  # FP
            "model_confidence": 0.55,
            "is_evasion": False,
            "strategy_name": None,
        },
        {
            "variant_text": "뭐하냐",
            "original_text": "뭐하냐",
            "original_label": 0,
            "model_prediction": 0,
            "model_confidence": 0.45,  # Boundary
            "is_evasion": False,
            "strategy_name": None,
        },
    ]

    collected = miner.mine_from_attack(attack_results)
    print(f"\nCollected from attack: {collected}")

    # 통계
    stats = miner.get_statistics()
    print(f"\n[Statistics]")
    print(f"  Total: {stats['total_samples']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Retrain ready: {stats['retrain_ready']}")

    # 우선순위 배치
    batch = miner.get_priority_batch(batch_size=10)
    print(f"\n[Priority Batch] {len(batch)} samples")
    for s in batch:
        print(f"  {s.sample_type}: '{s.text[:20]}...' (weight={s.weight})")

    print(f"\n{miner}")
