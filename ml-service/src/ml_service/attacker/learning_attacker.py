"""Learning Attacker - Intelligent Attack Orchestrator.

3개의 컴포넌트(AdaptiveStrategySelector, FailurePatternAnalyzer, AutoAttackGenerator)를
통합하여 학습하는 공격자 시스템을 구현합니다.

KoreanAttackRunner와 동일한 인터페이스를 제공하여 기존 파이프라인과 호환됩니다.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml_service.attacker.adaptive_selector import AdaptiveStrategySelector
from ml_service.attacker.auto_generator import AutoAttackGenerator
from ml_service.attacker.failure_analyzer import FailurePatternAnalyzer
from ml_service.attacker.korean_strategies import (
    KOREAN_ATTACK_STRATEGIES,
    apply_korean_attack,
)
from ml_service.pipeline.korean_attack_runner import (
    KoreanAttackResult,
    KoreanBatchResult,
)
from ml_service.pipeline.korean_config import KoreanAttackConfig

logger = logging.getLogger(__name__)


@dataclass
class LearningAttackerConfig:
    """Configuration for Learning Attacker."""

    # 기본 공격 설정
    batch_size: int = 100
    num_variants: int = 15
    strategies: list[str] = field(default_factory=list)

    # UCB1 탐색 설정
    exploration_weight: float = 2.0
    min_exploration_rate: float = 0.1

    # 패턴 분석 설정
    analyze_every_n_batches: int = 5
    min_samples_for_analysis: int = 50

    # 자동 생성 설정
    auto_generate_slang: bool = True
    auto_generate_strategies: bool = True
    min_slang_confidence: float = 0.4

    # 진화 설정
    evolve_threshold_low: float = 0.05  # 이 미만이면 공격 강화
    evolve_threshold_high: float = 0.30  # 이 초과면 전략 재조정

    # 상태 저장
    state_path: Path = field(default_factory=lambda: Path("data/korean/learning_state.json"))
    corpus_path: Path = field(default_factory=lambda: Path("data/korean/korean_hate_speech_balanced.csv"))


class LearningAttacker:
    """학습하는 공격자 시스템.

    세 가지 핵심 컴포넌트를 통합:
    1. AdaptiveStrategySelector: UCB1 기반 전략 선택
    2. FailurePatternAnalyzer: 성공 패턴 분석
    3. AutoAttackGenerator: 새 공격 자동 생성

    KoreanAttackRunner와 동일한 인터페이스 제공.
    """

    def __init__(
        self,
        config: LearningAttackerConfig | KoreanAttackConfig | None = None,
        classifier: Any = None,
    ) -> None:
        """Initialize Learning Attacker.

        Args:
            config: 설정 (LearningAttackerConfig 또는 KoreanAttackConfig)
            classifier: 분류기 (predict 메서드 필요)
        """
        # 설정 처리
        if config is None:
            self.config = LearningAttackerConfig()
        elif isinstance(config, KoreanAttackConfig):
            # KoreanAttackConfig를 LearningAttackerConfig로 변환
            self.config = LearningAttackerConfig(
                batch_size=config.batch_size,
                num_variants=config.num_variants,
                strategies=config.strategies,
            )
        else:
            self.config = config

        self.classifier = classifier

        # 전략 목록 초기화
        if not self.config.strategies:
            self.config.strategies = [s.name for s in KOREAN_ATTACK_STRATEGIES]

        # 컴포넌트 초기화
        self._selector = AdaptiveStrategySelector(
            exploration_weight=self.config.exploration_weight,
            min_exploration_rate=self.config.min_exploration_rate,
            state_path=self.config.state_path,
        )
        self._analyzer = FailurePatternAnalyzer()
        self._generator = AutoAttackGenerator(
            min_confidence=self.config.min_slang_confidence,
        )

        # 내부 상태
        self._corpus: pd.DataFrame | None = None
        self._batch_count = 0
        self._total_attacks = 0
        self._total_evasions = 0
        self._history: list[dict] = []

        # 생성된 조합 전략
        self._combined_strategies: dict[str, Any] = {}

        # 외부 Evolver 연결 (선택적)
        self._external_evolver: Any = None

        # 상태 로드
        self.load_state()

    def set_evolver(self, evolver: Any) -> None:
        """Set external evolver for coordinated evolution.

        Args:
            evolver: AttackerEvolver instance
        """
        self._external_evolver = evolver
        # 양방향 연결
        if hasattr(evolver, "set_learning_attacker"):
            evolver.set_learning_attacker(self)

    def load_corpus(self, corpus_path: Path | None = None) -> None:
        """Load toxic corpus for attacks."""
        path = corpus_path or self.config.corpus_path

        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")

        self._corpus = pd.read_csv(path)
        self._corpus = self._corpus[self._corpus["label"] == 1]  # toxic만
        logger.info(f"Loaded {len(self._corpus)} toxic samples")

    def _get_sample_texts(self, n: int) -> list[tuple[str, int]]:
        """Get sample texts from corpus."""
        if self._corpus is None:
            self.load_corpus()

        samples = self._corpus.sample(n=min(n, len(self._corpus)))
        return [(str(row["text"]), int(row["label"])) for _, row in samples.iterrows()]

    def _select_strategies_for_text(
        self,
        num_strategies: int,
    ) -> list[str]:
        """Select strategies using UCB1.

        Args:
            num_strategies: 선택할 전략 수

        Returns:
            List of strategy names
        """
        # UCB1로 전략 선택
        selected = self._selector.select_strategy_names(num=num_strategies)

        # 조합 전략도 포함
        if self._combined_strategies:
            combined_names = list(self._combined_strategies.keys())
            if combined_names and random.random() < 0.2:  # 20% 확률로 조합 전략 사용
                selected.append(random.choice(combined_names))

        return selected[:num_strategies]

    def _generate_variants(
        self,
        text: str,
        num_variants: int,
    ) -> list[tuple[str, str]]:
        """Generate attack variants with adaptive strategy selection.

        Args:
            text: 원본 텍스트
            num_variants: 생성할 변형 수

        Returns:
            List of (strategy_name, variant_text) tuples
        """
        variants = []

        # UCB1로 전략 선택
        strategies = self._select_strategies_for_text(num_strategies=num_variants)

        for strategy_name in strategies:
            try:
                # 조합 전략인 경우
                if strategy_name in self._combined_strategies:
                    transform = self._combined_strategies[strategy_name].transform
                    variant = transform(text)
                else:
                    # 기본 전략
                    variant = apply_korean_attack(text, strategy_name)

                if variant != text:
                    variants.append((strategy_name, variant))
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")

        # 발견된 슬랭 패턴으로 추가 변형
        if self.config.auto_generate_slang:
            slang_variants = self._generator.generate_variants_with_discovered_slang(
                text, num_variants=2
            )
            variants.extend(slang_variants)

        return variants[:num_variants]

    def run_batch(
        self,
        num_samples: int | None = None,
        texts: list[tuple[str, int]] | None = None,
    ) -> KoreanBatchResult:
        """Run batch attack with learning.

        KoreanAttackRunner와 동일한 인터페이스.

        Args:
            num_samples: 공격할 샘플 수 (corpus에서)
            texts: 공격할 (text, label) 튜플 리스트

        Returns:
            KoreanBatchResult
        """
        start_time = time.perf_counter()
        self._batch_count += 1

        # 텍스트 준비
        if texts is None:
            num_samples = num_samples or self.config.batch_size
            texts = self._get_sample_texts(num_samples)

        logger.info(f"[LearningAttacker] Batch #{self._batch_count}: {len(texts)} samples")

        all_results: list[KoreanAttackResult] = []
        batch_evasions = 0

        # 공격 실행
        for original_text, original_label in texts:
            variants = self._generate_variants(
                original_text,
                num_variants=self.config.num_variants,
            )

            if not variants:
                continue

            # 예측
            variant_texts = [v[1] for v in variants]
            predictions = self.classifier.predict(variant_texts)

            for (strategy_name, variant_text), pred in zip(variants, predictions):
                is_evasion = pred["label"] != original_label

                result = KoreanAttackResult(
                    original_text=original_text,
                    variant_text=variant_text,
                    strategy_name=strategy_name,
                    original_label=original_label,
                    model_prediction=pred["label"],
                    model_confidence=pred["confidence"],
                    is_evasion=is_evasion,
                )
                all_results.append(result)

                # UCB1 업데이트
                reward = 1.0 if is_evasion else 0.0
                # 신뢰도가 낮은 경우 부분 보상
                if not is_evasion and pred["confidence"] < 0.7:
                    reward = 0.3 * (1 - pred["confidence"])

                self._selector.update(strategy_name, is_evasion, reward)

                if is_evasion:
                    batch_evasions += 1

        elapsed = time.perf_counter() - start_time

        # 결과 생성
        batch_result = KoreanBatchResult(
            attack_results=all_results,
            total_attacks=len(all_results),
            total_evasions=batch_evasions,
            elapsed_seconds=elapsed,
        )

        # 통계 업데이트
        self._total_attacks += len(all_results)
        self._total_evasions += batch_evasions

        # 학습 수행
        self._learn_from_batch(batch_result)

        # 로그
        logger.info(
            f"[LearningAttacker] Batch complete: "
            f"{batch_evasions}/{len(all_results)} evasions "
            f"({batch_result.evasion_rate:.1%}) in {elapsed:.1f}s"
        )

        return batch_result

    def _learn_from_batch(self, result: KoreanBatchResult) -> None:
        """Learn from batch results.

        Args:
            result: 배치 결과
        """
        # 히스토리 기록
        self._history.append({
            "batch": self._batch_count,
            "total_attacks": result.total_attacks,
            "evasions": result.total_evasions,
            "evasion_rate": result.evasion_rate,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # 주기적 패턴 분석
        if self._batch_count % self.config.analyze_every_n_batches == 0:
            self._analyze_patterns(result)

        # 새 슬랭 발견
        if self.config.auto_generate_slang:
            self._discover_new_slang(result)

        # 상태 저장
        self.save_state()

    def _analyze_patterns(self, result: KoreanBatchResult) -> None:
        """Analyze attack patterns.

        Args:
            result: 배치 결과
        """
        # 결과를 분석기 형식으로 변환
        analysis_data = [
            {
                "original_text": r.original_text,
                "variant_text": r.variant_text,
                "strategy_name": r.strategy_name,
                "is_evasion": r.is_evasion,
            }
            for r in result.attack_results
        ]

        if len(analysis_data) < self.config.min_samples_for_analysis:
            return

        # 분석 실행
        analysis = self._analyzer.analyze_batch(analysis_data)

        # 약점 식별
        weak_spots = self._analyzer.get_weak_spots(min_samples=5, min_success_rate=0.3)

        if weak_spots:
            logger.info(f"[LearningAttacker] Found {len(weak_spots)} weak spots")

            # 자동 전략 생성
            if self.config.auto_generate_strategies:
                weak_spot_dicts = [ws.to_dict() for ws in weak_spots]
                new_strategies = self._generator.auto_generate_strategies(
                    weak_spot_dicts, top_n=3
                )
                for strategy in new_strategies:
                    self._combined_strategies[strategy.name] = strategy
                    logger.info(f"[LearningAttacker] Created combined strategy: {strategy.name}")

    def _discover_new_slang(self, result: KoreanBatchResult) -> None:
        """Discover new slang from successful evasions.

        Args:
            result: 배치 결과
        """
        successful = [
            {
                "original_text": r.original_text,
                "variant_text": r.variant_text,
                "strategy_name": r.strategy_name,
            }
            for r in result.attack_results
            if r.is_evasion
        ]

        if not successful:
            return

        new_slang = self._generator.discover_new_slang(successful)
        if new_slang:
            logger.info(f"[LearningAttacker] Discovered {len(new_slang)} new slang patterns")

    def learn_from_results(self, result: KoreanBatchResult) -> None:
        """Explicitly learn from external results.

        외부에서 결과를 받아 학습할 때 사용.

        Args:
            result: 배치 결과
        """
        self._learn_from_batch(result)

    def evolve(self, force: bool = False) -> dict[str, Any]:
        """Evolve attack strategies based on performance.

        Args:
            force: 강제 진화 여부

        Returns:
            Evolution summary
        """
        # 외부 evolver 사용 (설정된 경우)
        if self._external_evolver is not None:
            recent = self._history[-10:] if len(self._history) >= 10 else self._history
            if recent:
                avg_evasion = sum(h["evasion_rate"] for h in recent) / len(recent)
                force_mode = "aggressive" if force else None
                return self._external_evolver.evolve(
                    evasion_rate=avg_evasion,
                    force_mode=force_mode,
                )

        summary = {
            "action": "none",
            "changes": [],
        }

        if not self._history:
            return summary

        # 최근 성능 계산
        recent = self._history[-10:] if len(self._history) >= 10 else self._history
        avg_evasion = sum(h["evasion_rate"] for h in recent) / len(recent)

        # 진화 결정
        if avg_evasion < self.config.evolve_threshold_low or force:
            # 성능이 낮음 → 공격 강화
            summary["action"] = "strengthen"

            # 1. 탐색 가중치 증가
            old_weight = self._selector.exploration_weight
            self._selector.exploration_weight = min(4.0, old_weight + 0.5)
            summary["changes"].append(
                f"exploration_weight: {old_weight:.1f} → {self._selector.exploration_weight:.1f}"
            )

            # 2. 슬랭 사전 업데이트
            added = self._generator.update_slang_dictionary()
            if added > 0:
                summary["changes"].append(f"added {added} new slang")

            # 3. 새 전략 조합 생성
            weak_spots = self._analyzer.get_weak_spots(min_samples=5, min_success_rate=0.3)
            if weak_spots:
                new_strategies = self._generator.auto_generate_strategies(
                    [ws.to_dict() for ws in weak_spots], top_n=3
                )
                for s in new_strategies:
                    self._combined_strategies[s.name] = s
                if new_strategies:
                    summary["changes"].append(f"created {len(new_strategies)} combined strategies")

            # 4. 변형 수 증가
            self.config.num_variants = min(30, self.config.num_variants + 5)
            summary["changes"].append(f"num_variants → {self.config.num_variants}")

            logger.info(f"[LearningAttacker] Evolved (strengthen): {summary['changes']}")

        elif avg_evasion > self.config.evolve_threshold_high:
            # 성능이 높음 → 전략 재조정 (과적합 방지)
            summary["action"] = "rebalance"

            # 오래된 통계 감쇠
            self._selector.decay_old_stats(decay_factor=0.9)
            summary["changes"].append("decayed old stats")

            # 탐색 가중치 감소 (활용 증가)
            old_weight = self._selector.exploration_weight
            self._selector.exploration_weight = max(1.0, old_weight - 0.3)
            summary["changes"].append(
                f"exploration_weight: {old_weight:.1f} → {self._selector.exploration_weight:.1f}"
            )

            logger.info(f"[LearningAttacker] Evolved (rebalance): {summary['changes']}")

        return summary

    def get_stats(self) -> dict[str, Any]:
        """Get learning attacker statistics.

        Returns:
            Statistics dictionary
        """
        # 전략 성능
        top_strategies = self._selector.get_top_strategies(num=10)

        # 슬랭 통계
        slang_stats = self._generator.get_discovered_slang_stats()

        # 분석 요약
        analysis_summary = self._analyzer.get_summary()

        return {
            "batch_count": self._batch_count,
            "total_attacks": self._total_attacks,
            "total_evasions": self._total_evasions,
            "overall_evasion_rate": (
                self._total_evasions / self._total_attacks
                if self._total_attacks > 0 else 0
            ),
            "top_strategies": [
                {"name": name, "success_rate": rate}
                for name, rate in top_strategies
            ],
            "combined_strategies": len(self._combined_strategies),
            "selector": str(self._selector),
            "slang_stats": slang_stats,
            "analysis_summary": analysis_summary,
        }

    def save_state(self, path: Path | None = None) -> None:
        """Save learning state.

        Args:
            path: 저장 경로
        """
        save_path = path or self.config.state_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Selector 상태 저장
        self._selector.save_state(save_path)

        # 추가 상태 저장
        state_file = save_path.with_suffix(".attacker.json")
        state = {
            "batch_count": self._batch_count,
            "total_attacks": self._total_attacks,
            "total_evasions": self._total_evasions,
            "history": self._history[-100:],  # 최근 100개만
            "combined_strategies": list(self._combined_strategies.keys()),
            "config": {
                "num_variants": self.config.num_variants,
                "exploration_weight": self._selector.exploration_weight,
            },
            "saved_at": datetime.now(UTC).isoformat(),
        }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self, path: Path | None = None) -> bool:
        """Load learning state.

        Args:
            path: 로드 경로

        Returns:
            True if loaded successfully
        """
        load_path = path or self.config.state_path

        # Selector 상태 로드
        selector_loaded = self._selector.load_state(load_path)

        # 추가 상태 로드
        state_file = load_path.with_suffix(".attacker.json")
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                self._batch_count = state.get("batch_count", 0)
                self._total_attacks = state.get("total_attacks", 0)
                self._total_evasions = state.get("total_evasions", 0)
                self._history = state.get("history", [])

                # Config 복원
                config_state = state.get("config", {})
                if "num_variants" in config_state:
                    self.config.num_variants = config_state["num_variants"]

                return True

            except Exception as e:
                logger.warning(f"Failed to load attacker state: {e}")

        return selector_loaded

    def reset(self) -> None:
        """Reset all learning state."""
        self._selector.reset()
        self._analyzer.clear_history()
        self._generator.reset()
        self._combined_strategies.clear()
        self._batch_count = 0
        self._total_attacks = 0
        self._total_evasions = 0
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"LearningAttacker("
            f"batches={self._batch_count}, "
            f"attacks={self._total_attacks}, "
            f"evasion_rate={self._total_evasions/max(1,self._total_attacks):.1%})"
        )


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Learning Attacker Demo")
    print("=" * 60)

    # 가짜 분류기
    class FakeClassifier:
        def predict(self, texts):
            results = []
            for text in texts:
                # 초성만 있거나 숫자 치환이면 탐지 실패
                is_evasion = (
                    any(c in "ㅅㅂㅈㄹㅁㅊ" for c in text) and
                    len([c for c in text if c.isalnum() and not c.isdigit()]) < 4
                ) or "8" in text

                results.append({
                    "label": 0 if is_evasion else 1,
                    "confidence": 0.6 if is_evasion else 0.9,
                })
            return results

    config = LearningAttackerConfig(
        batch_size=10,
        num_variants=5,
    )

    attacker = LearningAttacker(config=config, classifier=FakeClassifier())
    print(f"\nInitialized: {attacker}")

    # 테스트 데이터
    test_texts = [
        ("시발놈아", 1),
        ("병신같은놈", 1),
        ("꺼져 미친놈", 1),
        ("죽어라", 1),
        ("쓰레기같은", 1),
    ]

    # 배치 실행
    for i in range(3):
        result = attacker.run_batch(texts=test_texts)
        print(f"\nBatch {i+1}: {result.evasion_rate:.1%} evasion rate")

    # 진화
    print("\n[Evolve]")
    evolution = attacker.evolve(force=True)
    print(f"  Action: {evolution['action']}")
    for change in evolution['changes']:
        print(f"  - {change}")

    # 통계
    print("\n[Statistics]")
    stats = attacker.get_stats()
    print(f"  Total attacks: {stats['total_attacks']}")
    print(f"  Overall evasion rate: {stats['overall_evasion_rate']:.1%}")
    print(f"  Top strategies:")
    for s in stats['top_strategies'][:5]:
        print(f"    {s['name']}: {s['success_rate']:.1%}")
