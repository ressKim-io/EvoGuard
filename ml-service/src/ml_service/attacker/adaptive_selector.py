"""Adaptive Strategy Selector using Multi-Armed Bandit (UCB1).

UCB1 알고리즘을 사용하여 공격 전략을 적응적으로 선택합니다.
성공률이 높은 전략에 가중치를 주면서도 탐색-활용 균형을 유지합니다.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from ml_service.attacker.korean_strategies import KOREAN_ATTACK_STRATEGIES


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""

    name: str
    total_attempts: int = 0
    successes: int = 0  # Evasion successes (탐지 우회)
    failures: int = 0  # Detected by model
    total_reward: float = 0.0
    last_used: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successes / self.total_attempts

    @property
    def avg_reward(self) -> float:
        """Calculate average reward."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_reward / self.total_attempts

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total_attempts": self.total_attempts,
            "successes": self.successes,
            "failures": self.failures,
            "total_reward": self.total_reward,
            "last_used": self.last_used,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyStats:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            total_attempts=data.get("total_attempts", 0),
            successes=data.get("successes", 0),
            failures=data.get("failures", 0),
            total_reward=data.get("total_reward", 0.0),
            last_used=data.get("last_used", ""),
        )


class AdaptiveStrategySelector:
    """UCB1 기반 적응형 전략 선택기.

    Multi-Armed Bandit 알고리즘을 사용하여:
    1. 성공률이 높은 전략 우선 선택
    2. 탐색(exploration)과 활용(exploitation) 균형
    3. 실시간 성능 추적 및 학습
    """

    def __init__(
        self,
        exploration_weight: float = 2.0,
        min_exploration_rate: float = 0.1,
        state_path: Path | None = None,
    ) -> None:
        """Initialize selector.

        Args:
            exploration_weight: UCB1 탐색 가중치 (c). 클수록 탐색 증가.
            min_exploration_rate: 최소 탐색률 (epsilon-greedy 하이브리드)
            state_path: 상태 저장 경로
        """
        self.exploration_weight = exploration_weight
        self.min_exploration_rate = min_exploration_rate
        self.state_path = state_path or Path("data/korean/learning_state.json")

        # 전략별 통계 초기화
        self._stats: dict[str, StrategyStats] = {}
        self._total_rounds = 0

        # 전략 목록 초기화
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize strategy statistics from available strategies."""
        for strategy in KOREAN_ATTACK_STRATEGIES:
            if strategy.name not in self._stats:
                self._stats[strategy.name] = StrategyStats(name=strategy.name)

    def _calculate_ucb1(self, stats: StrategyStats) -> float:
        """Calculate UCB1 score for a strategy.

        UCB1 = avg_reward + c * sqrt(ln(n) / n_i)

        Where:
            avg_reward: 해당 전략의 평균 보상
            c: 탐색 가중치
            n: 전체 라운드 수
            n_i: 해당 전략 시도 횟수
        """
        if stats.total_attempts == 0:
            # 한 번도 시도하지 않은 전략은 무한대 점수 (우선 탐색)
            return float("inf")

        if self._total_rounds == 0:
            return stats.avg_reward

        # UCB1 공식
        avg_reward = stats.avg_reward
        exploration_bonus = self.exploration_weight * math.sqrt(
            math.log(self._total_rounds + 1) / stats.total_attempts
        )

        return avg_reward + exploration_bonus

    def select_strategies(
        self,
        num: int = 3,
        exclude: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Select strategies using UCB1 algorithm.

        Args:
            num: 선택할 전략 수
            exclude: 제외할 전략 목록

        Returns:
            List of (strategy_name, ucb_score) tuples
        """
        exclude = exclude or []
        available = [
            name for name in self._stats.keys()
            if name not in exclude
        ]

        if not available:
            return []

        # UCB1 점수 계산
        scores = []
        for name in available:
            stats = self._stats[name]
            ucb_score = self._calculate_ucb1(stats)
            scores.append((name, ucb_score))

        # 점수순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        # epsilon-greedy 하이브리드: 일부는 랜덤 선택
        selected = []
        num_exploit = int(num * (1 - self.min_exploration_rate))
        num_explore = num - num_exploit

        # 활용 (exploitation): 상위 점수 전략
        selected.extend(scores[:num_exploit])

        # 탐색 (exploration): 랜덤 선택
        remaining = [s for s in scores[num_exploit:] if s not in selected]
        if remaining and num_explore > 0:
            random_picks = random.sample(remaining, min(num_explore, len(remaining)))
            selected.extend(random_picks)

        return selected[:num]

    def select_strategy_names(self, num: int = 3) -> list[str]:
        """Select strategy names only.

        Args:
            num: 선택할 전략 수

        Returns:
            List of strategy names
        """
        return [name for name, _ in self.select_strategies(num)]

    def update(
        self,
        strategy_name: str,
        success: bool,
        reward: float | None = None,
    ) -> None:
        """Update strategy statistics after an attack attempt.

        Args:
            strategy_name: 사용된 전략 이름
            success: 탐지 우회 성공 여부
            reward: 커스텀 보상 (없으면 success 기반 계산)
        """
        if strategy_name not in self._stats:
            self._stats[strategy_name] = StrategyStats(name=strategy_name)

        stats = self._stats[strategy_name]
        stats.total_attempts += 1

        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        # 보상 계산: 기본적으로 success=1.0, failure=0.0
        # 하지만 confidence 기반 부분 보상도 가능
        if reward is None:
            reward = 1.0 if success else 0.0

        stats.total_reward += reward
        stats.last_used = datetime.now(UTC).isoformat()

        self._total_rounds += 1

    def update_batch(self, results: list[dict]) -> None:
        """Update multiple results at once.

        Args:
            results: List of dicts with 'strategy_name', 'success', optional 'reward'
        """
        for result in results:
            self.update(
                strategy_name=result["strategy_name"],
                success=result["success"],
                reward=result.get("reward"),
            )

    def get_stats(self, strategy_name: str) -> StrategyStats | None:
        """Get statistics for a specific strategy."""
        return self._stats.get(strategy_name)

    def get_all_stats(self) -> dict[str, StrategyStats]:
        """Get all strategy statistics."""
        return self._stats.copy()

    def get_top_strategies(self, num: int = 5) -> list[tuple[str, float]]:
        """Get top performing strategies by success rate.

        Args:
            num: 반환할 전략 수

        Returns:
            List of (strategy_name, success_rate) sorted by success rate
        """
        rates = [
            (name, stats.success_rate)
            for name, stats in self._stats.items()
            if stats.total_attempts > 0
        ]
        rates.sort(key=lambda x: x[1], reverse=True)
        return rates[:num]

    def get_underexplored_strategies(self, min_attempts: int = 10) -> list[str]:
        """Get strategies that haven't been tried enough.

        Args:
            min_attempts: 최소 시도 횟수 기준

        Returns:
            List of underexplored strategy names
        """
        return [
            name
            for name, stats in self._stats.items()
            if stats.total_attempts < min_attempts
        ]

    def decay_old_stats(self, decay_factor: float = 0.95) -> None:
        """Decay old statistics to favor recent performance.

        Args:
            decay_factor: 감쇠 계수 (0-1). 1에 가까울수록 느린 감쇠.
        """
        for stats in self._stats.values():
            stats.total_reward *= decay_factor
            # 시도 횟수도 감쇠 (하지만 정수이므로 가끔씩만)
            if random.random() < (1 - decay_factor):
                if stats.total_attempts > 0:
                    stats.total_attempts = max(1, stats.total_attempts - 1)
                if stats.successes > 0:
                    stats.successes = max(0, stats.successes - 1)
                if stats.failures > 0:
                    stats.failures = max(0, stats.failures - 1)

    def save_state(self, path: Path | None = None) -> None:
        """Save selector state to file.

        Args:
            path: 저장 경로 (없으면 기본 경로 사용)
        """
        save_path = path or self.state_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "selector": {
                "exploration_weight": self.exploration_weight,
                "min_exploration_rate": self.min_exploration_rate,
                "total_rounds": self._total_rounds,
                "strategies": {
                    name: stats.to_dict()
                    for name, stats in self._stats.items()
                },
            },
            "saved_at": datetime.now(UTC).isoformat(),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self, path: Path | None = None) -> bool:
        """Load selector state from file.

        Args:
            path: 로드 경로 (없으면 기본 경로 사용)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = path or self.state_path

        if not load_path.exists():
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            selector_state = state.get("selector", {})
            self.exploration_weight = selector_state.get(
                "exploration_weight", self.exploration_weight
            )
            self.min_exploration_rate = selector_state.get(
                "min_exploration_rate", self.min_exploration_rate
            )
            self._total_rounds = selector_state.get("total_rounds", 0)

            strategies = selector_state.get("strategies", {})
            for name, data in strategies.items():
                self._stats[name] = StrategyStats.from_dict(data)

            # 새로 추가된 전략 초기화
            self._initialize_strategies()

            return True

        except Exception:
            return False

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()
        self._total_rounds = 0
        self._initialize_strategies()

    def __repr__(self) -> str:
        active = sum(1 for s in self._stats.values() if s.total_attempts > 0)
        return (
            f"AdaptiveStrategySelector("
            f"strategies={len(self._stats)}, "
            f"active={active}, "
            f"rounds={self._total_rounds})"
        )


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Adaptive Strategy Selector Demo")
    print("=" * 60)

    selector = AdaptiveStrategySelector()
    print(f"\nInitialized: {selector}")

    # 시뮬레이션: 일부 전략은 성공률이 높음
    import random

    high_success = ["chosung", "slang", "number_sub"]
    medium_success = ["space_insertion", "similar_char", "emoji_insertion"]

    for _ in range(100):
        strategies = selector.select_strategy_names(num=3)
        for strategy in strategies:
            if strategy in high_success:
                success = random.random() < 0.7
            elif strategy in medium_success:
                success = random.random() < 0.4
            else:
                success = random.random() < 0.2

            selector.update(strategy, success)

    print("\n[Top Strategies by Success Rate]")
    for name, rate in selector.get_top_strategies(10):
        stats = selector.get_stats(name)
        print(f"  {name:20s}: {rate:.1%} ({stats.total_attempts} attempts)")

    print("\n[Next Selection (UCB1)]")
    selected = selector.select_strategies(num=5)
    for name, score in selected:
        print(f"  {name:20s}: UCB={score:.3f}")

    # 상태 저장
    selector.save_state()
    print(f"\nState saved to: {selector.state_path}")
