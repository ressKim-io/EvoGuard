"""Configuration and data classes for Attacker Evolver.

AttackerEvolver를 위한 설정 및 이벤트 데이터 클래스.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AttackerEvolverConfig:
    """Configuration for Attacker Evolver."""

    # 진화 모드 임계값
    aggressive_threshold: float = 0.03  # evasion < 3%: aggressive
    normal_threshold: float = 0.05  # evasion < 5%: normal
    maintenance_threshold: float = 0.08  # evasion < 8%: maintenance

    # 진화 강도
    aggressive_new_strategies: int = 5
    normal_new_strategies: int = 2
    maintenance_new_strategies: int = 0

    # 탐색 가중치 조정
    aggressive_exploration_boost: float = 1.0
    normal_exploration_boost: float = 0.3
    maintenance_exploration_boost: float = 0.0

    # 슬랭 확장
    expand_slang_on_evolve: bool = True
    max_slang_per_evolve: int = 20

    # 변형 수 조정
    increase_variants_on_evolve: bool = True
    max_variants: int = 50

    # 저장 경로
    state_path: Path = field(default_factory=lambda: Path("data/korean/evolver_state.json"))


@dataclass
class EvolutionEvent:
    """Record of an evolution event."""

    timestamp: str
    evasion_rate: float
    mode: str
    changes: list[str]
    new_strategies: int = 0
    new_slang: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "evasion_rate": self.evasion_rate,
            "mode": self.mode,
            "changes": self.changes,
            "new_strategies": self.new_strategies,
            "new_slang": self.new_slang,
        }


__all__ = [
    'AttackerEvolverConfig',
    'EvolutionEvent',
]
