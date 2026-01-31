"""Korean attacker module for adversarial text generation.

Components:
- korean_strategies: 한국어 공격 전략 구현
- slang_dictionary: 신조어/은어 사전
- adaptive_selector: UCB1 기반 전략 선택기
- failure_analyzer: 실패 패턴 분석기
- auto_generator: 자동 공격 생성기
- learning_attacker: 학습하는 공격자 (통합 오케스트레이터)
"""

from ml_service.attacker.korean_strategies import (
    KOREAN_ATTACK_STRATEGIES,
    KoreanAttackStrategy,
    get_korean_strategies,
    apply_korean_attack,
    apply_random_korean_attacks,
    decompose_syllable,
    compose_syllable,
    extract_choseong,
)
from ml_service.attacker.adaptive_selector import (
    AdaptiveStrategySelector,
    StrategyStats,
)
from ml_service.attacker.failure_analyzer import (
    FailurePatternAnalyzer,
    AttackFeatures,
    WeakSpot,
)
from ml_service.attacker.auto_generator import (
    AutoAttackGenerator,
    DiscoveredSlang,
    CombinedStrategy,
)
from ml_service.attacker.learning_attacker import (
    LearningAttacker,
    LearningAttackerConfig,
)
from ml_service.attacker.attacker_evolver import (
    AttackerEvolver,
    AttackerEvolverConfig,
    StrategyMutator,
    SlangEvolver,
)

__all__ = [
    # Korean strategies
    "KOREAN_ATTACK_STRATEGIES",
    "KoreanAttackStrategy",
    "get_korean_strategies",
    "apply_korean_attack",
    "apply_random_korean_attacks",
    "decompose_syllable",
    "compose_syllable",
    "extract_choseong",
    # Adaptive selector (UCB1)
    "AdaptiveStrategySelector",
    "StrategyStats",
    # Failure analyzer
    "FailurePatternAnalyzer",
    "AttackFeatures",
    "WeakSpot",
    # Auto generator
    "AutoAttackGenerator",
    "DiscoveredSlang",
    "CombinedStrategy",
    # Learning attacker (orchestrator)
    "LearningAttacker",
    "LearningAttackerConfig",
    # Attacker evolver
    "AttackerEvolver",
    "AttackerEvolverConfig",
    "StrategyMutator",
    "SlangEvolver",
]
