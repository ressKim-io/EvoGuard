"""Korean attacker module for adversarial text generation."""

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

__all__ = [
    "KOREAN_ATTACK_STRATEGIES",
    "KoreanAttackStrategy",
    "get_korean_strategies",
    "apply_korean_attack",
    "apply_random_korean_attacks",
    "decompose_syllable",
    "compose_syllable",
    "extract_choseong",
]
