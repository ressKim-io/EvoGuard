"""Korean MLOps Pipeline Configuration.

한국어 Adversarial MLOps 파이프라인 설정
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KoreanModelConfig:
    """Korean model configuration."""

    # 모델 선택 (추천 순서)
    # 1. beomi/KcELECTRA-base-v2022 - 댓글 특화, 가장 추천
    # 2. klue/bert-base - 범용 한국어 BERT
    # 3. monologg/koelectra-base-v3-discriminator - KoELECTRA
    model_name: str = "beomi/KcELECTRA-base-v2022"

    # 학습 설정
    num_labels: int = 2  # toxic, non-toxic
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # QLoRA 설정
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class KoreanDataConfig:
    """Korean data configuration."""

    # 데이터 경로
    train_data_path: Path = field(
        default_factory=lambda: Path("data/korean/korean_hate_speech_balanced.csv")
    )
    corpus_path: Path = field(
        default_factory=lambda: Path("data/korean/korean_hate_speech_balanced.csv")
    )

    # 데이터 분할
    test_size: float = 0.1
    eval_size: float = 0.1
    random_seed: int = 42

    # 전처리
    min_text_length: int = 3
    max_text_length: int = 500


@dataclass
class KoreanAttackConfig:
    """Korean attack configuration."""

    # 공격 설정
    num_variants: int = 15  # 텍스트당 변형 수 (증가)
    batch_size: int = 100  # 배치당 원본 텍스트 수 (증가)

    # 사용할 전략 - 기본 + 고급 전략 모두 포함
    strategies: list[str] = field(default_factory=lambda: [
        # 기본 전략
        "chosung",           # 초성 변환
        "jamo_decompose",    # 자모 분리
        "number_sub",        # 숫자 치환
        "english_phonetic",  # 영어 발음
        "space_insertion",   # 공백 삽입
        "similar_char",      # 유사 문자
        "emoji_insertion",   # 이모지 삽입
        "zero_width",        # 제로 너비 문자
        "consonant_elongation",  # 자음 늘리기
        "mixed",             # 복합 공격
        # 고급 전략
        "reverse",           # 역순 공격
        "slang",             # 신조어/줄임말
        "dialect",           # 방언 변형
        "compat_jamo",       # 호환용 자모
        "partial_mask",      # 부분 마스킹
        "unicode_variant",   # 유니코드 변형
        "context_injection", # 맥락 주입
        "leet_korean",       # 한글 리트
        "syllable_swap",     # 음절 교환
        "typo",              # 의도적 오타
        "heavy_mixed",       # 강력한 복합 공격
    ])


@dataclass
class KoreanPipelineConfig:
    """Korean MLOps pipeline configuration."""

    # 모델 설정
    model: KoreanModelConfig = field(default_factory=KoreanModelConfig)

    # 데이터 설정
    data: KoreanDataConfig = field(default_factory=KoreanDataConfig)

    # 공격 설정
    attack: KoreanAttackConfig = field(default_factory=KoreanAttackConfig)

    # 품질 임계값 (Quality Gate)
    max_evasion_rate: float = 0.15  # 15% 초과시 재학습
    min_f1_score: float = 0.85  # 85% 미만시 재학습

    # Co-evolution 설정
    defender_retrain_threshold: float = 0.30  # 30% 초과시 Defender 재학습
    attacker_evolve_threshold: float = 0.10  # 10% 미만시 Attacker 진화

    # 재학습 설정
    min_failed_samples: int = 30
    augmentation_multiplier: int = 3
    retrain_epochs: int = 2

    # 스케줄
    cycle_interval_minutes: int = 10

    # 경로
    model_output_dir: Path = field(
        default_factory=lambda: Path("models/korean-toxic-classifier")
    )
    coevolution_model_dir: Path = field(
        default_factory=lambda: Path("models/korean-coevolution-model")
    )


def get_korean_config() -> KoreanPipelineConfig:
    """Get default Korean pipeline configuration."""
    return KoreanPipelineConfig()


# 사전 정의된 설정들
KOREAN_CONFIGS = {
    "default": KoreanPipelineConfig(),
    "fast": KoreanPipelineConfig(
        model=KoreanModelConfig(
            num_epochs=1,
            batch_size=32,
        ),
        attack=KoreanAttackConfig(
            num_variants=5,
            batch_size=20,
        ),
        cycle_interval_minutes=5,
    ),
    "thorough": KoreanPipelineConfig(
        model=KoreanModelConfig(
            num_epochs=5,
            batch_size=8,
        ),
        attack=KoreanAttackConfig(
            num_variants=20,
            batch_size=100,
        ),
        cycle_interval_minutes=15,
    ),
}
