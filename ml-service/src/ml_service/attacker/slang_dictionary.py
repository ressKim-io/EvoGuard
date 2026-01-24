"""Korean Slang & Neologism Dictionary for Attack Evolution.

공격팀 진화를 위한 신조어/은어 사전.
방어팀이 수렴하면 이 사전에서 새로운 표현을 가져와 공격합니다.
"""

from __future__ import annotations
import random

# =============================================================================
# 신조어/은어 카테고리별 사전
# =============================================================================

# 가족 비하 (우회 표현)
FAMILY_SLANG = [
    "느금마", "느그엄마", "느그애비", "니애비", "니엠", "니앰",
    "에비", "에미", "느그미", "느그맘", "니미", "니미럴",
    "엄창", "애비충", "부모욕",
]

# 세대/성별 비하
DEMOGRAPHIC_SLANG = [
    "틀딱", "틀딱충", "틀딱년", "틀딱놈",
    "한남", "한남충", "한녀", "김치녀", "김치년",
    "맘충", "급식충", "페미충", "일베충",
    "보슬아치", "냄져", "쿵쾅이",
]

# 행동 비하
BEHAVIOR_SLANG = [
    "진지충", "설명충", "관심충", "어그로충",
    "노답", "답정너", "오지랖",
]

# 자해/자살 유도
HARM_SLANG = [
    "디지셈", "뒤지셈", "꺼지셈", "죽을래", "죽어라",
    "뒤져라", "뒤져버려", "걍 뒤져", "재기해", "재기하셈",
]

# 정치 관련 비하
POLITICAL_SLANG = [
    "노무현", "문재앙", "이명박근혜", "국뽕", "수꼴",
]

# 성적 비하
SEXUAL_SLANG = [
    "보빨러", "자댕이", "고자", "음란마귀",
]

# 장애 비하
DISABILITY_SLANG = [
    "장애인이냐", "애자냐", "불구냐", "정병이냐",
]

# =============================================================================
# 변형 패턴 (탐지 우회용)
# =============================================================================

# 문자 삽입 패턴
INSERTION_PATTERNS = [
    lambda w: w[0] + " " + w[1:],  # 첫글자 후 공백
    lambda w: w[:-1] + " " + w[-1],  # 마지막 글자 전 공백
    lambda w: ".".join(w),  # 점으로 분리
    lambda w: "-".join(w),  # 하이픈으로 분리
    lambda w: w[0] + "ㅋ" + w[1:],  # ㅋ 삽입
    lambda w: w + " ㅋㅋ",  # 끝에 ㅋㅋ
    lambda w: w + "ㅋㅋㅋ",  # 끝에 ㅋㅋㅋ
    lambda w: w + " ㅎㅎ",  # 끝에 ㅎㅎ
    lambda w: "아 " + w,  # 앞에 아
    lambda w: w + " 아",  # 뒤에 아
]

# 숫자/기호 치환
SUBSTITUTION_MAP = {
    "시": ["씨", "si", "ㅅi", "시1"],
    "발": ["bal", "ㅂal", "팔", "8"],
    "병": ["byung", "ㅂyung", "1병"],
    "신": ["sin", "ㅅin", "1신"],
    "새": ["sae", "ㅅae"],
    "끼": ["kki", "ㄲi"],
}


# =============================================================================
# 공격 진화 함수
# =============================================================================

def get_all_slang() -> list[str]:
    """모든 신조어/은어 반환."""
    all_slang = []
    all_slang.extend(FAMILY_SLANG)
    all_slang.extend(DEMOGRAPHIC_SLANG)
    all_slang.extend(BEHAVIOR_SLANG)
    all_slang.extend(HARM_SLANG)
    all_slang.extend(POLITICAL_SLANG)
    all_slang.extend(SEXUAL_SLANG)
    all_slang.extend(DISABILITY_SLANG)
    return all_slang


def get_slang_by_category(category: str) -> list[str]:
    """카테고리별 신조어 반환."""
    categories = {
        "family": FAMILY_SLANG,
        "demographic": DEMOGRAPHIC_SLANG,
        "behavior": BEHAVIOR_SLANG,
        "harm": HARM_SLANG,
        "political": POLITICAL_SLANG,
        "sexual": SEXUAL_SLANG,
        "disability": DISABILITY_SLANG,
    }
    return categories.get(category, [])


def generate_variants(word: str, num_variants: int = 5) -> list[str]:
    """단어의 변형 생성."""
    variants = [word]  # 원본 포함

    # 삽입 패턴 적용
    for pattern in INSERTION_PATTERNS:
        try:
            variant = pattern(word)
            if variant not in variants:
                variants.append(variant)
        except:
            pass

    # 랜덤 선택
    if len(variants) > num_variants:
        variants = [word] + random.sample(variants[1:], num_variants - 1)

    return variants


def evolve_attack_corpus(
    current_evasion_rate: float,
    blocked_strategies: list[str] | None = None,
) -> list[dict]:
    """
    공격 코퍼스 진화.

    방어팀이 강해지면 새로운 공격 표현을 생성합니다.

    Args:
        current_evasion_rate: 현재 공격 성공률
        blocked_strategies: 차단된 공격 전략들

    Returns:
        새로운 공격 샘플 리스트
    """
    new_samples = []

    # evasion rate가 낮으면 더 공격적인 표현 추가
    if current_evasion_rate < 0.05:
        # 신조어 전체에서 랜덤 선택
        all_slang = get_all_slang()
        selected = random.sample(all_slang, min(20, len(all_slang)))

        for word in selected:
            variants = generate_variants(word, num_variants=3)
            for variant in variants:
                new_samples.append({
                    "text": variant,
                    "label": 1,
                    "source": "slang_evolution",
                    "original": word,
                })

    elif current_evasion_rate < 0.10:
        # 중간 수준 - 일부 카테고리만
        categories = ["family", "demographic", "harm"]
        for cat in categories:
            slang = get_slang_by_category(cat)
            selected = random.sample(slang, min(5, len(slang)))
            for word in selected:
                new_samples.append({
                    "text": word,
                    "label": 1,
                    "source": f"slang_{cat}",
                })

    return new_samples


# =============================================================================
# 사전 업데이트 함수 (미래 확장용)
# =============================================================================

def add_new_slang(word: str, category: str) -> bool:
    """새로운 신조어 추가 (런타임)."""
    categories = {
        "family": FAMILY_SLANG,
        "demographic": DEMOGRAPHIC_SLANG,
        "behavior": BEHAVIOR_SLANG,
        "harm": HARM_SLANG,
        "political": POLITICAL_SLANG,
        "sexual": SEXUAL_SLANG,
        "disability": DISABILITY_SLANG,
    }

    if category in categories and word not in categories[category]:
        categories[category].append(word)
        return True
    return False


def discover_new_slang_from_failures(failed_samples: list) -> list[str]:
    """
    실패한 샘플에서 새로운 패턴 발견.

    방어팀이 탐지하지 못한 표현에서 새로운 패턴을 학습합니다.
    """
    new_patterns = []

    for sample in failed_samples:
        text = sample.get("variant_text", "") or sample.get("text", "")

        # 짧은 단어이면서 탐지 실패한 경우 → 새로운 은어 가능성
        if len(text) <= 5 and text not in get_all_slang():
            new_patterns.append(text)

    return list(set(new_patterns))
