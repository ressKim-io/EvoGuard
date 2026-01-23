#!/usr/bin/env python3
"""Data augmentation for improving Korean toxic classifier.

1. KOTOX-style obfuscation augmentation
2. Context-dependent hate speech patterns
"""

import json
import random
import re
from pathlib import Path
from typing import List, Tuple

# 난독화 변환 맵
CHAR_SUBSTITUTIONS = {
    # 자음 대체
    'ㄱ': ['ㅋ', '7', 'ㄲ', '기역', '匸'],
    'ㄴ': ['ㄵ', 'ㄶ', '니은', 'L', 'И'],
    'ㄷ': ['ㄸ', 'ㅌ', '디귿', 'C'],
    'ㄹ': ['ㄺ', '리을', '2'],
    'ㅁ': ['ㅂ', '미음', '口', 'ロ'],
    'ㅂ': ['ㅃ', 'ㅍ', '비읍', '8'],
    'ㅅ': ['ㅆ', '시옷', '人', 'ㄳ'],
    'ㅇ': ['ㅎ', '이응', '0', 'O', '○'],
    'ㅈ': ['ㅉ', 'ㅊ', '지읒', '즤'],
    'ㅊ': ['ㅉ', 'ㅈ', '치읓'],
    'ㅋ': ['ㄱ', 'ㅌ', '키읔', 'ㅋㅋ'],
    'ㅌ': ['ㄷ', 'ㅋ', '티읕'],
    'ㅍ': ['ㅂ', '피읖'],
    'ㅎ': ['ㅇ', '히읗', 'ㅗ'],

    # 모음 대체
    'ㅏ': ['ㅑ', 'ㅓ', 'ㅐ', '아', 'ㄱ'],
    'ㅓ': ['ㅕ', 'ㅏ', 'ㅔ', '어'],
    'ㅗ': ['ㅛ', 'ㅜ', '오', '0'],
    'ㅜ': ['ㅠ', 'ㅗ', '우', 'ㅡ'],
    'ㅡ': ['ㅜ', '으', '-', '一'],
    'ㅣ': ['ㅢ', '이', '1', 'l', 'I', '|'],
    'ㅐ': ['ㅔ', 'ㅏ', '애'],
    'ㅔ': ['ㅐ', 'ㅓ', '에'],

    # 숫자/특수문자
    '1': ['일', 'l', 'I', '|', 'ㅣ'],
    '2': ['이', '둘', 'ㄹ'],
    '3': ['삼', '셋', '㉢'],

    # 욕설 핵심 글자
    '씨': ['씌', '시', 'ㅆ', 'ㅅㅣ', 'c'],
    '발': ['빨', '바', '벌', '팔'],
    '새': ['세', '쌔', 'ㅅㅐ'],
    '끼': ['키', '기', 'ㄲㅣ'],
    '년': ['연', '뇬', 'ㄴㅕㄴ'],
    '놈': ['넘', '뇸', 'ㄴㅗㅁ'],
}

# 특수문자 삽입용
SPECIAL_CHARS = ['★', '☆', '◎', '●', '◆', '◇', '■', '□', '▲', '△', '▼', '▽',
                 '♠', '♣', '♥', '♦', '♤', '♧', '♡', '♢', '✦', '✧', '✩', '✪',
                 '《', '》', '「', '」', '『', '』', '【', '】', '〃', '∥', '¨', '´',
                 '。', '·', '‥', '…', '¸', 'ε', 'з', 'T^T', '≥ㅃ≤', 'ㅎㅅㅎ']

# 유니코드 특수 조합
UNICODE_CIRCLES = ['㈀', '㈁', '㈂', '㈃', '㈄', '㈅', '㈆', '㈇', '㈈', '㈉',
                   '㈊', '㈋', '㈌', '㈍', '㈎', '㈏', '㈐', '㈑', '㈒', '㈓',
                   '㉠', '㉡', '㉢', '㉣', '㉤', '㉥', '㉦', '㉧', '㉨', '㉩',
                   '㉪', '㉫', '㉬', '㉭', '㉮', '㉯', '㉰', '㉱', '㉲', '㉳']

# 한글 자모 분리/조합
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
           'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
            'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
            'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def decompose_hangul(char: str) -> Tuple[str, str, str]:
    """한글 문자를 초성, 중성, 종성으로 분리."""
    if '가' <= char <= '힣':
        code = ord(char) - ord('가')
        cho = code // (21 * 28)
        jung = (code % (21 * 28)) // 28
        jong = code % 28
        return CHOSUNG[cho], JUNGSUNG[jung], JONGSUNG[jong]
    return char, '', ''


def compose_hangul(cho: str, jung: str, jong: str = '') -> str:
    """초성, 중성, 종성을 한글 문자로 조합."""
    if cho in CHOSUNG and jung in JUNGSUNG:
        cho_idx = CHOSUNG.index(cho)
        jung_idx = JUNGSUNG.index(jung)
        jong_idx = JONGSUNG.index(jong) if jong in JONGSUNG else 0
        code = ord('가') + (cho_idx * 21 + jung_idx) * 28 + jong_idx
        return chr(code)
    return cho + jung + jong


def obfuscate_text(text: str, intensity: float = 0.3) -> str:
    """텍스트를 KOTOX 스타일로 난독화."""
    result = []

    for char in text:
        if random.random() > intensity:
            result.append(char)
            continue

        # 한글인 경우
        if '가' <= char <= '힣':
            cho, jung, jong = decompose_hangul(char)

            # 랜덤하게 다양한 난독화 적용
            method = random.choice(['substitute', 'decompose', 'special', 'unicode'])

            if method == 'substitute':
                # 자모 대체
                if cho in CHAR_SUBSTITUTIONS and random.random() < 0.5:
                    cho = random.choice(CHAR_SUBSTITUTIONS[cho])
                if jung in CHAR_SUBSTITUTIONS and random.random() < 0.5:
                    jung = random.choice(CHAR_SUBSTITUTIONS[jung])
                try:
                    result.append(compose_hangul(cho, jung, jong))
                except:
                    result.append(char)

            elif method == 'decompose':
                # 자모 분리
                result.append(cho + jung + jong)

            elif method == 'special':
                # 특수문자 삽입
                result.append(char)
                if random.random() < 0.3:
                    result.append(random.choice(SPECIAL_CHARS))

            elif method == 'unicode':
                # 유니코드 원문자로 대체
                result.append(random.choice(UNICODE_CIRCLES))
        else:
            # 한글이 아닌 경우
            if char in CHAR_SUBSTITUTIONS:
                result.append(random.choice(CHAR_SUBSTITUTIONS[char]))
            else:
                result.append(char)

    return ''.join(result)


def add_special_decoration(text: str) -> str:
    """KOTOX 스타일 특수문자 장식 추가."""
    decorations = [
        lambda t: f"【{t}】",
        lambda t: f"『{t}』",
        lambda t: f"《{t}》",
        lambda t: f"「{t}」",
        lambda t: f"★{t}★",
        lambda t: f"♥{t}♥",
        lambda t: f"◎{t}◎",
        lambda t: ' '.join([f"{random.choice(SPECIAL_CHARS)}{w}" for w in t.split()]),
    ]
    return random.choice(decorations)(text)


# 맥락 의존적 혐오 표현 패턴
CONTEXT_HATE_PATTERNS = {
    # 지역 비하
    "regional": [
        ("전라도", 1),
        ("경상도", 1),
        ("충청도", 1),
        ("홍어", 1),
        ("짭새", 1),
        ("앞차 최소 전라도", 1),
        ("전라디언", 1),
        ("경상디언", 1),
        ("전라도 놈", 1),
        ("경상도 놈", 1),
        ("호남 충", 1),
        ("영남 충", 1),
    ],

    # 성별 비하
    "gender": [
        ("한남", 1),
        ("한녀", 1),
        ("김치녀", 1),
        ("된장녀", 1),
        ("맘충", 1),
        ("틀딱충", 1),
        ("급식충", 1),
        ("여적여", 1),
        ("남적남", 1),
        ("여판사", 1),
        ("여시국", 1),
        ("남시국", 1),
        ("재기해", 1),
        ("자살해", 1),
    ],

    # 외국인/인종 비하
    "xenophobia": [
        ("조선족", 1),
        ("짱깨", 1),
        ("쪽바리", 1),
        ("양키", 1),
        ("흑형", 1),
        ("난민 형님들 한국에", 1),
        ("불법체류자", 1),
        ("다문화 애들", 1),
    ],

    # 우회적 폭력/위협
    "violence": [
        ("백린탄", 1),
        ("인두로 지져", 1),
        ("땅크 부릉부릉", 1),
        ("총으로 쏴", 1),
        ("칼로 찔러", 1),
        ("불태워", 1),
        ("죽여버려", 1),
        ("패버려", 1),
        ("때려죽여", 1),
        ("갈아버려", 1),
        ("박살내", 1),
    ],

    # 장애인 비하
    "disability": [
        ("장애인 새끼", 1),
        ("병신", 1),
        ("지체장애", 1),
        ("정신병자", 1),
        ("또라이", 1),
        ("미친놈", 1),
        ("정신나간", 1),
        ("머저리", 1),
        ("바보", 1),
        ("멍청이", 1),
    ],

    # 세대 비하
    "generation": [
        ("틀딱", 1),
        ("꼰대", 1),
        ("급식이", 1),
        ("MZ충", 1),
        ("586충", 1),
        ("노인네", 1),
        ("젊은것들", 1),
        ("애새끼", 1),
    ],
}

# 맥락 의존적 문장 템플릿
CONTEXT_TEMPLATES = [
    "{target} 특징",
    "{target} ㅋㅋㅋ",
    "{target} 답네",
    "{target} 수준",
    "역시 {target}",
    "{target}이라서 그런가",
    "{target} 아웃",
    "{target} 가라",
    "{target} 꺼져",
    "이게 {target}이지",
    "{target} 냄새",
    "{target} 스럽다",
    "{target} 처럼",
    "{target} 같은",
    "진짜 {target}",
    "딱 {target}",
]

# 추가 암시적 혐오 문장
IMPLICIT_HATE_SENTENCES = [
    # 지역
    ("앞차 운전 실력 보니까 전라도인듯", 1),
    ("전라도 사람들은 원래 저래", 1),
    ("경상도 특: 시끄러움", 1),
    ("충청도 사람 특징 답답함", 1),

    # 성별
    ("여자가 운전을 하면 안되는 이유", 1),
    ("남자가 다 그렇지 뭐", 1),
    ("역시 여자들은", 1),
    ("남자들이 문제야", 1),
    ("여자 셋 모이면 접시가 깨진다", 1),
    ("이래서 여자는 안돼", 1),
    ("남자가 왜 울어", 1),

    # 외국인
    ("중국인들 매너 실화냐", 1),
    ("일본 가서 살아라", 1),
    ("동남아로 꺼져", 1),
    ("외노자들이 문제", 1),
    ("불체자들 다 잡아", 1),

    # 폭력 암시
    ("걔 한번 혼내줘야 돼", 1),
    ("한대 쳐맞아야 정신차리지", 1),
    ("손봐줘야겠네", 1),
    ("가만 안둬", 1),
    ("두고보자", 1),
    ("내가 찾아갈게", 1),

    # 장애인 비하
    ("장애인 주차구역에 왜 저런애가", 1),
    ("정상인이 아닌듯", 1),
    ("머리가 안좋은가봐", 1),
    ("뇌가 없나", 1),

    # 세대
    ("요즘 애들은 버릇이 없어", 1),
    ("늙은이들이 문제야", 1),
    ("꼰대 마인드", 1),
    ("MZ세대 특징", 1),

    # 종교
    ("개독", 1),
    ("무슬림들 테러나 하고", 1),
    ("불교 중놈들", 1),

    # 직업
    ("경찰 개돼지", 1),
    ("교사 새끼들", 1),
    ("공무원 세금도둑", 1),
]


def generate_augmented_data() -> List[Tuple[str, int]]:
    """증강 데이터 생성."""
    augmented = []

    # 1. 맥락 의존적 표현 생성
    print("맥락 의존적 표현 생성 중...")
    for category, patterns in CONTEXT_HATE_PATTERNS.items():
        for target, label in patterns:
            for template in CONTEXT_TEMPLATES:
                text = template.format(target=target)
                augmented.append((text, label))

                # 난독화 버전도 추가
                augmented.append((obfuscate_text(text, 0.2), label))

    # 2. 암시적 혐오 문장 추가
    print("암시적 혐오 문장 추가 중...")
    for text, label in IMPLICIT_HATE_SENTENCES:
        augmented.append((text, label))
        # 난독화 버전
        augmented.append((obfuscate_text(text, 0.2), label))
        augmented.append((obfuscate_text(text, 0.4), label))

    # 3. 기존 욕설에 난독화 적용
    print("기존 욕설 난독화 버전 생성 중...")
    base_curses = [
        "씨발", "개새끼", "병신", "지랄", "미친놈", "꺼져", "죽어",
        "니애미", "니애비", "좆같은", "썅", "엿먹어", "닥쳐",
        "시발", "개년", "쓰레기", "븅신", "ㅅㅂ", "ㅂㅅ",
    ]

    for curse in base_curses:
        # 다양한 강도로 난독화
        for intensity in [0.2, 0.4, 0.6, 0.8]:
            augmented.append((obfuscate_text(curse, intensity), 1))

        # 특수문자 장식 버전
        augmented.append((add_special_decoration(curse), 1))

    # 4. 복합 문장 생성
    print("복합 문장 생성 중...")
    sentence_templates = [
        "야 {curse} 뭐하냐",
        "{curse} 진짜",
        "ㅋㅋ {curse}",
        "아 {curse}",
        "{curse} 새끼야",
        "이 {curse}",
        "{curse} ㅋㅋㅋ",
        "진짜 {curse}네",
    ]

    for curse in base_curses:
        for template in sentence_templates:
            text = template.format(curse=curse)
            augmented.append((text, 1))
            augmented.append((obfuscate_text(text, 0.3), 1))

    # 5. False Negative에서 발견된 패턴 기반 증강
    print("False Negative 패턴 기반 증강 중...")
    fn_patterns = [
        # 에러 분석에서 발견된 놓친 패턴들
        "인두로 지져버리면 얼마나 재밌게",
        "백린탄이 필요하다",
        "난민 형님들 한국에 꼭 들어와 주세요",
        "땅크 부릉부릉",
        "앞차 최소 전라도",
        "오 여판사네",
        "걸린 적은 없다고 해야지",
        "오뎅드립하던분들이",
    ]

    for pattern in fn_patterns:
        augmented.append((pattern, 1))
        augmented.append((obfuscate_text(pattern, 0.2), 1))
        augmented.append((obfuscate_text(pattern, 0.4), 1))

    return augmented


def main():
    print("=" * 70)
    print("DATA AUGMENTATION")
    print("=" * 70)

    # 증강 데이터 생성
    augmented_data = generate_augmented_data()

    # 중복 제거
    unique_data = list(set(augmented_data))
    print(f"\n생성된 증강 데이터: {len(augmented_data)}건")
    print(f"중복 제거 후: {len(unique_data)}건")

    # 저장
    output_dir = Path("data/korean/augmented")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "augmented_toxic.tsv"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("text\tlabel\n")
        for text, label in unique_data:
            # 탭과 개행문자 제거
            text = text.replace("\t", " ").replace("\n", " ")
            f.write(f"{text}\t{label}\n")

    print(f"\n저장 완료: {output_file}")

    # 통계
    toxic_count = sum(1 for _, label in unique_data if label == 1)
    clean_count = sum(1 for _, label in unique_data if label == 0)
    print(f"\n독성 데이터: {toxic_count}건")
    print(f"정상 데이터: {clean_count}건")

    # 샘플 출력
    print("\n" + "=" * 70)
    print("샘플 데이터 (처음 20개)")
    print("=" * 70)
    for i, (text, label) in enumerate(unique_data[:20]):
        label_str = "독성" if label == 1 else "정상"
        print(f"{i+1}. [{label_str}] {text[:60]}{'...' if len(text) > 60 else ''}")


if __name__ == "__main__":
    main()
