# ML Service - 한국어 독성 텍스트 분류기

한국어 독성/혐오 표현 탐지를 위한 ML 서비스입니다.

## 주요 기능

- 한국어 독성 텍스트 분류 (욕설, 혐오 표현, 차별 발언 등)
- 난독화된 텍스트 탐지 (ㅅㅂ, 씌발, 특수문자 조합 등)
- 맥락 의존적 혐오 표현 탐지 (지역 비하, 성차별 등)
- 앙상블 모델을 통한 높은 정확도

## 성능

| Model | F1 Score | 설명 |
|-------|----------|------|
| **앙상블 (권장)** | **0.9594** | Phase2 + Phase4 결합 |
| Phase 2 | 0.9597 | 통합 데이터 학습 |
| Phase 4 | 0.9580 | 증강 데이터 학습 |

자세한 성능 비교는 [models/TRAINING_RESULTS.md](models/TRAINING_RESULTS.md) 참조.

## 빠른 시작

### 설치

```bash
cd ml-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 사용법

```python
from ml_service.inference.ensemble_classifier import create_ensemble

# 분류기 로드
classifier = create_ensemble()

# 단일 텍스트 분류
result = classifier.predict("분류할 텍스트")
print(result)
# {'label': 0, 'label_text': 'clean', 'confidence': 0.98, 'toxic_prob': 0.02}

# 독성 여부만 확인
is_toxic = classifier.is_toxic("씨발")  # True

# 점수만 확인
score = classifier.get_toxicity_score("텍스트")  # 0.0 ~ 1.0

# 배치 처리
results = classifier.predict_batch(["텍스트1", "텍스트2", ...])
```

### CLI 사용

```bash
# 단일 텍스트
python src/ml_service/inference/ensemble_classifier.py "분류할 텍스트"

# 파일에서 읽기 (-v: 상세 출력)
python src/ml_service/inference/ensemble_classifier.py -f input.txt -v

# 임계값 조정
python src/ml_service/inference/ensemble_classifier.py -t 0.6 "텍스트"
```

## 프로젝트 구조

```
ml-service/
├── src/ml_service/
│   ├── inference/
│   │   └── ensemble_classifier.py  # 앙상블 추론 모듈 (권장)
│   └── training/
│       └── trainer.py
├── scripts/
│   ├── run_training.sh             # 학습 실행 스크립트 (SSH 끊김 방지)
│   ├── check_training.sh           # 학습 상태 확인
│   ├── error_analysis.py           # 에러 분석
│   ├── augment_data.py             # 데이터 증강
│   ├── phase1_deobfuscation.py     # Phase 1: 난독화 해제
│   ├── phase2_combined_data.py     # Phase 2: 통합 데이터
│   ├── phase3_large_model.py       # Phase 3: 대형 모델
│   └── phase4_augmented.py         # Phase 4: 증강 데이터
├── models/
│   ├── phase2-combined/            # Phase 2 모델
│   ├── phase4-augmented/           # Phase 4 모델
│   └── TRAINING_RESULTS.md         # 학습 결과 요약
├── data/korean/
│   ├── KOTOX/                      # KOTOX 데이터셋
│   ├── beep_*.tsv                  # BEEP 데이터셋
│   ├── unsmile_*.tsv               # UnSmile 데이터셋
│   └── augmented/                  # 증강 데이터
├── logs/                           # 학습 로그
└── docs/experiments/               # 실험 문서
```

## 모델 설명

### Phase 2 - Combined (단일 모델 최고 성능)
- 기반 모델: `beomi/KcELECTRA-base`
- 학습 데이터: KOTOX + BEEP + UnSmile + 욕설 데이터 (48,199건)
- F1: 0.9597

### Phase 4 - Augmented (독성 탐지 강화)
- 기반 모델: Phase 2
- 추가 데이터: 난독화 패턴 + 맥락 의존적 표현 (2,070건)
- F1: 0.9580
- 특징: False Negative 감소 (놓치는 독성 줄임)

### 앙상블 (권장)
- 구성: Phase2 (60%) + Phase4 (40%) 가중 평균
- F1: 0.9594
- 특징: 오탐과 미탐의 균형이 가장 좋음

## 학습하기

### 학습 실행 (SSH 끊김 방지)

```bash
# tmux에서 실행 (권장)
./scripts/run_training.sh phase4_augmented.py --epochs 10

# 백그라운드 실행
./scripts/run_training.sh -b phase4_augmented.py --epochs 10

# 학습 상태 확인
./scripts/check_training.sh
```

### 개별 Phase 학습

```bash
# Phase 1: 난독화 해제 학습
python scripts/phase1_deobfuscation.py --epochs 15

# Phase 2: 통합 데이터 학습
python scripts/phase2_combined_data.py --epochs 10

# Phase 3: 대형 모델 학습
python scripts/phase3_large_model.py --epochs 10

# Phase 4: 증강 데이터 학습
python scripts/phase4_augmented.py --epochs 10
```

### 에러 분석

```bash
python scripts/error_analysis.py
# 결과: logs/error_analysis.json
```

### 데이터 증강

```bash
python scripts/augment_data.py
# 결과: data/korean/augmented/augmented_toxic.tsv
```

## 데이터셋

| 데이터셋 | 크기 | 설명 |
|----------|------|------|
| KOTOX | ~11,000 | 난독화 포함 독성 데이터 |
| BEEP | ~8,000 | 한국어 혐오 표현 |
| UnSmile | ~15,000 | 다중 레이블 혐오 표현 |
| 욕설 데이터 | ~5,800 | 한국어 욕설 |
| 증강 데이터 | ~2,000 | 패턴 기반 생성 |

## 문서

- [학습 결과 요약](models/TRAINING_RESULTS.md)
- [실험 로그](docs/experiments/)
- [프로젝트 전체 문서](../.claude/docs/)

## 라이선스

Internal Use Only
