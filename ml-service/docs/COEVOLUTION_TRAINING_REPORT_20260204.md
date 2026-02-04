# 공진화 학습 결과 보고서

> **날짜**: 2026-02-04
> **모델**: Coevolution-Latest
> **데이터셋**: korean_standard_v1 (표준 데이터셋)

---

## 1. 요약

| 항목 | 값 |
|------|-----|
| **최종 F1 Score** | **0.9621** |
| **False Positive (오탐)** | 182 |
| **False Negative (미탐)** | 51 |
| **정확도** | 96.25% |
| **학습 시간** | 약 85분 (KcELECTRA 재학습 80분 + 공진화 2.1분) |

---

## 2. 학습 과정

### 2.1 Stage 1: KcELECTRA 표준 데이터셋 재학습

**목적**: 표준 데이터셋(korean_standard_v1)으로 베이스라인 모델 생성

```bash
python scripts/train_multi_model.py --models kcelectra --epochs 10 --dataset-version korean_standard_v1
```

| 설정 | 값 |
|------|-----|
| 베이스 모델 | beomi/KcELECTRA-base-v2022 |
| Train 샘플 | 38,911 |
| Valid 샘플 | 5,796 |
| Test 샘플 | 6,207 |
| Epochs | 10 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Best Epoch | 3 |

**Stage 1 결과** (Valid Set):
- F1: 0.8806
- FP: 260, FN: 439

**Stage 1 결과** (Test Set):
- F1: 0.8752
- FP: 307, FN: 475

### 2.2 Stage 2: 균형 공진화 학습

**목적**: 적대적 공격-방어를 통한 모델 강건성 향상

```bash
python scripts/run_balanced_coevolution.py --max-cycles 100 --verbose
```

| 설정 | 값 |
|------|-----|
| 최대 사이클 | 100 |
| 초기 Evasion Rate | 29.2% |
| 최종 Evasion Rate | 0~4.5% |
| 재학습 횟수 | 1회 |
| 공격자 진화 횟수 | 92회 |
| HNM 적용 횟수 | 96회 |
| 소요 시간 | 2.1분 |

**공진화 메커니즘**:
1. **AttackerEvolver**: Evasion < 5%일 때 공격자 전략 진화
   - Aggressive mode (< 3%): 새 전략 5개, 슬랭 확장
   - Normal mode (3-5%): 새 전략 2개
2. **HardNegativeMiner**: 어려운 샘플 집중 학습
   - FN 가중치: 2.0 (가장 위험한 케이스)
   - FP 가중치: 1.5
   - 경계 케이스 가중치: 1.0

---

## 3. 최종 성능

### 3.1 Test Set 평가 (korean_standard_v1_test.csv, 6,207 samples)

| 지표 | 값 |
|------|-----|
| **F1 (weighted)** | **0.9621** |
| Accuracy | 0.9625 |
| Precision (weighted) | 0.9630 |
| Recall (weighted) | 0.9625 |

### 3.2 Confusion Matrix

|  | Predicted Normal | Predicted Toxic |
|--|------------------|-----------------|
| **Actual Normal** | 1,856 (TN) | 182 (FP) |
| **Actual Toxic** | 51 (FN) | 4,118 (TP) |

### 3.3 클래스별 성능

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Normal (0) | 0.97 | 0.91 | 0.94 | 2,038 |
| Toxic (1) | 0.96 | 0.99 | 0.97 | 4,169 |

---

## 4. 성능 비교

### 4.1 Before vs After

| 지표 | Before (KcELECTRA) | After (Coevolution) | 개선율 |
|------|-------------------|---------------------|--------|
| **F1** | 0.8752 | **0.9621** | **+9.9%** |
| **FP** | 307 | 182 | -40.7% |
| **FN** | 475 | 51 | **-89.3%** |
| **Accuracy** | 0.8740 | 0.9625 | +10.1% |

### 4.2 모델 순위 (표준 데이터셋 기준)

| 순위 | 모델 | F1 Score | FP | FN |
|------|------|----------|----|----|
| **1** | **Coevolution-Latest** | **0.9621** | **182** | **51** |
| 2 | PMF Meta-Learner | 0.8793 | 367 | 383 |
| 3 | kcelectra (PMF) | 0.8752 | 307 | 475 |
| 4 | PMF Voting | 0.8730 | 284 | 514 |
| 5 | koelectra-v3 (PMF) | 0.8517 | 249 | 691 |

---

## 5. 주요 개선 사항

### 5.1 False Negative 89% 감소

가장 중요한 개선점은 **미탐(FN)의 89% 감소**입니다.

- Before: 475건의 독성 콘텐츠가 정상으로 오분류
- After: 51건으로 감소

이는 공진화 학습에서 HardNegativeMiner가 FN에 2.0의 가중치를 부여하여 집중 학습한 결과입니다.

### 5.2 False Positive 41% 감소

오탐도 크게 줄었습니다.

- Before: 307건의 정상 콘텐츠가 독성으로 오분류
- After: 182건으로 감소

### 5.3 강건성 향상

공진화를 통해 다양한 난독화 공격에 대한 방어력이 향상되었습니다:

- 한글 자모 분리 공격 방어
- 특수문자 삽입 공격 방어
- 유니코드 변형 공격 방어
- 신조어/슬랭 대응력 향상

---

## 6. 모델 정보

### 6.1 저장 위치

```
ml-service/models/coevolution-latest/
├── config.json
├── model.safetensors (487MB)
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

### 6.2 사용 방법

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/coevolution-latest"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 추론
text = "분석할 텍스트"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1).item()  # 0: Normal, 1: Toxic
```

---

## 7. 다음 단계

1. **프로덕션 배포**: coevolution-latest를 production 모델로 배포
2. **추가 공진화**: 더 많은 사이클로 성능 추가 향상 가능
3. **앙상블**: 다른 모델과 결합하여 추가 성능 향상 검토
4. **모니터링**: 실제 서비스에서 성능 모니터링 및 지속적 개선

---

## 8. 재현 방법

```bash
cd ml-service
source .venv/bin/activate

# Stage 1: 베이스 모델 학습
python scripts/train_multi_model.py --models kcelectra --epochs 10

# Stage 2: 공진화 학습
python scripts/run_balanced_coevolution.py --max-cycles 100

# 평가
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained('models/coevolution-latest')
tokenizer = AutoTokenizer.from_pretrained('models/coevolution-latest')
model.eval()

# 테스트 데이터
test_df = pd.read_csv('data/korean/korean_standard_v1_test.csv')
# ... 평가 코드
"
```

---

## 9. 로그 파일

- 학습 로그: `logs/train_kcelectra_20260204_085145.log`
- 공진화 로그: `logs/coevolution_20260204_101951.log`

---

*Report generated: 2026-02-04*
