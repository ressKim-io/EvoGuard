# EvoGuard 학습 표준 설정

> **목적**: 모든 모델 학습에서 동일한 조건을 사용하여 공정한 비교가 가능하도록 함

## 1. 표준 데이터셋

### 1.1 공식 학습 데이터셋: `korean_standard_v1`

모든 학습에서 **동일한 데이터셋**을 사용해야 결과 비교가 유의미합니다.

```
data/korean/
├── korean_standard_v1_train.csv   # 학습용 (생성 필요)
├── korean_standard_v1_valid.csv   # 검증용 (생성 필요)
└── korean_standard_v1_test.csv    # 테스트용 (생성 필요)
```

### 1.2 데이터셋 구성 (권장)

| 데이터셋 | 포함 여부 | 사유 |
|----------|-----------|------|
| KOTOX | ✅ 포함 | 고품질, 기존 검증됨 |
| BEEP | ✅ 포함 | 고품질, 기존 검증됨 |
| UnSmile | ✅ 포함 | 고품질, 기존 검증됨 |
| curse_dataset | ✅ 포함 | 욕설 특화, 소량 |
| korean_hate_speech_balanced | ✅ 포함 | 균형 데이터 |
| **K-HATERS** | ❌ 제외 | 라벨 변환 노이즈, 성능 저하 원인 |
| **K-MHaS** | ❌ 제외 | 라벨 변환 노이즈, 성능 저하 원인 |

### 1.3 데이터 분할 비율

| 분할 | 비율 | 용도 |
|------|------|------|
| Train | 80% | 학습 |
| Valid | 10% | 하이퍼파라미터 튜닝, Early stopping |
| Test | 10% | 최종 성능 평가 (학습에 사용 금지) |

**중요**: Test 셋은 최종 평가에만 사용. 모델 선택/튜닝에 사용하면 안 됨.

---

## 2. 표준 하이퍼파라미터

### 2.1 기본 학습 설정

```python
STANDARD_CONFIG = {
    # 모델
    "base_model": "beomi/KcELECTRA-base-v2022",
    "max_length": 256,

    # 학습
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_clip": 1.0,

    # Loss
    "loss_function": "FocalLoss",
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,

    # 최적화
    "optimizer": "AdamW",
    "scheduler": "linear_warmup",
    "use_amp": True,  # Mixed Precision

    # 시드
    "seed": 42,
}
```

### 2.2 PMF 멀티모델 설정

PMF 앙상블의 각 모델은 동일한 하이퍼파라미터 사용:

| 모델 | pretrained | 설정 |
|------|------------|------|
| kcelectra | beomi/KcELECTRA-base-v2022 | STANDARD_CONFIG |
| klue-bert | klue/bert-base | STANDARD_CONFIG |
| koelectra-v3 | monologg/koelectra-base-v3-discriminator | STANDARD_CONFIG |

### 2.3 공진화 재학습 설정

```python
COEVOLUTION_CONFIG = {
    "retrain_epochs": 2,        # 재학습 에폭 (빠른 적응)
    "batch_size": 16,
    "learning_rate": 2e-5,
    "attack_batch_size": 150,   # 공격 샘플 수
}
```

---

## 3. 표준 평가 메트릭

### 3.1 메트릭 정의

```python
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    """표준 평가 메트릭 계산"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        # Primary metric (모델 선택 기준)
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),

        # Secondary metrics
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_binary": f1_score(y_true, y_pred, average="binary"),  # toxic class
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),

        # Confusion matrix
        "tp": int(tp),  # True Positive (독성 정확히 탐지)
        "tn": int(tn),  # True Negative (정상 정확히 분류)
        "fp": int(fp),  # False Positive (정상→독성 오분류)
        "fn": int(fn),  # False Negative (독성→정상 오분류, 가장 위험)
    }
```

### 3.2 주요 메트릭 의미

| 메트릭 | 설명 | 중요도 |
|--------|------|--------|
| **F1 Weighted** | 클래스 불균형 고려한 F1 | ⭐⭐⭐ Primary |
| **FN (False Negative)** | 독성을 놓친 경우 (가장 위험) | ⭐⭐⭐ Critical |
| **FP (False Positive)** | 정상을 독성으로 오탐 | ⭐⭐ Important |
| F1 Macro | 클래스별 동등 가중치 | ⭐ Reference |

### 3.3 모델 선택 기준

```python
def is_better_model(new_metrics, best_metrics):
    """모델 비교: F1 우선, 동점 시 FN 낮은 것 선택"""
    if new_metrics["f1_weighted"] > best_metrics["f1_weighted"]:
        return True
    if new_metrics["f1_weighted"] == best_metrics["f1_weighted"]:
        return new_metrics["fn"] < best_metrics["fn"]
    return False
```

---

## 4. 현재 스크립트별 설정 차이 (수정 필요)

### 4.1 데이터셋 불일치

| 스크립트 | 현재 데이터셋 | 수정 필요 |
|----------|--------------|-----------|
| phase1_deobfuscation.py | KOTOX only | ✅ 표준 데이터셋으로 |
| phase2_combined_data.py | KOTOX+BEEP+UnSmile+curse+khate | ⚠️ 기준 (표준화) |
| phase3_large_model.py | KOTOX+khate | ✅ 표준 데이터셋으로 |
| phase4_augmented.py | KOTOX+증강 | ✅ 표준 데이터셋으로 |
| phase5_cnn_enhanced.py | KOTOX+khate | ✅ 표준 데이터셋으로 |
| phase6_combined_korean.py | korean_combined_v2 (K-HATERS 포함) | ✅ 표준 데이터셋으로 |
| train_multi_model.py | korean_combined_v2 (K-HATERS 포함) | ✅ 표준 데이터셋으로 |
| run_*_coevolution.py | 각각 다름 | ✅ 표준 데이터셋으로 |

### 4.2 하이퍼파라미터 불일치

| 스크립트 | epochs | batch | lr | 수정 필요 |
|----------|--------|-------|------|-----------|
| phase1 | 15 | 16 | 2e-5 | ✅ epochs→10 |
| phase2 | 10 | 16 | 2e-5 | ✓ 기준 |
| phase3 | 10 | 8 | 1e-5 | ✅ batch→16, lr→2e-5 |
| phase4 | 10 | 16 | 2e-5 | ✓ OK |
| phase5 | - | - | - | ✅ 확인 필요 |
| phase6 | 5 | 32 | 2e-5 | ✅ epochs→10, batch→16 |
| PMF | 10 | 16 | 2e-5 | ✓ OK |

### 4.3 F1 계산 방식 불일치

| 스크립트 | 현재 average | 수정 필요 |
|----------|-------------|-----------|
| phase1~4, 6, PMF | "weighted" | ✓ OK |
| phase5 | (없음=binary) | ✅ "weighted"로 |

---

## 5. 표준 데이터셋 생성 스크립트

```bash
# 표준 데이터셋 생성
python scripts/create_standard_dataset.py

# 결과:
# - data/korean/korean_standard_v1_train.csv
# - data/korean/korean_standard_v1_valid.csv
# - data/korean/korean_standard_v1_test.csv
```

---

## 6. 체크리스트

### 새 학습 실행 전 확인사항

- [ ] 데이터셋: `korean_standard_v1_*.csv` 사용
- [ ] epochs: 10
- [ ] batch_size: 16
- [ ] learning_rate: 2e-5
- [ ] Loss: FocalLoss(gamma=2.0, alpha=0.25)
- [ ] F1: average="weighted"
- [ ] seed: 42
- [ ] Test 셋 미사용 확인

### 결과 기록 항목

- [ ] F1 Weighted
- [ ] F1 Macro
- [ ] FP, FN, TP, TN
- [ ] 학습 시간
- [ ] GPU 메모리 사용량

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-02 | 초기 작성 - 학습 표준 정의 |
