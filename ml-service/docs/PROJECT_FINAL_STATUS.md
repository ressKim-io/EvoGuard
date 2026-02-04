# EvoGuard ML Service - 최종 프로젝트 상태

> **작성일**: 2026-02-04
> **상태**: 개발 완료, 프로덕션 배포 완료

---

## 1. 프로젝트 요약

### 1.1 목표
한국어 혐오표현 탐지를 위한 강건한 ML 모델 개발

### 1.2 달성 성과

| 목표 | 달성 | 비고 |
|------|------|------|
| F1 Score > 0.95 | ✅ **0.9621** | 목표 초과 달성 |
| FP < 200 | ✅ **182** | 목표 달성 |
| FN < 100 | ✅ **51** | 목표 초과 달성 |
| 프로덕션 배포 | ✅ 완료 | models/production/ |

---

## 2. 최종 모델 성능

### 2.1 프로덕션 모델 (Coevolution-Latest v20260204)

| 지표 | 값 |
|------|-----|
| **F1 Score (weighted)** | 0.9621 |
| **Accuracy** | 96.25% |
| **Precision (weighted)** | 0.9630 |
| **Recall (weighted)** | 0.9625 |

### 2.2 Confusion Matrix

|  | Predicted Normal | Predicted Toxic |
|--|------------------|-----------------|
| **Actual Normal** | 1,856 (TN) | 182 (FP) |
| **Actual Toxic** | 51 (FN) | 4,118 (TP) |

### 2.3 클래스별 성능

| 클래스 | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Normal (0) | 0.97 | 0.91 | 0.94 |
| Toxic (1) | 0.96 | 0.99 | 0.97 |

---

## 3. 학습 이력

### 3.1 주요 마일스톤

| 날짜 | 작업 | 결과 |
|------|------|------|
| 2026-01-20 | Phase 2 Combined 학습 | F1: 0.9596 (다른 데이터셋) |
| 2026-02-01 | 균형 공진화 시스템 구현 | - |
| 2026-02-02 | 표준 데이터셋 구축 | korean_standard_v1 |
| 2026-02-02 | PMF 앙상블 학습 | F1: 0.8793 |
| 2026-02-04 | KcELECTRA 표준 재학습 | F1: 0.8752 (baseline) |
| **2026-02-04** | **공진화 100 사이클** | **F1: 0.9621** |

### 3.2 성능 향상 추이

```
Baseline (KcELECTRA)     : F1 0.8752, FP 307, FN 475
    ↓ 공진화 100 사이클
Coevolution-Latest       : F1 0.9621, FP 182, FN 51

개선율:
- F1: +9.9%
- FP: -40.7%
- FN: -89.3%
```

---

## 4. 기술 스택

### 4.1 모델

| 항목 | 값 |
|------|-----|
| 베이스 모델 | beomi/KcELECTRA-base-v2022 |
| 프레임워크 | PyTorch 2.0+ |
| 토크나이저 | BertTokenizer |
| 최대 길이 | 256 tokens |

### 4.2 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 10 (베이스) + 공진화 |
| Loss | FocalLoss (gamma=2.0, alpha=0.25) |
| AMP | 사용 (Mixed Precision) |

### 4.3 공진화 설정

| 항목 | 값 |
|------|-----|
| 사이클 | 100 |
| 재학습 조건 | Evasion > 8% |
| 공격자 진화 조건 | Evasion < 5% |
| HNM FN 가중치 | 2.0 |
| HNM FP 가중치 | 1.5 |

---

## 5. 데이터셋

### 5.1 표준 데이터셋: korean_standard_v1

| 분할 | 샘플 수 | 비율 |
|------|---------|------|
| Train | 38,911 | 80% |
| Valid | 5,796 | 10% |
| Test | 6,207 | 10% |
| **Total** | **50,914** | 100% |

### 5.2 라벨 분포

| 라벨 | Train | Test |
|------|-------|------|
| Normal (0) | 14,274 | 2,038 |
| Toxic (1) | 24,637 | 4,169 |

### 5.3 데이터 소스

**포함:**
- KOTOX (Korean Toxic)
- BEEP (한국어 혐오표현)
- UnSmile (여성가족부)
- curse_dataset (욕설)
- korean_hate_speech_balanced

**제외 (라벨 노이즈):**
- K-HATERS
- K-MHaS

---

## 6. 파일 구조

```
ml-service/
├── models/
│   ├── production/              # 프로덕션 모델 (F1: 0.9621)
│   ├── coevolution-latest/      # 최신 공진화 모델
│   ├── coevolution/versions/    # 공진화 버전 히스토리
│   └── pmf/                     # PMF 앙상블 모델
├── src/ml_service/
│   ├── inference/
│   │   ├── production_classifier.py
│   │   ├── ensemble_classifier.py
│   │   └── pmf_ensemble.py
│   ├── attacker/
│   │   ├── attacks/
│   │   ├── evolver/
│   │   └── korean_strategies.py
│   └── training/
│       └── standard_config.py
├── scripts/
│   ├── run_balanced_coevolution.py
│   ├── train_multi_model.py
│   └── evaluate_all_models.py
├── data/korean/
│   ├── korean_standard_v1_train.csv
│   ├── korean_standard_v1_valid.csv
│   └── korean_standard_v1_test.csv
└── docs/
    ├── COEVOLUTION_TRAINING_REPORT_20260204.md
    └── PROJECT_FINAL_STATUS.md (이 문서)
```

---

## 7. 사용 방법

### 7.1 추론

```python
from ml_service.inference.production_classifier import ProductionClassifier

classifier = ProductionClassifier()
result = classifier.predict("분석할 텍스트")
# {'label': 1, 'confidence': 0.99, 'toxic': True}
```

### 7.2 배치 처리

```python
results = classifier.predict_batch(texts, batch_size=32)
```

### 7.3 공진화 학습

```bash
python scripts/run_balanced_coevolution.py --max-cycles 100
```

---

## 8. 한계 및 향후 과제

### 8.1 현재 한계

| 한계 | 설명 |
|------|------|
| 성능 포화 | F1 0.96+ 에서 추가 개선 어려움 |
| 신조어 대응 | 새로운 은어/신조어에 대한 지속 업데이트 필요 |
| 문맥 의존성 | 문맥에 따라 같은 단어도 다르게 해석 필요 |

### 8.2 향후 가능한 개선 (미구현)

| 방법 | 예상 효과 | 난이도 |
|------|----------|--------|
| 대형 모델 | F1 +1~3% | 중간 |
| LLM 데이터 증강 | 다양성 증가 | 중간 |
| 멀티모달 | 이미지+텍스트 | 높음 |
| 온라인 학습 | 실시간 적응 | 높음 |

---

## 9. 결론

EvoGuard ML Service는 공진화 학습을 통해 **F1 0.9621**의 높은 성능을 달성했습니다.

- **베이스라인 대비 FN 89% 감소**: 혐오표현 탐지 능력 대폭 향상
- **균형 잡힌 성능**: FP와 FN 모두 낮은 수준 유지
- **프로덕션 배포 완료**: 즉시 사용 가능한 상태

현재 성능은 프로덕션 사용에 충분하며, 추가 개선은 비용 대비 효과가 점점 감소합니다.

---

*문서 작성: 2026-02-04*
