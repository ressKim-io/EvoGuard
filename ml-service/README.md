# EvoGuard ML Service

> **한국어 혐오표현 탐지를 위한 공격-방어 공진화(Co-Evolution) 시스템**

---

## 프로젝트 개요

EvoGuard ML Service는 적대적 공격 시뮬레이션을 통해 한국어 혐오표현 탐지 모델의 강건성을 지속적으로 향상시키는 머신러닝 파이프라인입니다.

### 핵심 성과

| 지표 | 값 | 비고 |
|------|-----|------|
| **F1 Score** | **0.9621** | 표준 테스트셋 기준 |
| **정확도** | 96.25% | 6,207 샘플 평가 |
| **False Positive** | 182 | 오탐률 2.9% |
| **False Negative** | 51 | 미탐률 1.2% |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    EvoGuard ML Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     공진화      ┌─────────────┐            │
│  │   Attacker  │ ◄────────────► │  Defender   │            │
│  │  (공격자)    │    진화/학습    │  (방어자)    │            │
│  └─────────────┘                └─────────────┘            │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌─────────────┐              ┌─────────────┐              │
│  │ 난독화 공격  │              │ HNM 학습    │              │
│  │ 전략 생성   │              │ (어려운샘플)  │              │
│  └─────────────┘              └─────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 공진화 메커니즘

1. **Attacker (공격자)**: 모델을 우회하는 난독화 공격 생성
   - 한글 자모 분리
   - 특수문자 삽입
   - 유니코드 변형, 신조어 생성

2. **Defender (방어자)**: 공격을 방어하도록 모델 재학습
   - Hard Negative Mining (어려운 샘플 집중 학습)
   - FN 가중치 2.0 (미탐 최소화)
   - FP 가중치 1.5 (오탐 감소)

3. **균형 공진화**: 양측이 함께 진화하며 성능 향상
   - Evasion > 8%: 방어자 재학습
   - Evasion < 5%: 공격자 진화
   - 5~8%: 균형 구간

---

## 빠른 시작

### 설치

```bash
cd ml-service
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 추론 (Production)

```python
from ml_service.inference.production_classifier import ProductionClassifier

classifier = ProductionClassifier()
result = classifier.predict("분석할 텍스트")
# {'label': 0, 'confidence': 0.99, 'toxic': False}
```

### 공진화 학습

```bash
python scripts/run_balanced_coevolution.py --max-cycles 100
```

---

## 모델 정보

### 프로덕션 모델

| 항목 | 값 |
|------|-----|
| **모델명** | Coevolution-Latest (v20260204) |
| **베이스 모델** | beomi/KcELECTRA-base-v2022 |
| **학습 방법** | 균형 공진화 100 사이클 |
| **저장 위치** | models/production/ |

### 성능 비교

| 모델 | F1 | FP | FN | 비고 |
|------|-----|-----|-----|------|
| **Coevolution-Latest** | **0.9621** | 182 | 51 | 프로덕션 |
| KcELECTRA (Baseline) | 0.8752 | 307 | 475 | 베이스라인 |
| PMF Meta-Learner | 0.8793 | 367 | 383 | 3모델 앙상블 |

---

## 데이터셋

### 표준 데이터셋: korean_standard_v1

| 분할 | 샘플 수 |
|------|---------|
| Train | 38,911 |
| Valid | 5,796 |
| Test | 6,207 |

---

## 프로젝트 구조

```
ml-service/
├── src/ml_service/
│   ├── inference/              # 추론 모듈
│   ├── attacker/               # 공격자 모듈
│   └── training/               # 학습 모듈
├── scripts/                    # 실행 스크립트
├── models/                     # 모델 저장소
├── data/korean/                # 데이터셋
└── docs/                       # 문서
```

---

## 완료된 작업

- [x] KcELECTRA 베이스 모델 학습
- [x] 공진화 시스템 구현
- [x] Hard Negative Mining 적용
- [x] PMF 앙상블 구현
- [x] 프로덕션 배포 시스템
- [x] 표준 데이터셋 구축

---

## 참고 문서

- [학습 보고서](docs/COEVOLUTION_TRAINING_REPORT_20260204.md)
- [학습 표준 설정](TRAINING_STANDARDS.md)

---

*Last Updated: 2026-02-04*
