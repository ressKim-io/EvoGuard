# EvoGuard 다음 트레이닝 계획

> 작성일: 2026-01-26

## 현재 상태 요약

| 항목 | 값 |
|------|-----|
| 최고 모델 | AND 앙상블 (Phase2 + Coevolution) |
| F1 Score | 0.9696 |
| FP (오탐) | 60 |
| FN (미탐) | 168 |
| 공진화 사이클 | 500회 완료 |
| Evasion Rate | 0.0% |

---

## 다음 목표

| 목표 | 현재 | 타겟 | 우선순위 |
|------|------|------|----------|
| F1 Score | 0.9696 | **0.98+** | 높음 |
| FN (미탐) | 168 | **<100** | 높음 |
| FP (오탐) | 60 | **<50** | 중간 |
| 추론 속도 | - | **<50ms** | 중간 |

---

## Phase 6: FN 감소 집중 학습

### 6-1. FN 에러 분석 및 데이터 증강

```bash
cd ml-service && source .venv/bin/activate

# 1. 현재 FN 케이스 상세 분석
python scripts/error_analysis.py --model ensemble --focus fn --output fn_analysis.json

# 2. FN 패턴 기반 데이터 증강
python scripts/augment_data.py --source fn_analysis.json --strategy targeted
```

**예상 FN 패턴:**
- 맥락 의존적 혐오 (단어 자체는 중립)
- 신조어/유행어 기반 비하
- 간접적 표현 (비유, 은유)
- 특정 커뮤니티 은어

### 6-2. 맥락 인식 모델 학습

| 항목 | 설명 |
|------|------|
| 베이스 | Phase 2 Combined |
| 추가 학습 | FN 케이스 + 맥락 증강 데이터 |
| 목표 | FN 168 → 100 이하 |

```bash
# 맥락 강화 학습
python scripts/phase6_context_aware.py \
  --base-model models/phase2-combined \
  --augmented-data data/fn_augmented.jsonl \
  --epochs 5
```

---

## Phase 7: 새로운 공격 전략 개발

### 7-1. 추가 공격 전략

| 전략 | 설명 | 우선순위 |
|------|------|----------|
| **GPT 기반 패러프레이징** | LLM으로 의미 유지 변형 | 높음 |
| **커뮤니티 특화 은어** | DC, 루리웹, 에펨코리아 등 | 높음 |
| **음성 변환 텍스트** | 발음 기반 변형 (예: 시1발 → 씨발) | 중간 |
| **이모지 조합** | 텍스트+이모지 혼합 | 낮음 |

### 7-2. 공격자 강화

```bash
# 새로운 공격 전략 추가 후 공진화 재시작
python scripts/run_continuous_coevolution.py \
  --max-cycles 200 \
  --attack-strategies extended \
  --base-model models/phase2-combined
```

---

## Phase 8: 앙상블 최적화

### 8-1. 3-모델 앙상블 실험

```
현재: Phase2 AND Coevolution = F1 0.9696

실험 조합:
┌──────────────┬──────────────┬──────────────┐
│   Phase2     │  Coevolution │   Phase4     │
│  (일반화)    │  (공격방어)  │  (증강학습)  │
└──────────────┴──────────────┴──────────────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
                 Voting / Weighted
```

### 8-2. 앙상블 전략 비교

| 전략 | 설명 | 예상 효과 |
|------|------|----------|
| AND (현재) | 모두 toxic → toxic | FP 최소화 |
| OR | 하나라도 toxic → toxic | FN 최소화 |
| Weighted | 가중치 투표 | 균형 |
| Stacking | 메타 모델 학습 | 최적화 |

```bash
# 앙상블 전략 실험
python scripts/ensemble_experiment.py \
  --models phase2,coevolution,phase4 \
  --strategies and,or,weighted,stacking \
  --output ensemble_results.json
```

---

## Phase 9: 모델 경량화 (배포용)

### 9-1. 지식 증류 (Knowledge Distillation)

| 항목 | Teacher | Student |
|------|---------|---------|
| 모델 | AND 앙상블 | DistilKcELECTRA |
| 파라미터 | ~220M | ~66M |
| 추론 속도 | ~100ms | ~30ms |
| 목표 F1 | 0.9696 | 0.96+ |

```bash
python scripts/distill_model.py \
  --teacher ensemble \
  --student distil-kcelectra \
  --epochs 10
```

### 9-2. 양자화 (Quantization)

```bash
# INT8 양자화
python scripts/quantize_model.py \
  --model models/phase2-combined \
  --precision int8 \
  --output models/phase2-quantized
```

---

## 실행 계획

### Week 1: FN 분석 및 증강
```bash
# Day 1-2: 에러 분석
python scripts/error_analysis.py --model ensemble --focus fn

# Day 3-5: 데이터 증강 및 Phase 6 학습
python scripts/phase6_context_aware.py
```

### Week 2: 새로운 공격 전략
```bash
# Day 1-3: 공격 전략 구현
# attacker/src/attacker/strategies/ 에 새 전략 추가

# Day 4-7: 공진화 200 사이클
python scripts/run_continuous_coevolution.py --max-cycles 200
```

### Week 3: 앙상블 및 경량화
```bash
# Day 1-3: 앙상블 실험
python scripts/ensemble_experiment.py

# Day 4-7: 지식 증류
python scripts/distill_model.py
```

---

## 예상 결과

| Phase | 예상 F1 | 예상 FP | 예상 FN |
|-------|---------|---------|---------|
| 현재 | 0.9696 | 60 | 168 |
| Phase 6 완료 | 0.975 | 65 | 100 |
| Phase 7 완료 | 0.978 | 60 | 80 |
| Phase 8 완료 | **0.98+** | **<50** | **<80** |

---

## 즉시 실행 명령어

```bash
cd /home/resshome/project/EvoGuard/ml-service
source .venv/bin/activate

# 1. FN 에러 분석 (먼저 실행)
python scripts/error_analysis.py --model phase2-combined --focus fn

# 2. 또는 바로 공진화 계속 (새 전략 없이)
python scripts/run_continuous_coevolution.py --max-cycles 100 --target-evasion 0.01
```

---

## 참고

- 현재 모델: `models/coevolution-latest/`
- 학습 로그: `logs/coevolution_v2_*.log`
- 결과 기록: `models/TRAINING_RESULTS.md`
