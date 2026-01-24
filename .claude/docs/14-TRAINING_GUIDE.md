# EvoGuard 학습 가이드

> Claude Code가 학습을 시작할 때 참조하는 문서

## 빠른 시작

```bash
cd /home/resshome/project/EvoGuard/ml-service

# 연속 공진화 학습 (권장)
source .venv/bin/activate
nohup python scripts/run_continuous_coevolution.py --max-cycles 100 > logs/coevolution_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 학습 스크립트 선택 가이드

| 상황 | 스크립트 | 명령어 |
|------|----------|--------|
| **일반 학습** | `run_continuous_coevolution.py` | `--max-cycles 100` |
| **목표 달성까지** | `run_continuous_coevolution.py` | `--target-evasion 0.03` |
| **무한 실행** | `run_continuous_coevolution.py` | (옵션 없음, Ctrl+C로 중단) |
| **시간 제한** | `run_optimized_coevolution.py` | `--hours 4` |
| **단계별 학습** | `phase1~5_*.py` | 개별 실행 |

---

## 현재 최고 성능 모델

| 모델 | F1 Score | 경로 |
|------|----------|------|
| **앙상블 (P2+P4)** | **0.9594** | `ensemble_classifier.py` |
| Phase 2 Combined | 0.9597 | `models/phase2-combined/` |
| Phase 4 Augmented | 0.9580 | `models/phase4-augmented/` |

---

## 공진화 시스템 작동 원리

```
┌─────────────────────────────────────────────────────────┐
│                    CONTINUOUS LOOP                       │
├─────────────────────────────────────────────────────────┤
│  1. ATTACK: 공격 전략으로 모델 테스트                      │
│     ↓                                                    │
│  2. TRIGGER CHECK:                                       │
│     - Evasion > 8%? → RETRAIN                           │
│     - 50개 샘플 축적? → RETRAIN                          │
│     - 3 사이클 경과? → RETRAIN                           │
│     ↓                                                    │
│  3. RETRAIN: GPU로 모델 재학습 (AMP 적용)                 │
│     ↓                                                    │
│  4. CONVERGENCE CHECK:                                   │
│     - Evasion < 3% 연속 10회? → 공격 강화                │
│     ↓                                                    │
│  [반복]                                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 실행 전 체크리스트

### 1. GPU 상태 확인
```bash
nvidia-smi
# memory.used가 200MB 이하인지 확인
# 다른 프로세스가 GPU 사용 중이면 종료
```

### 2. 기존 학습 프로세스 확인
```bash
ps aux | grep coevolution
# 실행 중인 프로세스 있으면:
pkill -9 -f coevolution
```

### 3. 가상환경 활성화
```bash
cd /home/resshome/project/EvoGuard/ml-service
source .venv/bin/activate
```

---

## 모니터링 명령어

```bash
# 실시간 로그
tail -f logs/coevolution_*.log

# GPU 상태 (5초마다)
watch -n 5 nvidia-smi

# 진행 요약
grep "\[Cycle" logs/coevolution_*.log | tail -20

# Evasion rate 추이
grep "evasion=" logs/coevolution_*.log | tail -20
```

---

## 주요 설정값 (run_continuous_coevolution.py)

```python
# 트리거 조건
retrain_evasion_threshold = 0.08   # 8% 초과시 재학습
retrain_sample_threshold = 50      # 50개 샘플 축적시 재학습
retrain_cycle_interval = 3         # 3 사이클마다 강제 재학습

# 수렴 감지
convergence_threshold = 0.03       # 3% 미만이면 수렴
convergence_patience = 10          # 10회 연속 수렴시 공격 강화

# 학습 설정 (RTX 4060 Ti 8GB)
batch_size = 16
retrain_epochs = 2
use_amp = True                     # Mixed Precision

# 공격 설정
attack_batch_size = 150
attack_variants = 15
max_attack_variants = 40
```

---

## 예상 시간

| 사이클 수 | 예상 시간 | 재학습 횟수 |
|----------|----------|------------|
| 10 | ~5분 | ~10 |
| 50 | ~25분 | ~50 |
| 100 | ~50분 | ~100 |

> 사이클당 약 30초 (공격 2초 + 재학습 28초)

---

## 결과 확인

### 히스토리 파일
```bash
cat data/korean/coevolution_continuous_history.json | python -m json.tool | tail -50
```

### 학습 결과 요약
```bash
# 시작/끝 evasion rate
head -5 data/korean/coevolution_continuous_history.json
tail -5 data/korean/coevolution_continuous_history.json
```

---

## 트러블슈팅

### CUDA Out of Memory
```bash
# 1. 기존 프로세스 종료
pkill -9 -f python

# 2. GPU 메모리 확인
nvidia-smi

# 3. 배치 사이즈 줄여서 실행
# config에서 batch_size = 8로 수정
```

### 학습이 진행되지 않음
```bash
# 로그 확인
tail -100 logs/coevolution_*.log | grep -E "(ERROR|RETRAIN|Cycle)"

# Evasion rate 확인 - 너무 낮으면 수렴 상태
grep "evasion=" logs/coevolution_*.log | tail -10
```

### 프로세스 강제 종료
```bash
pkill -9 -f coevolution
sleep 3
nvidia-smi  # GPU 메모리 해제 확인
```

---

## 학습 완료 후

1. **결과 확인**
   ```bash
   tail -50 logs/coevolution_*.log
   ```

2. **모델 테스트**
   ```bash
   python -c "
   from ml_service.inference.ensemble_classifier import create_ensemble
   clf = create_ensemble()
   print(clf.predict('테스트 문장'))
   "
   ```

3. **TRAINING_RESULTS.md 업데이트**
   - 새로운 성능 기록 추가
   - 최고 성능 모델 경로 업데이트

---

## 파일 구조

```
ml-service/
├── scripts/
│   ├── run_continuous_coevolution.py  # 연속 공진화 (권장)
│   ├── run_optimized_coevolution.py   # 시간 제한 공진화
│   ├── phase1_deobfuscation.py        # Phase 1
│   ├── phase2_combined_data.py        # Phase 2 (최고 성능)
│   ├── phase3_large_model.py          # Phase 3
│   ├── phase4_augmented.py            # Phase 4
│   └── phase5_cnn_enhanced.py         # Phase 5
├── models/
│   ├── phase2-combined/               # 현재 최고 성능
│   ├── phase4-augmented/
│   └── coevolution-optimized/         # 공진화 결과
├── logs/
│   └── coevolution_*.log              # 학습 로그
└── data/korean/
    └── coevolution_continuous_history.json  # 히스토리
```
