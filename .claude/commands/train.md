---
description: ML 학습 실행 및 모니터링 (파이프라인/공진화/단일 학습)
---

# 학습 실행

## 사전 점검
먼저 `/pre-check`에 해당하는 항목들을 빠르게 점검합니다:
- GPU 사용 가능 여부 (nvidia-smi)
- 기존 학습 프로세스 충돌 여부
- 디스크 공간 여유

## 학습 모드
$ARGUMENTS

인자 예시:
- `12h` 또는 `pipeline` → 12시간 파이프라인 실행
- `coevolution --max-cycles 100` → 균형 공진화
- `phase 6` → 특정 Phase만 실행
- `resume 3` → Phase 3부터 재개

## 실행 명령
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate
```

### 12시간 파이프라인
```bash
nohup python scripts/run_12h_pipeline.py --time-scale 1.0 > logs/pipeline_12h_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

### 균형 공진화
```bash
nohup python scripts/run_balanced_coevolution.py --max-cycles 100 > logs/balanced_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

### 특정 Phase만
```bash
python scripts/run_12h_pipeline.py --phase N
```

## 모니터링
학습 시작 후:
1. PID와 로그 경로 알려주기
2. `tail -f` 로 초반 몇 줄 확인 (정상 시작 여부)
3. 주의할 점: Loss가 0이 되거나, F1이 급락하면 경고

## 완료 후
- `/evaluate` 실행하여 성능 확인
- 메모리 업데이트 (학습 결과)
