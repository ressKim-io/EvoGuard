# MLOps 파이프라인

> Champion/Challenger 패턴, 자동 배포, 모니터링

## MLOps 개요

```
Data → Train → Register (MLflow) → Compare → Promote/Reject → Deploy → Monitor
                                     ↑                                    │
                                     └────────── Feedback Loop ───────────┘
```

**관련 문서**:
- `py-05-mlflow.md` - MLflow 상세
- `dev-07-monitoring.md` - Prometheus + Grafana
- `09-CI_CD.md` - GitHub Actions

## Champion/Challenger 패턴

### 개념

| 역할 | 설명 | MLflow Alias |
|------|------|--------------|
| **Champion** | 프로덕션 트래픽 100% 처리 | `champion` |
| **Challenger** | Shadow 모드 (로깅만) | `challenger` |

### 승격 조건

1. F1 Score > Champion + 0.01 (1% 개선)
2. Precision >= Champion (정밀도 유지)
3. 최소 1000건 평가 완료

### MLflow Model Registry

```python
# Champion 조회
version = client.get_model_version_by_alias(name="content-filter", alias="champion")

# Challenger 등록
mv = mlflow.register_model(f"runs:/{run_id}/model", "content-filter")
client.set_registered_model_alias(name="content-filter", alias="challenger", version=mv.version)

# Challenger → Champion 승격
client.delete_registered_model_alias(name="content-filter", alias="champion")
client.set_registered_model_alias(name="content-filter", alias="champion", version=challenger.version)
```

상세: `py-05-mlflow.md`

## 비교 평가 시스템

### 테스트셋 평가

```python
# Champion과 Challenger 동일 테스트셋으로 비교
comparison = {
    "champion": {"f1": 0.82, "precision": 0.85},
    "challenger": {"f1": 0.87, "precision": 0.86},
    "improvement": {"f1": 0.05, "precision": 0.01},
    "should_promote": True
}
```

### Shadow 평가

프로덕션 트래픽으로 Champion/Challenger 동시 추론 (결과만 로깅)

| 메트릭 | 설명 |
|--------|------|
| Agreement Rate | 두 모델 결과 일치율 |
| Champion Accuracy | Champion 정확도 |
| Challenger Accuracy | Challenger 정확도 |

## 자동 배포 파이프라인

### 배포 플로우

```
Training 완료 → Challenger 등록 → 평가 → 승격 판단 → Inference 핫 리로드 → 알림
```

### 핫 리로드

```python
# Inference 서비스 모델 재로드
await httpx.post(f"{inference_url}/reload")
```

### 전체 파이프라인

```python
async def run_full_pipeline():
    dataset = prepare_training_data()
    run_id = trainer.train(dataset["train"], dataset["test"])
    registry.register_challenger(run_id)
    result = await deployer.deploy_if_better(dataset["test"])
    return result  # {"status": "promoted"} or {"status": "rejected"}
```

## 모니터링 메트릭

### Prometheus 메트릭

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `content_arena_model_version` | Gauge | 현재 모델 버전 |
| `content_arena_model_f1_score` | Gauge | 모델 F1 점수 |
| `content_arena_inference_requests_total` | Counter | 추론 요청 수 |
| `content_arena_inference_latency_seconds` | Histogram | 추론 지연 시간 |
| `content_arena_battle_detection_rate` | Gauge | 배틀 탐지율 |
| `content_arena_retrain_events_total` | Counter | 재학습 이벤트 |
| `content_arena_champion_changes_total` | Counter | Champion 교체 횟수 |

### 알림 규칙

| 알림 | 조건 | 심각도 |
|------|------|--------|
| DetectionRateDrop | < 50% (10분) | Warning |
| ModelF1ScoreDrop | < 0.7 (5분) | Critical |
| HighInferenceLatency | p99 > 0.5s (5분) | Warning |
| ChallengerStale | 24시간 이상 대기 | Info |

상세: `dev-07-monitoring.md`

## CI/CD 통합

### GitHub Actions 트리거

| 트리거 | 설명 |
|--------|------|
| `push` | ML 코드 변경 시 테스트 |
| `schedule` | 매일 새벽 2시 재학습 (선택) |
| `workflow_dispatch` | 수동 재학습 |

### 워크플로우

```yaml
jobs:
  test:       # pytest 실행
  train:      # GPU 러너에서 QLoRA 학습
  notify:     # Slack 알림
```

상세: `09-CI_CD.md`

## 파일 구조

```
mlops/
├── model_registry.py      # MLflow 모델 레지스트리
├── evaluator.py           # Champion/Challenger 평가
├── deployer.py            # 자동 배포
├── metrics.py             # Prometheus 메트릭
└── alerting.py            # 알림 시스템

training/
├── data_preparation.py    # 데이터셋 준비
├── qlora_trainer.py       # QLoRA 학습
├── auto_retrain.py        # 자동 재학습 트리거
├── train.py               # 학습 엔트리포인트
└── evaluate_and_deploy.py # 평가 & 배포

scripts/
├── cleanup_models.py         # 모델 정리 (--dry-run/--execute)
├── model_version_manager.py  # 버전 관리 (save/list/prune/restore)
└── run_continuous_coevolution.py  # 연속 공진화 학습
```

## 모델 버전 관리 (2026-01-25 추가)

### 정리 스크립트
```bash
# 삭제 대상 확인
python scripts/cleanup_models.py --dry-run

# 실제 삭제 실행
python scripts/cleanup_models.py --execute
```

### 버전 관리
```bash
# 현재 coevolution-latest를 버전으로 저장
python scripts/model_version_manager.py save --tag stable

# 저장된 버전 목록
python scripts/model_version_manager.py list

# 오래된 버전 삭제 (최근 3개만 유지)
python scripts/model_version_manager.py prune --keep 3

# 특정 버전 복원
python scripts/model_version_manager.py restore v_20260125_054518_initial
```

### 모델 레지스트리
- 위치: `ml-service/models/MODEL_REGISTRY.json`
- 프로덕션 모델, 앙상블 설정, 아카이브 정보 포함
