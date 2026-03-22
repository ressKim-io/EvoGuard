---
description: MLflow 실험 결과 조회 및 비교
---

# MLflow Query

사용자의 요청에 따라 MLflow 실험 결과를 조회합니다.

## 사용법

MLflow가 ml-service/.venv 환경에 설치되어 있습니다.

### 실험 목록 조회
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate && mlflow experiments search
```

### 최근 실행 조회
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate && mlflow runs list --experiment-id 0
```

### 특정 메트릭으로 정렬
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate && mlflow runs list --experiment-id 0 --order-by "metrics.f1_weighted DESC"
```

### 실행 상세 조회
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate && mlflow runs describe --run-id <RUN_ID>
```

## 자주 쓰는 쿼리

사용자 요청에 맞는 MLflow CLI 명령을 실행하고 결과를 정리해서 보여주세요.
결과가 있으면 표 형태로 정리하고, 최고 성능 모델을 하이라이트하세요.

$ARGUMENTS
