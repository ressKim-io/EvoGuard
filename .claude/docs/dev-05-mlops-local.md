# MLOps 로컬 개발 환경 가이드

> Docker Compose 기반 MLOps 스택 설정

## 개요

EvoGuard는 다음 MLOps 스택을 Docker Compose로 제공합니다:

| 서비스 | 용도 | 포트 |
|--------|------|------|
| PostgreSQL | 메인 DB + MLflow 백엔드 | 5432 |
| Redis | 캐시 & 큐 | 6379 |
| MinIO | S3 호환 객체 저장소 | 9000, 9001 |
| MLflow | 실험 추적 & 모델 레지스트리 | 5000 |
| Prometheus | 메트릭 수집 | 9090 |
| Grafana | 시각화 | 3000 |

## 빠른 시작

```bash
# 1. 환경 변수 설정
cp .env.example .env

# 2. 핵심 서비스 시작
make docker-up
# 또는
cd infra && docker compose up -d

# 3. 모니터링 포함 시작 (선택)
cd infra && docker compose --profile monitoring up -d

# 4. 상태 확인
make docker-ps
```

## 서비스 접속

| 서비스 | URL | 기본 계정 |
|--------|-----|-----------|
| MLflow UI | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |

## MLflow 사용법

### Python 클라이언트 설정

```python
import mlflow
import os

# 환경 변수 설정 (또는 .env에서 로드)
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

# 실험 설정
mlflow.set_experiment("evoguard-defender")

# 실험 로깅
with mlflow.start_run():
    mlflow.log_param("learning_rate", 2e-4)
    mlflow.log_param("batch_size", 8)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pth")
```

### 모델 레지스트리

```python
# 모델 등록
mlflow.pytorch.log_model(model, "model", registered_model_name="defender")

# 모델 로드
model = mlflow.pytorch.load_model("models:/defender/Production")
```

## DVC 사용법

### 초기 설정

```bash
# DVC 설치
uv pip install dvc dvc-s3

# MinIO 자격 증명 설정
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

### 데이터 추적

```bash
# 데이터 추가
dvc add data/raw/dataset.csv

# Git에 커밋
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "data: add raw dataset"

# 원격 저장소에 푸시
dvc push
```

### 데이터 가져오기

```bash
# 원격에서 데이터 다운로드
dvc pull

# 특정 버전 체크아웃
git checkout v1.0.0
dvc checkout
```

### 파이프라인 정의

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/dataset.csv
      - src/preprocess.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
    outs:
      - models/defender/model.pth
    metrics:
      - metrics.json:
          cache: false
```

```bash
# 파이프라인 실행
dvc repro
```

## Docker Compose 구성

### 서비스 선택 실행

```bash
# 핵심 서비스만
docker compose up -d postgres redis minio mlflow

# 모니터링 포함
docker compose --profile monitoring up -d
```

### 로그 확인

```bash
# 모든 서비스 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f mlflow
```

### 데이터 초기화

```bash
# 볼륨 삭제 (데이터 초기화)
docker compose down -v
```

## 환경 변수

`.env` 파일에서 설정:

```bash
# PostgreSQL
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=evoguard

# MLflow
MLFLOW_PORT=5000
MLFLOW_DB=mlflow

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET=mlflow

# Redis
REDIS_PASSWORD=
```

## 트러블슈팅

### MLflow 연결 오류

```bash
# MinIO 버킷 확인
docker compose exec minio mc ls local/

# PostgreSQL 연결 확인
docker compose exec postgres pg_isready
```

### DVC 푸시 실패

```bash
# 자격 증명 확인
aws s3 ls --endpoint-url http://localhost:9000

# 버킷 생성 (필요시)
docker compose exec minio mc mb local/dvc
```

### 포트 충돌

```bash
# 사용 중인 포트 확인
lsof -i :5432
lsof -i :5000

# .env에서 포트 변경
DB_PORT=5433
MLFLOW_PORT=5001
```

## Best Practices

### 1. 실험 구조화
- 프로젝트별 experiment 분리
- 의미 있는 run 이름 사용
- 하이퍼파라미터 전체 로깅

### 2. 데이터 버전 관리
- raw 데이터는 절대 수정 금지
- processed 데이터는 코드로 재현 가능하게
- `dvc.lock` 항상 커밋

### 3. 모델 관리
- 의미 있는 버전 태그
- Production 스테이지 전환 전 검증
- 모델 메타데이터 상세 기록

## 참고 자료

- [MLflow Docker Setup](https://erikdao.com/machine-learning/production-ready-mlflow-setup-in-your-local-machine/)
- [DVC Get Started](https://doc.dvc.org/start)
- [MinIO with MLflow](https://blog.min.io/setting-up-a-development-machine-with-mlflow-and-minio/)

---

*관련 문서: `05-MLOPS.md`, `py-05-mlflow.md`, `dev-02-environment.md`*
