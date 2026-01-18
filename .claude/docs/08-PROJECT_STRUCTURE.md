# 프로젝트 구조

> 전체 디렉토리 구조 및 모듈별 역할

## 전체 디렉토리 구조

```
content-arena/
├── README.md
├── docker-compose.yml
├── Makefile
├── .env.example
│
├── api-service/          # Go API 서버
├── ml-service/           # Python ML 추론 서버
├── attacker/             # 공격 전략 모듈
├── defender/             # 방어 모델 모듈
├── training/             # QLoRA 학습 파이프라인
├── mlops/                # MLOps 자동화
├── infra/                # Docker Compose, 설정
├── k8s/                  # Kubernetes 매니페스트
├── data/                 # 데이터셋 (gitignore)
├── models/               # 모델 체크포인트 (gitignore)
├── scripts/              # 유틸리티 스크립트
└── .claude/docs/         # 프로젝트 문서
```

## 모듈별 역할

| 모듈 | 언어 | 역할 | 상세 문서 |
|------|------|------|-----------|
| `api-service/` | Go | HTTP API, 배틀 오케스트레이션 | `go-02-structure.md` |
| `ml-service/` | Python | 모델 추론 서버 (FastAPI) | `py-01-uv-setup.md` |
| `attacker/` | Python | 우회 패턴 생성 전략 | `04-ML_PIPELINE.md` |
| `defender/` | Python | 유해 콘텐츠 분류 모델 | `04-ML_PIPELINE.md` |
| `training/` | Python | QLoRA Fine-tuning | `py-03-transformers-peft.md` |
| `mlops/` | Python | 모델 레지스트리, 배포 | `05-MLOPS.md` |
| `infra/` | YAML | Docker Compose, Prometheus | `dev-07-monitoring.md` |
| `k8s/` | YAML | Kubernetes 매니페스트 | - |

## Go API 서버 (api-service/)

```
api-service/
├── cmd/server/main.go           # 엔트리포인트
├── internal/
│   ├── config/                  # Viper 설정
│   ├── handler/                 # HTTP 핸들러
│   ├── service/                 # 비즈니스 로직
│   ├── repository/              # 데이터 접근 (GORM)
│   ├── model/                   # 도메인 엔티티
│   ├── dto/                     # Request/Response DTO
│   ├── middleware/              # 로깅, CORS, Recovery
│   ├── client/                  # ML/Attacker 클라이언트
│   └── router/                  # Gin 라우터
├── pkg/                         # 공유 유틸리티
├── Dockerfile
└── go.mod
```

상세: `go-02-structure.md`, `go-03-config-logging.md`, `go-04-gorm-redis.md`

## Python 서비스 구조

### ML 추론 서버 (ml-service/)

```
ml-service/
├── app/
│   ├── main.py                  # FastAPI 앱
│   ├── config.py                # pydantic-settings
│   ├── api/                     # 라우터 (/classify, /health)
│   ├── services/                # 모델 로드/추론
│   └── models/                  # Pydantic 스키마
├── tests/
├── Dockerfile
└── pyproject.toml
```

### 공격자 모듈 (attacker/)

```
attacker/
├── strategies/
│   ├── base.py                  # AttackStrategy ABC
│   ├── unicode_evasion.py       # 유니코드 변형
│   ├── llm_evasion.py           # Ollama LLM 기반
│   ├── homoglyph.py             # 동형 문자
│   └── leetspeak.py             # 리트스피크
├── orchestrator.py              # 전략 조합 실행
└── prompts/                     # LLM 프롬프트 템플릿
```

### 방어자 모듈 (defender/)

```
defender/
├── model.py                     # ContentFilter 클래스
└── api.py                       # FastAPI 엔드포인트
```

### 학습 파이프라인 (training/)

```
training/
├── data_preparation.py          # 데이터셋 준비
├── qlora_trainer.py             # QLoRA 학습
├── auto_retrain.py              # 자동 재학습 트리거
├── train.py                     # CLI 엔트리포인트
└── evaluate_and_deploy.py       # 평가 & 배포
```

### MLOps 모듈 (mlops/)

```
mlops/
├── model_registry.py            # MLflow Registry
├── evaluator.py                 # Champion/Challenger 평가
├── deployer.py                  # 자동 배포
└── metrics.py                   # Prometheus 메트릭
```

상세: `py-05-mlflow.md`, `05-MLOPS.md`

## 인프라 구조 (infra/)

```
infra/
├── docker-compose.yml           # 개발 인프라
├── init.sql                     # PostgreSQL 초기화
├── prometheus.yml               # Prometheus 설정
└── grafana/
    └── provisioning/
        ├── dashboards/          # 대시보드 JSON
        └── datasources/         # Prometheus 연결
```

상세: `dev-07-monitoring.md`

## Kubernetes 구조 (k8s/)

```
k8s/
├── base/                        # 기본 매니페스트
│   ├── namespace.yaml
│   ├── api/                     # API 서버 배포
│   ├── ml/                      # ML 서버 배포 (GPU)
│   ├── postgres/                # StatefulSet
│   ├── redis/
│   └── monitoring/              # Prometheus, Grafana
│
└── overlays/                    # Kustomize 오버레이
    ├── dev/
    └── prod/
```

## 주요 설정 파일

### .env.example

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `DB_HOST` | PostgreSQL 호스트 | `localhost` |
| `REDIS_HOST` | Redis 호스트 | `localhost` |
| `MLFLOW_TRACKING_URI` | MLflow 서버 | `http://localhost:5000` |
| `OLLAMA_HOST` | Ollama 서버 | `http://localhost:11434` |
| `OLLAMA_MODEL` | 공격자 LLM | `mistral:7b-instruct-v0.2-q4_K_S` |
| `ML_SERVICE_HOST` | ML 추론 서버 | `http://localhost:8001` |

### Makefile 주요 타겟

| 타겟 | 설명 |
|------|------|
| `make setup` | 전체 환경 설정 |
| `make infra-up` | Docker 인프라 시작 |
| `make run-api` | Go API 서버 실행 |
| `make run-ml` | ML 추론 서버 실행 |
| `make train` | 모델 학습 |
| `make battle` | 배틀 실행 |
| `make test` | 전체 테스트 |

## 네이밍 컨벤션

CLAUDE.md의 개발 컨벤션 참조:
- Go: `snake_case.go`, `PascalCase` 구조체
- Python: `snake_case.py`, `PascalCase` 클래스
- 테스트: `test_*.py` (Python), `*_test.go` (Go)

## 초기 설정

`03-ENVIRONMENT_SETUP.md` 참조

```bash
# 빠른 시작
make setup
```
