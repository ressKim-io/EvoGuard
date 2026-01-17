# 📁 08. 프로젝트 구조

> 전체 디렉토리 구조 및 각 파일/폴더 역할 설명

---

## 🌳 전체 디렉토리 구조

```
content-arena/
├── README.md                          # 프로젝트 메인 문서
├── docker-compose.yml                 # 개발 환경 Docker Compose
├── docker-compose.prod.yml            # 프로덕션 Docker Compose
├── Makefile                           # 빌드/실행 자동화
├── .env.example                       # 환경변수 템플릿
├── .gitignore
│
├── docs/                              # 프로젝트 문서
│   ├── 01-ARCHITECTURE.md
│   ├── 02-TECH_STACK.md
│   ├── 03-ENVIRONMENT_SETUP.md
│   ├── 04-ML_PIPELINE.md
│   ├── 05-MLOPS.md
│   ├── 06-API_SPEC.md
│   ├── 07-DEVELOPMENT_ROADMAP.md
│   └── 08-PROJECT_STRUCTURE.md
│
├── api-service/                       # Go API 서버
│   ├── cmd/
│   │   └── server/
│   │       └── main.go                # 엔트리포인트
│   ├── internal/
│   │   ├── config/                    # 설정 관리
│   │   ├── handler/                   # HTTP 핸들러
│   │   ├── service/                   # 비즈니스 로직
│   │   ├── repository/                # 데이터 접근
│   │   ├── model/                     # 도메인 모델
│   │   ├── dto/                       # Request/Response DTO
│   │   ├── middleware/                # 미들웨어
│   │   └── client/                    # 외부 서비스 클라이언트
│   ├── pkg/                           # 공유 유틸리티
│   ├── Dockerfile
│   ├── go.mod
│   └── go.sum
│
├── ml-service/                        # Python ML 서버
│   ├── app/
│   │   ├── main.py                    # FastAPI 앱
│   │   ├── config.py                  # 설정
│   │   ├── models/                    # 모델 클래스
│   │   ├── services/                  # 비즈니스 로직
│   │   └── api/                       # API 라우터
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-dev.txt
│
├── attacker/                          # 공격자 모듈 (Python)
│   ├── strategies/                    # 공격 전략들
│   │   ├── __init__.py
│   │   ├── base.py                    # 베이스 클래스
│   │   ├── unicode_evasion.py
│   │   ├── llm_evasion.py
│   │   ├── homoglyph.py
│   │   └── leetspeak.py
│   ├── orchestrator.py                # 전략 오케스트레이터
│   └── __init__.py
│
├── defender/                          # 방어자 모듈 (Python)
│   ├── model.py                       # 분류 모델
│   ├── api.py                         # FastAPI 엔드포인트
│   └── __init__.py
│
├── training/                          # 학습 파이프라인 (Python)
│   ├── data_preparation.py            # 데이터 준비
│   ├── qlora_trainer.py               # QLoRA 학습
│   ├── auto_retrain.py                # 자동 재학습
│   ├── train.py                       # 학습 엔트리포인트
│   └── evaluate_and_deploy.py         # 평가 & 배포
│
├── mlops/                             # MLOps 모듈 (Python)
│   ├── model_registry.py              # MLflow 레지스트리
│   ├── evaluator.py                   # 모델 평가
│   ├── deployer.py                    # 자동 배포
│   ├── metrics.py                     # Prometheus 메트릭
│   └── __init__.py
│
├── infra/                             # 인프라 설정
│   ├── docker-compose.yml             # 개발 인프라
│   ├── init.sql                       # DB 초기화
│   ├── prometheus.yml                 # Prometheus 설정
│   └── grafana/
│       └── provisioning/
│           ├── dashboards/
│           └── datasources/
│
├── k8s/                               # Kubernetes 매니페스트
│   ├── base/
│   │   ├── namespace.yaml
│   │   ├── api-deployment.yaml
│   │   ├── ml-deployment.yaml
│   │   └── ...
│   └── overlays/
│       ├── dev/
│       └── prod/
│
├── data/                              # 데이터 디렉토리 (gitignore)
│   ├── jigsaw_sample.csv              # 원본 데이터셋 샘플
│   └── battle_collected/              # 배틀에서 수집된 데이터
│
├── models/                            # 모델 저장소 (gitignore)
│   ├── champion/                      # 현재 프로덕션 모델
│   └── challenger/                    # 평가 대기 모델
│
└── scripts/                           # 유틸리티 스크립트
    ├── setup.sh                       # 환경 설정
    ├── download_data.sh               # 데이터 다운로드
    └── run_battle.sh                  # 배틀 실행
```

---

## 🐹 Go API 서버 구조 (api-service/)

### 디렉토리별 역할

```
api-service/
├── cmd/server/main.go                 # 메인 엔트리포인트
│
├── internal/                          # 비공개 패키지
│   │
│   ├── config/                        # 설정 관리
│   │   └── config.go                  # Viper 기반 설정 로드
│   │
│   ├── handler/                       # HTTP 핸들러 (컨트롤러)
│   │   ├── battle_handler.go          # /battles 엔드포인트
│   │   ├── model_handler.go           # /models 엔드포인트
│   │   └── metrics_handler.go         # /metrics 엔드포인트
│   │
│   ├── service/                       # 비즈니스 로직
│   │   ├── battle_service.go          # 배틀 관리
│   │   ├── model_service.go           # 모델 관리
│   │   └── metrics_service.go         # 메트릭 집계
│   │
│   ├── repository/                    # 데이터 접근 (DAO)
│   │   ├── battle_repository.go       # 배틀 CRUD
│   │   ├── round_repository.go        # 라운드 CRUD
│   │   └── interfaces.go              # 인터페이스 정의
│   │
│   ├── model/                         # 도메인 모델 (Entity)
│   │   ├── battle.go                  # Battle 엔티티
│   │   ├── round.go                   # Round 엔티티
│   │   └── model_version.go           # ModelVersion 엔티티
│   │
│   ├── dto/                           # Data Transfer Objects
│   │   ├── battle_dto.go              # 배틀 Request/Response
│   │   ├── model_dto.go               # 모델 Request/Response
│   │   └── common_dto.go              # 공통 응답 구조
│   │
│   ├── middleware/                    # 미들웨어
│   │   ├── logger.go                  # 요청 로깅
│   │   ├── recovery.go                # 패닉 복구
│   │   ├── cors.go                    # CORS 설정
│   │   └── request_id.go              # Request ID 생성
│   │
│   ├── client/                        # 외부 서비스 클라이언트
│   │   ├── ml_client.go               # ML 서비스 호출
│   │   ├── attacker_client.go         # 공격자 서비스 호출
│   │   └── mlflow_client.go           # MLflow API 호출
│   │
│   └── router/                        # 라우터 설정
│       └── router.go                  # Gin 라우터 구성
│
└── pkg/                               # 공개 패키지 (재사용 가능)
    ├── logger/                        # 로깅 유틸리티
    └── errors/                        # 커스텀 에러
```

### 주요 파일 설명

**cmd/server/main.go**
```go
// 애플리케이션 시작점
// - 설정 로드
// - DB/Redis 연결
// - 라우터 설정
// - 서버 시작
```

**internal/service/battle_service.go**
```go
// 배틀 핵심 비즈니스 로직
// - 배틀 생성 및 실행
// - 라운드별 공격/방어 조율
// - 결과 집계 및 이벤트 발행
```

**internal/client/ml_client.go**
```go
// ML 서비스 HTTP 클라이언트
// - /classify 호출
// - /reload 호출
// - 타임아웃/재시도 처리
```

---

## 🐍 Python 서비스 구조

### ML 서비스 (ml-service/)

```
ml-service/
├── app/
│   ├── main.py                        # FastAPI 앱 인스턴스
│   ├── config.py                      # 설정 (pydantic-settings)
│   │
│   ├── api/                           # API 라우터
│   │   ├── __init__.py
│   │   ├── classify.py                # /classify 엔드포인트
│   │   └── health.py                  # /health 엔드포인트
│   │
│   ├── services/                      # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── model_service.py           # 모델 로드/추론
│   │   └── metrics_service.py         # 메트릭 수집
│   │
│   └── models/                        # Pydantic 모델
│       ├── __init__.py
│       └── schemas.py                 # Request/Response 스키마
│
├── tests/
│   ├── test_classify.py
│   └── test_model_service.py
│
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

### 공격자 모듈 (attacker/)

```
attacker/
├── __init__.py
│
├── strategies/                        # 공격 전략
│   ├── __init__.py
│   ├── base.py                        # AttackStrategy ABC
│   │
│   ├── unicode_evasion.py             # 유니코드 변형
│   │   # - 공백 삽입
│   │   # - 자모 분리
│   │   # - 유사 문자 치환
│   │   # - Zero-width 문자
│   │
│   ├── llm_evasion.py                 # LLM 기반 변형
│   │   # - Ollama 연동
│   │   # - 프롬프트 기반 변형
│   │
│   ├── homoglyph.py                   # 동형 문자 치환
│   │   # - Cyrillic, Greek 문자
│   │
│   └── leetspeak.py                   # 리트스피크
│       # - 문자 → 숫자/기호
│
├── orchestrator.py                    # 전략 오케스트레이터
│   # - 전략 조합 실행
│   # - 가중치 기반 선택
│   # - 전략 진화
│
└── prompts/                           # LLM 프롬프트 템플릿
    ├── evasion_basic.txt
    └── evasion_adversarial.txt
```

### 방어자 모듈 (defender/)

```
defender/
├── __init__.py
│
├── model.py                           # ContentFilter 클래스
│   # - 모델 로드 (base + LoRA)
│   # - 단일/배치 추론
│   # - 핫 리로드
│
└── api.py                             # FastAPI 엔드포인트
    # - /classify
    # - /classify/batch
    # - /classify/shadow
    # - /reload
```

### 학습 파이프라인 (training/)

```
training/
├── __init__.py
│
├── data_preparation.py                # 데이터 준비
│   # - Jigsaw 데이터셋 로드
│   # - Battle 데이터 수집
│   # - 토크나이징
│
├── qlora_trainer.py                   # QLoRA 학습
│   # - 4-bit 양자화 설정
│   # - LoRA 설정
│   # - Trainer 실행
│   # - MLflow 로깅
│
├── auto_retrain.py                    # 자동 재학습
│   # - 트리거 조건 모니터링
│   # - Redis 이벤트 구독
│   # - 학습 실행
│
├── train.py                           # 학습 CLI
│   # python train.py --config config.yaml
│
└── evaluate_and_deploy.py             # 평가 & 배포
    # - Champion/Challenger 비교
    # - 자동 승격
```

### MLOps 모듈 (mlops/)

```
mlops/
├── __init__.py
│
├── model_registry.py                  # MLflow Registry
│   # - Champion/Challenger 관리
│   # - Alias 설정
│   # - 모델 버전 조회
│
├── evaluator.py                       # 모델 평가
│   # - 테스트셋 평가
│   # - Champion vs Challenger
│   # - Shadow 평가
│
├── deployer.py                        # 자동 배포
│   # - 승격 판단
│   # - 핫 리로드 트리거
│   # - 알림 발송
│
├── metrics.py                         # Prometheus 메트릭
│   # - 메트릭 정의
│   # - 수집기 클래스
│
└── config.py                          # MLOps 설정
```

---

## 🐳 인프라 구조 (infra/)

```
infra/
├── docker-compose.yml                 # 개발 환경
│   # services:
│   #   - postgres
│   #   - redis
│   #   - mlflow
│   #   - prometheus
│   #   - grafana
│
├── init.sql                           # PostgreSQL 초기화
│   # - 데이터베이스 생성
│   # - 테이블 생성
│   # - 인덱스 생성
│
├── prometheus.yml                     # Prometheus 설정
│   # - scrape_configs
│   # - alerting rules
│
└── grafana/
    └── provisioning/
        ├── dashboards/
        │   ├── dashboard.yml          # 대시보드 프로비저닝
        │   └── content-arena.json     # 메인 대시보드
        │
        └── datasources/
            └── datasource.yml         # Prometheus 데이터소스
```

---

## ☸️ Kubernetes 구조 (k8s/)

```
k8s/
├── base/                              # 기본 매니페스트
│   ├── namespace.yaml
│   │
│   ├── api/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   │
│   ├── ml/
│   │   ├── deployment.yaml            # GPU 리소스 요청
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   │
│   ├── postgres/
│   │   ├── statefulset.yaml
│   │   ├── service.yaml
│   │   └── pvc.yaml
│   │
│   ├── redis/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   │
│   └── monitoring/
│       ├── prometheus/
│       └── grafana/
│
└── overlays/                          # 환경별 오버레이
    ├── dev/
    │   ├── kustomization.yaml
    │   └── patches/
    │
    └── prod/
        ├── kustomization.yaml
        └── patches/
```

---

## 📝 주요 설정 파일

### .env.example

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=content_arena
DB_USER=arena
DB_PASSWORD=secret

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct-v0.2-q4_K_S

# ML Service
ML_SERVICE_HOST=http://localhost:8001

# Training
TRAINING_BATCH_SIZE=2
TRAINING_LEARNING_RATE=0.0002
TRAINING_EPOCHS=3
LORA_R=16
LORA_ALPHA=32

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Makefile

```makefile
.PHONY: all build run test clean

# 전체 빌드
all: build

# 인프라 시작
infra-up:
	docker compose -f infra/docker-compose.yml up -d

infra-down:
	docker compose -f infra/docker-compose.yml down

# Go 서버
build-api:
	cd api-service && go build -o bin/server cmd/server/main.go

run-api:
	cd api-service && go run cmd/server/main.go

test-api:
	cd api-service && go test ./...

# Python ML 서버
run-ml:
	cd ml-service && uvicorn app.main:app --reload --port 8001

test-ml:
	cd ml-service && pytest

# 학습
train:
	cd training && python train.py

# 배틀 실행
battle:
	curl -X POST http://localhost:8080/api/v1/battles \
		-H "Content-Type: application/json" \
		-d '{"rounds": 100, "strategy": "mixed"}'

# 정리
clean:
	rm -rf api-service/bin
	rm -rf ml-service/__pycache__
	find . -name "*.pyc" -delete
```

---

## 🔍 파일 네이밍 컨벤션

### Go
```
# 파일명: snake_case
battle_service.go
battle_handler.go

# 패키지명: lowercase
package service
package handler

# 구조체: PascalCase
type BattleService struct {}
type BattleHandler struct {}
```

### Python
```
# 파일명: snake_case
battle_service.py
qlora_trainer.py

# 클래스명: PascalCase
class BattleService:
class QLoRATrainer:

# 함수/변수: snake_case
def create_battle():
def run_training():
```

### 공통
```
# 설정 파일: lowercase
config.yaml
.env

# 문서: UPPERCASE or Title Case
README.md
01-ARCHITECTURE.md

# 테스트: test_ prefix
test_battle_service.py
battle_service_test.go
```

---

## 📦 의존성 관리

### Go (go.mod)
```go
module content-arena/api-service

go 1.24

require (
    github.com/gin-gonic/gin v1.10.0
    gorm.io/gorm v1.25.12
    gorm.io/driver/postgres v1.5.9
    github.com/redis/go-redis/v9 v9.7.0
    github.com/prometheus/client_golang v1.20.0
    github.com/spf13/viper v1.19.0
    go.uber.org/zap v1.27.0
)
```

### Python (requirements.txt)
```txt
# Core ML (보수적 - 안정성 우선)
torch==2.5.1
transformers==4.48.3
peft==0.14.0
bitsandbytes==0.49.1
accelerate==1.5.2
datasets==3.2.0
safetensors==0.4.5

# API
fastapi==0.115.6
pydantic==2.10.3
uvicorn[standard]==0.32.1

# MLOps
mlflow==2.22.4

# Utils
python-dotenv==1.0.1
httpx==0.28.1
redis==5.2.1
```

---

## ✅ 체크리스트: 프로젝트 초기 설정

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/content-arena.git
cd content-arena

# 2. 환경 파일 생성
cp .env.example .env
# .env 수정

# 3. 인프라 시작
make infra-up

# 4. Go 의존성 설치
cd api-service && go mod download

# 5. Python 환경 설정
cd ../ml-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Ollama 모델 다운로드
ollama pull mistral:7b-instruct-v0.2-q4_K_S

# 7. 서비스 실행
make run-api   # 터미널 1
make run-ml    # 터미널 2

# 8. 테스트
make test-api
make test-ml

# 9. 첫 배틀 실행
make battle
```
