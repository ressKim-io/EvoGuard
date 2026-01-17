# 🛠️ 02. 기술 스택 상세

> 각 기술의 선택 이유, 버전, 대안 비교

---

## 📊 기술 스택 요약

| 레이어 | 기술 | 버전 | 용도 |
|--------|------|------|------|
| **서비스** | Go + Gin | Go 1.24 / Gin 1.10.0 | API 서버 |
| **ML 추론** | Python + FastAPI | 3.12.x / 0.115.x | 모델 서빙 |
| **ML 학습** | Transformers + PEFT | 4.48.x / 0.14.x | QLoRA Fine-tuning |
| **로컬 LLM** | Ollama + Mistral 7B | latest | 우회 패턴 생성 |
| **DB** | PostgreSQL | 16.6 / 17.x | 메인 데이터베이스 |
| **캐시** | Redis | 7.4.x | 캐싱, 이벤트 큐 |
| **ML 추적** | MLflow | 2.22.4 | 실험 추적, 모델 레지스트리 |
| **컨테이너** | Docker + Compose | 27.4.x | 컨테이너화 |
| **오케스트레이션** | k3d (Kubernetes) | 5.7.5 | 컨테이너 오케스트레이션 (Docker 기반) |
| **모니터링** | Prometheus + Grafana | 2.54.x / 11.3.x | 메트릭 수집/시각화 |
| **CI/CD** | GitHub Actions | - | 자동화 |

---

## 🐹 서비스 레이어: Go + Gin

### 왜 Go인가?

```
┌─────────────────────────────────────────────────────────────────┐
│                     Go 선택 이유                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DevOps 친화도 (K8s, Prometheus, Docker 모두 Go 기반)        │
│  2. 컨테이너 크기 (10-20MB vs Java 200MB+)                     │
│  3. 빠른 시작 시간 (콜드 스타트 최소화)                         │
│  4. 단일 바이너리 배포                                          │
│  5. 부트캠프 프로젝트와 차별화 (Java는 다른 팀원 담당)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Go vs Java vs Node.js 비교

| 항목 | Go | Java | Node.js |
|------|-----|------|---------|
| 컨테이너 크기 | 10-20MB | 200MB+ | 100MB+ |
| 콜드 스타트 | ~10ms | ~1000ms | ~100ms |
| 동시성 모델 | 고루틴 (경량) | 스레드 | 이벤트 루프 |
| 메모리 사용량 | 낮음 | 높음 | 중간 |
| DevOps 생태계 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| ML 생태계 연동 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### Gin Framework 선택 이유

```go
// Gin의 장점 코드 예시

// 1. 빠른 라우팅 (httprouter 기반)
router := gin.Default()
router.GET("/battles/:id", getBattle)
router.POST("/battles", createBattle)

// 2. 내장 미들웨어
router.Use(gin.Logger())
router.Use(gin.Recovery())
router.Use(cors.Default())

// 3. JSON 바인딩/검증
type CreateBattleRequest struct {
    Rounds   int    `json:"rounds" binding:"required,min=1,max=1000"`
    Strategy string `json:"strategy" binding:"required,oneof=unicode llm homoglyph"`
}

func createBattle(c *gin.Context) {
    var req CreateBattleRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    // ...
}

// 4. 그룹 라우팅
v1 := router.Group("/api/v1")
{
    battles := v1.Group("/battles")
    {
        battles.GET("", listBattles)
        battles.POST("", createBattle)
        battles.GET("/:id", getBattle)
    }
}
```

### 주요 Go 패키지

```go
// go.mod
module content-arena/api-service

go 1.24

require (
    github.com/gin-gonic/gin v1.10.0          // 웹 프레임워크
    github.com/gin-contrib/cors v1.7.3        // CORS
    gorm.io/gorm v1.25.12                     // ORM
    gorm.io/driver/postgres v1.5.9            // PostgreSQL 드라이버
    github.com/redis/go-redis/v9 v9.7.0       // Redis 클라이언트
    github.com/prometheus/client_golang v1.20.0 // Prometheus 메트릭
    github.com/google/uuid v1.6.0             // UUID 생성
    github.com/spf13/viper v1.19.0            // 설정 관리
    go.uber.org/zap v1.27.0                   // 구조화된 로깅
)
```

---

## 🐍 ML 레이어: Python

### Python 환경

```
Python 3.12.x (CUDA 호환성 확인됨)

주요 패키지 (보수적 - 안정성 우선):
├── torch==2.5.1+cu124          # PyTorch (CUDA 12.4)
├── transformers==4.48.3        # Hugging Face Transformers
├── peft==0.14.0                # LoRA/QLoRA
├── bitsandbytes==0.49.1        # 4-bit 양자화
├── accelerate==1.5.2           # 분산/혼합 정밀도 학습
├── datasets==3.2.0             # 데이터셋 처리
├── safetensors==0.4.5          # 모델 저장 포맷
├── mlflow==2.22.4              # 실험 추적
├── fastapi==0.115.6            # 추론 API
├── pydantic==2.10.3            # 데이터 검증
├── uvicorn==0.32.1             # ASGI 서버
├── httpx==0.28.1               # HTTP 클라이언트
└── redis==5.2.1                # Redis 클라이언트

⚠️ Breaking Changes 주의:
├── Go 1.23.x는 EOL (2025년 8월) - 1.24+ 사용
├── Python 3.9는 MLflow 3.x, PEFT 0.18+ 미지원
├── PEFT < 0.18은 Transformers v5 호환 불가
└── MLflow 2→3 마이그레이션 시 DB 스키마 변경 필요
```

### FastAPI 선택 이유

```python
# FastAPI의 장점

# 1. 자동 API 문서화 (Swagger UI)
from fastapi import FastAPI
app = FastAPI(
    title="Content Filter Inference API",
    description="콘텐츠 분류 추론 서비스",
    version="1.0.0"
)

# 2. Pydantic 검증
from pydantic import BaseModel, Field

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model_alias: str = Field(default="champion")

class ClassifyResponse(BaseModel):
    toxic_score: float
    is_toxic: bool
    confidence: float
    model_version: str

# 3. 비동기 지원
@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    result = await inference_service.classify(request.text)
    return result

# 4. 의존성 주입
from fastapi import Depends

def get_model_service():
    return ModelService.get_instance()

@app.post("/classify")
async def classify(
    request: ClassifyRequest,
    model: ModelService = Depends(get_model_service)
):
    return await model.classify(request.text)
```

---

## 🦙 로컬 LLM: Ollama + Mistral

### 왜 Ollama인가?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ollama 선택 이유                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 원클릭 설치 (curl -fsSL https://ollama.com/install.sh | sh)│
│  2. 모델 관리 간편 (ollama pull, ollama run)                   │
│  3. REST API 기본 제공 (localhost:11434)                       │
│  4. 양자화 모델 최적화                                          │
│  5. GPU 메모리 관리 자동화                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ollama vs 대안 비교

| 항목 | Ollama | vLLM | llama.cpp | HuggingFace TGI |
|------|--------|------|-----------|-----------------|
| 설치 난이도 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| API 기본 제공 | ✅ | ✅ | ❌ | ✅ |
| GPU 최적화 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 메모리 효율 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 문서/커뮤니티 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Mistral 7B 모델 선택

```bash
# 사용할 모델
ollama pull mistral:7b-instruct-v0.2-q4_K_S

# 모델 스펙
# - 파라미터: 7B
# - 양자화: 4-bit (Q4_K_S)
# - VRAM 요구량: ~4-5GB
# - 추론 속도: ~30 tokens/sec (4060Ti 기준)
```

### 4060Ti VRAM 8GB로 가능한 모델들

| 모델 | 파라미터 | 양자화 | VRAM 사용량 | 추천도 |
|------|----------|--------|-------------|--------|
| Mistral 7B Q4 | 7B | 4-bit | ~4-5GB | ⭐⭐⭐⭐⭐ |
| Llama 3.2 3B | 3B | FP16 | ~6GB | ⭐⭐⭐⭐ |
| Llama 3.1 8B Q4 | 8B | 4-bit | ~5-6GB | ⭐⭐⭐⭐ |
| Qwen 2.5 7B Q4 | 7B | 4-bit | ~5GB | ⭐⭐⭐⭐ |
| Phi-3 Mini | 3.8B | 4-bit | ~3GB | ⭐⭐⭐ |

---

## 🎯 QLoRA Fine-tuning

### QLoRA 핵심 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                      QLoRA 메모리 절감                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Fine-tuning (7B 모델):                                   │
│  ├── 모델 가중치: 7B × 2 bytes = 14GB                          │
│  ├── 그래디언트:  7B × 4 bytes = 28GB                          │
│  ├── 옵티마이저: 7B × 8 bytes = 56GB                           │
│  └── 총 필요:    ~100GB VRAM                                   │
│                                                                 │
│  QLoRA (4-bit + LoRA):                                         │
│  ├── 모델 가중치: 7B × 0.5 bytes = 3.5GB (4-bit 양자화)        │
│  ├── LoRA 파라미터: ~10M × 2 bytes = 20MB (0.1% 학습)          │
│  ├── 그래디언트: 10M × 4 bytes = 40MB                          │
│  └── 총 필요:    ~6-8GB VRAM ✅                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### QLoRA 설정 예시

```python
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 (QLoRA 핵심)
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 시 bf16 사용
    bnb_4bit_use_double_quant=True          # 이중 양자화 (메모리 추가 절감)
)

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    quantization_config=bnb_config,
    device_map="auto",
    num_labels=2
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,                                    # LoRA rank (작을수록 파라미터 적음)
    lora_alpha=32,                           # 스케일링 팩터 (보통 r * 2)
    target_modules=["query", "value"],       # 적용할 레이어
    lora_dropout=0.05,                       # 드롭아웃
    bias="none",
    task_type="SEQ_CLS"
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# 출력: trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%
```

### 8GB VRAM에서 안전한 학습 설정

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    
    # 배치 설정 (메모리 절약)
    per_device_train_batch_size=2,          # 작은 배치
    gradient_accumulation_steps=8,          # 그래디언트 누적으로 실효 배치 = 16
    
    # 정밀도 설정
    fp16=False,                             # 4060Ti는 bf16 더 효율적
    bf16=True,
    
    # 메모리 최적화
    optim="adamw_8bit",                     # 8-bit 옵티마이저
    gradient_checkpointing=True,            # 메모리-속도 트레이드오프
    
    # 학습 설정
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.1,
    
    # 로깅
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
)
```

---

## 📊 MLflow

### MLflow 구성요소

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLflow 구성                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Tracking                                                    │
│     - 실험 파라미터 기록                                        │
│     - 메트릭 기록 (loss, accuracy, f1)                         │
│     - 아티팩트 저장 (모델, 그래프)                              │
│                                                                 │
│  2. Model Registry                                              │
│     - 모델 버전 관리                                            │
│     - Alias 기반 배포 (champion, challenger)                    │
│     - Stage 관리 (Staging → Production)                         │
│                                                                 │
│  3. Projects                                                    │
│     - 재현 가능한 학습 환경                                     │
│     - MLproject 파일로 실행 정의                                │
│                                                                 │
│  4. Models                                                      │
│     - 모델 서빙 (mlflow models serve)                           │
│     - 다양한 플레이버 지원 (sklearn, pytorch, ...)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MLflow 사용 예시

```python
import mlflow
from mlflow.tracking import MlflowClient

# 1. 실험 추적
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("content-filter-training")

with mlflow.start_run(run_name="bert-qlora-v1"):
    # 파라미터 로깅
    mlflow.log_params({
        "model": "bert-base-uncased",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
        "learning_rate": 2e-4
    })
    
    # 메트릭 로깅
    mlflow.log_metrics({
        "train_loss": 0.35,
        "eval_loss": 0.42,
        "eval_f1": 0.87,
        "eval_accuracy": 0.89
    })
    
    # 모델 저장
    mlflow.pytorch.log_model(model, "model")

# 2. Model Registry
client = MlflowClient()

# 모델 등록
model_uri = f"runs:/{run.info.run_id}/model"
mv = client.create_model_version(
    name="content-filter",
    source=model_uri,
    run_id=run.info.run_id
)

# Alias 설정 (champion/challenger)
client.set_registered_model_alias(
    name="content-filter",
    alias="challenger",
    version=mv.version
)

# 3. Champion/Challenger 비교 후 승격
challenger_metrics = client.get_run(challenger_run_id).data.metrics
champion_metrics = client.get_run(champion_run_id).data.metrics

if challenger_metrics["eval_f1"] > champion_metrics["eval_f1"]:
    # Challenger를 Champion으로 승격
    client.set_registered_model_alias(
        name="content-filter",
        alias="champion",
        version=mv.version
    )
```

---

## 🗄️ 데이터베이스: PostgreSQL

### 선택 이유

```
PostgreSQL 선택 이유:
├── JSONB 지원 (설정, 메타데이터 저장에 유용)
├── 전문 검색 (텍스트 검색 기능)
├── 확장성 (파티셔닝, 레플리케이션)
├── Go/Python 드라이버 성숙
└── K8s 운영 경험 축적 (wealist 프로젝트)
```

---

## 🚀 Redis

### 사용 용도

```
Redis 사용 용도:
├── 캐싱
│   ├── 분류 결과 캐싱 (같은 텍스트 재요청 시)
│   ├── 모델 메타데이터 캐싱
│   └── 배틀 진행 상태
│
├── 이벤트/메시지 큐
│   ├── 재학습 트리거 이벤트
│   ├── 모델 교체 알림
│   └── 배틀 완료 이벤트
│
└── 분산 락
    └── 재학습 중복 실행 방지
```

---

## 📦 컨테이너 & 오케스트레이션

### Docker Compose (개발 환경)

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: ./api-service
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis

  ml-inference:
    build: ./ml-service
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:16
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=content_arena
      - POSTGRES_USER=arena
      - POSTGRES_PASSWORD=secret

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --backend-store-uri postgresql://arena:secret@postgres/mlflow

  prometheus:
    image: prom/prometheus:v2.48.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

### k3d (로컬 개발/데모)

```
k3d 선택 이유:
├── Docker 컨테이너로 k3s 실행 (별도 VM 불필요)
├── 클러스터 생성/삭제 초단위 (개발 중 빠른 반복)
├── Docker만 있으면 OK (WSL2 + Docker Desktop 환경 최적)
├── 풀 K8s API 호환 (k3s 기반)
├── 포트 매핑, 볼륨 마운트 Docker 친화적
└── 여러 클러스터 동시 운영 가능

# 설치
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# 클러스터 생성 (포트 매핑 포함)
k3d cluster create content-arena \
  --port "8080:80@loadbalancer" \
  --port "8443:443@loadbalancer" \
  --agents 1

# 클러스터 삭제
k3d cluster delete content-arena

# kubectl 컨텍스트 자동 설정됨
kubectl get nodes
```

---

## 📈 모니터링: Prometheus + Grafana

### 수집할 메트릭

```
# 비즈니스 메트릭
content_arena_battles_total{status="completed|failed"}
content_arena_rounds_total{detected="true|false"}
content_arena_detection_rate
content_arena_evasion_rate

# 모델 메트릭
content_arena_model_inference_duration_seconds
content_arena_model_version{alias="champion|challenger"}
content_arena_model_f1_score

# 시스템 메트릭
content_arena_api_request_duration_seconds
content_arena_api_requests_total{path, method, status}
content_arena_gpu_memory_used_bytes
```

---

## 🔄 CI/CD: GitHub Actions

### 워크플로우

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-api:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.24'
      - run: cd api-service && go test ./...

  test-ml:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: |
          cd ml-service
          pip install -r requirements-test.txt
          pytest

  build:
    needs: [test-api, test-ml]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: ./api-service
          push: true
          tags: ghcr.io/${{ github.repository }}/api:${{ github.sha }}
```
