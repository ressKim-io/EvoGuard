# 환경 변수 가이드

> 환경 변수 관리 및 설정 방법

## 개요

환경 변수는 `.env` 파일을 통해 관리됩니다.
- **개발**: `.env` (gitignore)
- **템플릿**: `.env.example` (git 추적)
- **프로덕션**: K8s Secrets / Vault

## 시작하기

```bash
# 템플릿에서 .env 생성
cp .env.example .env

# 또는 make 사용
make setup-env
```

## 환경 변수 구조

### 공통
```bash
# 환경 (development, staging, production)
APP_ENV=development
LOG_LEVEL=debug
```

### Go API Service
```bash
API_HOST=0.0.0.0
API_PORT=8080
```

### Python ML Service
```bash
ML_HOST=0.0.0.0
ML_PORT=8001
```

### 데이터베이스
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=evoguard
DB_USER=postgres
DB_PASSWORD=<your-password>
DB_SSLMODE=disable
```

### Redis
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

### MLflow
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=evoguard
```

### Ollama (Local LLM)
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct-q4_K_S
```

## 환경별 설정

### Development
- 로컬 Docker 서비스 사용
- 상세 로깅 (`LOG_LEVEL=debug`)
- 모든 CORS 허용

### Staging
- 실제 인프라와 유사
- 중간 로깅 (`LOG_LEVEL=info`)
- 제한된 CORS

### Production
- K8s Secrets 사용
- 최소 로깅 (`LOG_LEVEL=warn`)
- 엄격한 보안 설정

## Best Practices

### 1. 민감 정보 분리
```bash
# ❌ Bad: .env.example에 실제 값
DB_PASSWORD=my-secret-password

# ✅ Good: 플레이스홀더
DB_PASSWORD=<your-password>
```

### 2. 기본값 제공
```bash
# Go에서 기본값 처리
port := os.Getenv("API_PORT")
if port == "" {
    port = "8080"
}
```

### 3. 필수 변수 검증
```python
# Python에서 검증
import os

required = ["DB_HOST", "DB_PASSWORD"]
missing = [v for v in required if not os.getenv(v)]
if missing:
    raise ValueError(f"Missing env vars: {missing}")
```

### 4. 타입 변환
```go
// Go에서 정수 변환
port, _ := strconv.Atoi(os.Getenv("API_PORT"))
```

```python
# Python에서 불린 변환
debug = os.getenv("DEBUG", "false").lower() == "true"
```

## Docker Compose 연동

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env
    environment:
      - DB_HOST=postgres  # 오버라이드
```

## 보안 주의사항

1. **절대 .env를 git에 커밋하지 않음**
2. **민감 정보는 암호화된 저장소 사용** (Vault, AWS Secrets Manager)
3. **프로덕션에서는 K8s Secrets 또는 환경 변수 주입**
4. **정기적인 시크릿 로테이션**

## 참고 자료

- [12 Factor App - Config](https://12factor.net/config)
- [Viper Go Config](https://github.com/spf13/viper)
- [Python dotenv](https://github.com/theskumar/python-dotenv)

---

*관련 문서: `go-03-config-logging.md`, `03-ENVIRONMENT_SETUP.md`*
