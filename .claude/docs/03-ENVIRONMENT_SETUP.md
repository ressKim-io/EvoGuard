# ⚙️ 03. 환경 설정 가이드

> 개발 환경 구축부터 각 컴포넌트 설치까지 단계별 가이드

---

## 📋 사전 요구사항

### 하드웨어 최소 사양

```
┌─────────────────────────────────────────────────────────────────┐
│                      최소 사양 (권장 사양)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CPU:    4코어 이상 (8코어 이상)                               │
│  RAM:    16GB (32GB)                                           │
│  GPU:    NVIDIA RTX 3060 / VRAM 6GB (RTX 4060Ti / 8GB)        │
│  저장소: SSD 100GB (200GB)                                     │
│                                                                 │
│  ⚠️ 이 프로젝트는 4060Ti + RAM 32GB 기준으로 설계됨           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 소프트웨어 요구사항

| 항목 | 버전 | 확인 명령어 |
|------|------|------------|
| Docker | 27.4+ | `docker --version` |
| Docker Compose | 2.30+ | `docker compose version` |
| Go | 1.24+ | `go version` |
| Python | 3.12+ | `python --version` |
| NVIDIA Driver | 550+ | `nvidia-smi` |
| CUDA | 12.4+ | `nvcc --version` |
| Git | 2.40+ | `git --version` |

---

## 🪟 Windows (WSL2) 환경 설정

### 1. WSL2 설치 및 설정

```powershell
# 1. WSL 설치 (PowerShell 관리자 권한)
wsl --install

# 2. Ubuntu 22.04 설치
wsl --install -d Ubuntu-22.04

# 3. WSL 버전 확인
wsl -l -v
# Ubuntu-22.04가 VERSION 2로 표시되어야 함

# 4. 기본 배포판 설정
wsl --set-default Ubuntu-22.04
```

### 2. NVIDIA GPU 드라이버 (Windows)

```powershell
# 1. Windows에서 NVIDIA 드라이버 설치
# https://www.nvidia.com/download/index.aspx 에서 다운로드
# 또는 GeForce Experience 사용

# 2. 드라이버 확인
nvidia-smi
# CUDA Version: 12.x 표시 확인
```

### 3. Docker Desktop 설정

```
1. Docker Desktop 설치 (https://www.docker.com/products/docker-desktop/)

2. Settings > Resources > WSL Integration
   - "Enable integration with my default WSL distro" 활성화
   - Ubuntu-22.04 활성화

3. Settings > Resources > Advanced
   - Memory: 16GB 이상 할당
   - CPUs: 4개 이상 할당

4. WSL에서 Docker 확인
   $ docker run hello-world
```

### 4. NVIDIA Container Toolkit (WSL2 내부)

```bash
# WSL2 Ubuntu에서 실행

# 1. NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 2. Docker 재시작
sudo systemctl restart docker

# 3. GPU 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 🐧 Linux 환경 설정

### 1. NVIDIA 드라이버 설치

```bash
# Ubuntu 22.04 기준

# 1. 드라이버 설치
sudo apt update
sudo apt install -y nvidia-driver-535

# 2. 재부팅
sudo reboot

# 3. 확인
nvidia-smi
```

### 2. Docker 설치

```bash
# 1. 이전 버전 제거
sudo apt remove docker docker-engine docker.io containerd runc

# 2. Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. 사용자 그룹 추가 (sudo 없이 docker 명령 실행)
sudo usermod -aG docker $USER
newgrp docker

# 4. 확인
docker run hello-world
```

### 3. NVIDIA Container Toolkit

```bash
# (위 WSL2 섹션과 동일)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 🦙 Ollama 설치

### Linux / WSL2

```bash
# 1. Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 2. 서비스 시작
ollama serve &
# 또는 systemd로 관리
# sudo systemctl enable ollama
# sudo systemctl start ollama

# 3. Mistral 모델 다운로드
ollama pull mistral:7b-instruct-v0.2-q4_K_S

# 4. 모델 테스트
ollama run mistral:7b-instruct-v0.2-q4_K_S "Hello, world!"

# 5. API 테스트
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:7b-instruct-v0.2-q4_K_S",
  "prompt": "Hello, world!",
  "stream": false
}'
```

### macOS

```bash
# Homebrew로 설치
brew install ollama

# 나머지는 Linux와 동일
ollama serve &
ollama pull mistral:7b-instruct-v0.2-q4_K_S
```

### Ollama 환경 변수

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가

# Ollama 설정
export OLLAMA_HOST=0.0.0.0:11434     # 외부 접근 허용
export OLLAMA_MODELS=~/.ollama/models # 모델 저장 경로
export OLLAMA_NUM_PARALLEL=2          # 동시 요청 수
export OLLAMA_MAX_LOADED_MODELS=1     # 동시 로드 모델 수 (VRAM 절약)
```

---

## 🐍 Python 환경 설정

### 1. pyenv로 Python 버전 관리 (권장)

```bash
# 1. pyenv 설치
curl https://pyenv.run | bash

# 2. 환경 변수 추가 (~/.bashrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 3. 재로그인 후 Python 3.12 설치
pyenv install 3.12.8
pyenv global 3.12.8

# 4. 확인
python --version
# Python 3.12.8
```

### 2. 가상 환경 생성

```bash
# 프로젝트 디렉토리에서
cd content-arena/ml-service

# venv 생성
python -m venv venv

# 활성화
source venv/bin/activate  # Linux/macOS
# 또는
.\venv\Scripts\activate   # Windows

# pip 업그레이드
pip install --upgrade pip
```

### 3. PyTorch + CUDA 설치

```bash
# CUDA 12.4 버전 (4060Ti 최적)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 4. ML 패키지 설치

```bash
# requirements.txt
pip install -r requirements.txt
```

**requirements.txt:**

```txt
# Core ML (보수적 버전 - 안정성 우선)
torch==2.5.1
transformers==4.48.3
accelerate==1.5.2
datasets==3.2.0
safetensors==0.4.5

# QLoRA
peft==0.14.0
bitsandbytes==0.49.1

# API
fastapi==0.115.6
uvicorn[standard]==0.32.1
httpx==0.28.1
pydantic==2.10.3

# MLflow
mlflow==2.22.4

# Redis
redis==5.2.1

# Utilities
python-dotenv==1.0.1
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
tqdm==4.67.0

# Testing
pytest==8.0.0
pytest-asyncio==0.24.0
```

### 5. bitsandbytes 설치 (Windows 주의)

```bash
# Linux/WSL2 - 최신 버전은 Windows도 공식 지원
pip install bitsandbytes==0.49.1

# 설치 확인
python -c "import bitsandbytes as bnb; print(f'bitsandbytes: {bnb.__version__}')"

# ⚠️ Intel XPU 사용 시 PyTorch 2.6.0+ 필수
```

---

## 🐹 Go 환경 설정

### 1. Go 설치

```bash
# Linux/WSL2 - Go 1.24 (권장)
# ⚠️ Go 1.23.x는 EOL (2025년 8월) - 보안 패치 없음
wget https://go.dev/dl/go1.24.0.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz

# PATH 추가 (~/.bashrc)
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# 확인
go version
# go version go1.24 linux/amd64
```

### 2. 프로젝트 초기화

```bash
cd content-arena/api-service

# 모듈 초기화
go mod init content-arena/api-service

# 의존성 설치 (2026년 1월 기준 최신 안정 버전)
go get github.com/gin-gonic/gin@v1.10.0
go get gorm.io/gorm@v1.25.12
go get gorm.io/driver/postgres@v1.5.9
go get github.com/redis/go-redis/v9@v9.7.0
go get github.com/prometheus/client_golang@v1.20.0
go get github.com/spf13/viper@v1.19.0
go get go.uber.org/zap@v1.27.0

# 의존성 정리
go mod tidy
```

---

## 🐳 Docker 인프라 실행

### 1. 프로젝트 구조 생성

```bash
mkdir -p content-arena/{api-service,ml-service,infra}
cd content-arena
```

### 2. docker-compose.yml 생성

```yaml
# infra/docker-compose.yml
version: "3.9"

services:
  postgres:
    image: postgres:16-alpine
    container_name: arena-postgres
    environment:
      POSTGRES_DB: content_arena
      POSTGRES_USER: arena
      POSTGRES_PASSWORD: arena_secret_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U arena -d content_arena"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: arena-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    container_name: arena-mlflow
    ports:
      - "5000:5000"
    environment:
      MLFLOW_TRACKING_URI: postgresql://arena:arena_secret_123@postgres/mlflow
    command: >
      mlflow server 
      --host 0.0.0.0 
      --port 5000 
      --backend-store-uri postgresql://arena:arena_secret_123@postgres/mlflow 
      --default-artifact-root /mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: arena-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:10.2.0
    container_name: arena-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  mlflow_artifacts:
  prometheus_data:
  grafana_data:
```

### 3. 초기화 SQL

```sql
-- infra/init.sql
CREATE DATABASE mlflow;

\c content_arena;

-- battles 테이블
CREATE TABLE IF NOT EXISTS battles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    config JSONB NOT NULL,
    total_rounds INTEGER NOT NULL,
    completed_rounds INTEGER DEFAULT 0,
    evasion_count INTEGER DEFAULT 0,
    detection_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- battle_rounds 테이블
CREATE TABLE IF NOT EXISTS battle_rounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    battle_id UUID REFERENCES battles(id),
    round_number INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    evasion_text TEXT NOT NULL,
    attack_strategy VARCHAR(50) NOT NULL,
    toxic_score FLOAT NOT NULL,
    is_detected BOOLEAN NOT NULL,
    model_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(battle_id, round_number)
);

-- training_data 테이블
CREATE TABLE IF NOT EXISTS training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    label INTEGER NOT NULL,
    source VARCHAR(50),
    battle_id UUID REFERENCES battles(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_battle_rounds_battle_id ON battle_rounds(battle_id);
CREATE INDEX IF NOT EXISTS idx_training_data_label ON training_data(label);
```

### 4. Prometheus 설정

```yaml
# infra/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api-service'
    static_configs:
      - targets: ['host.docker.internal:8080']

  - job_name: 'ml-inference'
    static_configs:
      - targets: ['host.docker.internal:8001']
```

### 5. 인프라 실행

```bash
cd infra

# 시작
docker compose up -d

# 상태 확인
docker compose ps

# 로그 확인
docker compose logs -f

# 종료
docker compose down

# 볼륨까지 삭제
docker compose down -v
```

---

## ✅ 설치 확인 체크리스트

```bash
# 1. Docker & GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# ✅ GPU 정보 출력

# 2. Ollama
curl http://localhost:11434/api/generate -d '{"model":"mistral:7b-instruct-v0.2-q4_K_S","prompt":"hi","stream":false}'
# ✅ 응답 수신

# 3. Python & PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# ✅ True

# 4. PostgreSQL
docker exec -it arena-postgres psql -U arena -d content_arena -c "SELECT 1"
# ✅ 연결 성공

# 5. Redis
docker exec -it arena-redis redis-cli ping
# ✅ PONG

# 6. MLflow
curl http://localhost:5000/health
# ✅ 응답

# 7. Go
go version
# ✅ go version go1.24.x
```

---

## 🔧 트러블슈팅

### GPU 인식 안 됨

```bash
# 드라이버 상태 확인
nvidia-smi

# Docker GPU 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# nvidia-container-toolkit 재설치
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Ollama 연결 실패

```bash
# Ollama 서비스 상태 확인
systemctl status ollama

# 수동 실행
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 포트 확인
lsof -i :11434
```

### bitsandbytes 오류 (Windows)

```bash
# CUDA 경로 확인
where nvcc

# Windows용 빌드 사용
pip uninstall bitsandbytes
pip install bitsandbytes-windows
```

### PyTorch CUDA 버전 불일치

```bash
# 현재 설치된 CUDA 확인
nvcc --version

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 맞는 버전 재설치
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
