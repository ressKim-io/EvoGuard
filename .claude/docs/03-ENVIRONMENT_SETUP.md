# 환경 설정 가이드

> 개발 환경 구축 단계별 가이드

## 사전 요구사항

### 하드웨어 최소 사양

| 항목 | 최소 | 권장 |
|------|------|------|
| CPU | 4코어 | 8코어 |
| RAM | 16GB | 32GB |
| GPU | RTX 3060 (6GB) | RTX 4060Ti (8GB) |
| 저장소 | SSD 100GB | SSD 200GB |

### 소프트웨어 요구사항

| 항목 | 버전 | 확인 명령어 |
|------|------|------------|
| Docker | 27.4+ | `docker --version` |
| Go | 1.24+ | `go version` |
| Python | 3.12+ | `python --version` |
| NVIDIA Driver | 550+ | `nvidia-smi` |
| CUDA | 12.4+ | `nvcc --version` |

## 빠른 시작 (원커맨드)

```bash
# 저장소 클론 후
make setup
```

## OS별 설정

### Windows (WSL2)

```powershell
# 1. WSL2 설치
wsl --install -d Ubuntu-22.04

# 2. NVIDIA 드라이버 (Windows에서)
# https://www.nvidia.com/download/ 에서 다운로드

# 3. Docker Desktop 설정
# Settings > WSL Integration > Ubuntu-22.04 활성화
```

WSL2 내부에서:
```bash
# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Linux

```bash
# 1. NVIDIA 드라이버
sudo apt update && sudo apt install -y nvidia-driver-535
sudo reboot

# 2. Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 3. NVIDIA Container Toolkit (WSL2와 동일)
```

## Ollama 설치

```bash
# 설치
curl -fsSL https://ollama.com/install.sh | sh

# 서비스 시작
ollama serve &

# Mistral 모델 다운로드
ollama pull mistral:7b-instruct-v0.2-q4_K_S

# 테스트
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:7b-instruct-v0.2-q4_K_S",
  "prompt": "Hello!",
  "stream": false
}'
```

**환경 변수** (`~/.bashrc`):
```bash
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MAX_LOADED_MODELS=1  # VRAM 절약
```

## Python 환경

```bash
# pyenv로 Python 설치
curl https://pyenv.run | bash
pyenv install 3.12.8
pyenv global 3.12.8

# uv 패키지 매니저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
cd ml-service
uv sync
```

**PyTorch + CUDA 확인**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# True
```

상세: `py-01-uv-setup.md`

## Go 환경

```bash
# Go 1.24 설치
wget https://go.dev/dl/go1.24.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz

# PATH (~/.bashrc)
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go

# 의존성 설치
cd api-service
go mod tidy
```

## Docker 인프라

```bash
cd infra

# 시작
docker compose up -d

# 상태 확인
docker compose ps

# 종료
docker compose down
```

**서비스 포트**:
| 서비스 | 포트 |
|--------|------|
| PostgreSQL | 5432 |
| Redis | 6379 |
| MLflow | 5000 |
| Prometheus | 9090 |
| Grafana | 3000 |

설정 파일: `infra/docker-compose.yml`

## 설치 확인 체크리스트

```bash
# Docker GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Ollama
curl http://localhost:11434/api/tags

# Python + PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# PostgreSQL
docker exec arena-postgres psql -U arena -d content_arena -c "SELECT 1"

# Redis
docker exec arena-redis redis-cli ping

# Go
go version
```

## 트러블슈팅

### GPU 인식 안 됨
```bash
nvidia-smi  # 드라이버 확인
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Ollama 연결 실패
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
lsof -i :11434  # 포트 확인
```

### PyTorch CUDA 버전 불일치
```bash
nvcc --version  # 시스템 CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
# 버전 맞춰서 재설치
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
