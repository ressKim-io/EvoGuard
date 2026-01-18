# 기술 스택

> 각 기술의 선택 이유 및 버전

## 스택 요약

| 레이어 | 기술 | 버전 | 용도 | 상세 문서 |
|--------|------|------|------|-----------|
| **서비스** | Go + Gin | Go 1.24 / Gin 1.10 | API 서버 | `go-*.md` |
| **ML 추론** | Python + FastAPI | 3.12 / 0.115 | 모델 서빙 | `py-*.md` |
| **ML 학습** | Transformers + PEFT | 4.48 / 0.14 | QLoRA Fine-tuning | `py-03-*.md` |
| **로컬 LLM** | Ollama + Mistral 7B | latest | 우회 패턴 생성 | - |
| **DB** | PostgreSQL | 16.x | 메인 데이터베이스 | - |
| **캐시** | Redis | 7.x | 캐싱, 이벤트 큐 | `go-04-*.md` |
| **ML 추적** | MLflow | 2.22 | 실험 추적, 모델 레지스트리 | `py-05-*.md` |
| **컨테이너** | Docker + Compose | 27.x | 컨테이너화 | - |
| **오케스트레이션** | k3d | 5.7 | 로컬 K8s | - |
| **모니터링** | Prometheus + Grafana | 2.54 / 11.3 | 메트릭 수집/시각화 | `dev-07-*.md` |
| **CI/CD** | GitHub Actions | - | 자동화 | `09-CI_CD.md` |

## Go + Gin 선택 이유

| 항목 | Go | Java | Node.js |
|------|-----|------|---------|
| 컨테이너 크기 | 10-20MB | 200MB+ | 100MB+ |
| 콜드 스타트 | ~10ms | ~1000ms | ~100ms |
| DevOps 생태계 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**선택 포인트**: K8s, Prometheus, Docker가 모두 Go 기반. 단일 바이너리 배포, 빠른 시작 시간.

## Python ML 환경

**주요 패키지** (버전 고정 - 안정성 우선):
- `torch==2.5.1+cu124` - PyTorch (CUDA 12.4)
- `transformers==4.48.3` - Hugging Face
- `peft==0.14.0` - LoRA/QLoRA
- `bitsandbytes==0.49.1` - 4-bit 양자화
- `mlflow==2.22.4` - 실험 추적
- `fastapi==0.115.6` - 추론 API

**Breaking Changes 주의**:
- Python 3.9는 MLflow 3.x, PEFT 0.18+ 미지원
- PEFT < 0.18은 Transformers v5 호환 불가

## Ollama + Mistral 7B

| 항목 | Ollama | vLLM | llama.cpp |
|------|--------|------|-----------|
| 설치 난이도 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| API 기본 제공 | ✅ | ✅ | ❌ |
| 메모리 효율 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**사용 모델**: `mistral:7b-instruct-v0.2-q4_K_S`
- VRAM: ~4-5GB
- 추론 속도: ~30 tokens/sec (4060Ti)

### 8GB VRAM 호환 모델

| 모델 | 양자화 | VRAM | 추천도 |
|------|--------|------|--------|
| Mistral 7B | Q4 | ~4-5GB | ⭐⭐⭐⭐⭐ |
| Llama 3.2 3B | FP16 | ~6GB | ⭐⭐⭐⭐ |
| Llama 3.1 8B | Q4 | ~5-6GB | ⭐⭐⭐⭐ |
| Qwen 2.5 7B | Q4 | ~5GB | ⭐⭐⭐⭐ |

## QLoRA 메모리 절감

```
Full Fine-tuning (7B):  ~100GB VRAM 필요
QLoRA (4-bit + LoRA):   ~6-8GB VRAM ✅
```

**핵심 설정**: `py-04-quantization.md` 참조
- `bnb_4bit_quant_type="nf4"` - NormalFloat4
- `bnb_4bit_compute_dtype=torch.bfloat16`
- `bnb_4bit_use_double_quant=True` - 이중 양자화

## MLflow 구성요소

| 컴포넌트 | 역할 |
|----------|------|
| **Tracking** | 파라미터, 메트릭, 아티팩트 기록 |
| **Model Registry** | 버전 관리, Alias (champion/challenger) |
| **Projects** | 재현 가능한 학습 환경 |

상세: `py-05-mlflow.md`

## PostgreSQL & Redis

**PostgreSQL 선택 이유**:
- JSONB 지원 (설정, 메타데이터)
- Go/Python 드라이버 성숙
- K8s 운영 경험

**Redis 용도**:
- 캐싱 (분류 결과, 모델 메타데이터)
- 이벤트 큐 (재학습 트리거)
- 분산 락 (중복 실행 방지)

## 컨테이너 환경

### Docker Compose (개발)
`infra/docker-compose.yml` 참조

### k3d (로컬 K8s)
```bash
# 클러스터 생성
k3d cluster create content-arena \
  --port "8080:80@loadbalancer" \
  --agents 1

# 삭제
k3d cluster delete content-arena
```

**k3d 선택 이유**: Docker 컨테이너로 K8s 실행, 클러스터 생성/삭제 초단위

## 모니터링

**핵심 메트릭**: `dev-07-monitoring.md` 참조
- `content_arena_detection_rate` - 탐지율
- `content_arena_evasion_rate` - 우회율
- `content_arena_model_inference_duration_seconds` - 추론 시간

## CI/CD

GitHub Actions 워크플로우: `09-CI_CD.md` 참조
- PR 테스트 (Go test, pytest)
- Docker 이미지 빌드
- 배포 자동화
