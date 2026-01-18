# Content Arena (EvoGuard)

> Adversarial Learning 기반 자가 개선 콘텐츠 모더레이션 시스템

## 프로젝트 개요

공격자(Attacker)와 방어자(Defender)가 경쟁하며 진화하는 MLOps 프로젝트
- **공격자**: Ollama + Mistral 7B로 필터 우회 패턴 생성
- **방어자**: BERT + QLoRA Fine-tuned 모델로 유해 콘텐츠 탐지
- **목표**: 탐지율 60% → 90%+ 자동 개선

## 기술 스택

| 레이어 | 기술 | 버전 |
|--------|------|------|
| Backend API | Go + Gin | Go 1.24, Gin 1.10 |
| ML Service | Python + FastAPI | Python 3.12, FastAPI 0.115 |
| ML Training | Transformers + PEFT | 4.48, 0.14 |
| Local LLM | Ollama + Mistral 7B | Q4_K_S |
| Database | PostgreSQL + Redis | 16.x, 7.x |
| ML Tracking | MLflow | 2.22 |
| Container | Docker + k3d | - |

## 프로젝트 구조

```
content-arena/
├── api-service/     # Go API 서버 (Gin)
├── ml-service/      # Python ML 추론 서버 (FastAPI)
├── attacker/        # 공격 전략 모듈
├── defender/        # 방어 모델 모듈
├── training/        # QLoRA 학습 파이프라인
├── mlops/           # MLOps 자동화
├── infra/           # Docker Compose, 설정
├── k8s/             # Kubernetes 매니페스트
└── .claude/docs/    # 상세 문서
```

## Git 규칙

### 브랜치: GitHub Flow
- `main` = 항상 배포 가능
- `{type}/{ticket}-{description}` 형식
- Types: `feature`, `fix`, `hotfix`, `refactor`, `docs`, `chore`

### 커밋: Conventional Commits
```
<type>(<scope>): <subject>
```
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`
- 영어, 소문자, 명령형, 50자 이내

### 규칙
- main 직접 커밋/force push 금지
- 500줄+ 대규모 커밋 금지
- 민감정보 커밋 금지 (`.env`, credentials)

상세: `.claude/docs/git-01-rules.md`

## 개발 컨벤션

| 언어 | 파일명 | 클래스/구조체 | 함수/변수 | 테스트 |
|------|--------|---------------|-----------|--------|
| Go | `snake_case.go` | `PascalCase` | `camelCase` | `*_test.go` |
| Python | `snake_case.py` | `PascalCase` | `snake_case` | `test_*.py` |

## 주요 API

### Go API (`:8080`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/v1/battles` | 배틀 생성 |
| GET | `/api/v1/battles/:id` | 배틀 조회 |
| GET | `/api/v1/models` | 모델 목록 |
| POST | `/api/v1/models/promote` | Challenger 승격 |

### ML Service (`:8001`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/classify` | 텍스트 분류 |
| POST | `/classify/batch` | 배치 분류 |
| POST | `/reload` | 모델 핫 리로드 |

## 환경 요구사항

- GPU: NVIDIA RTX 4060Ti (8GB VRAM) 이상
- RAM: 32GB 권장
- Docker + CUDA 12.4+

## 핵심 설정

### Go - GORM/Redis
```go
// GORM 필수 설정
gorm.Config{SkipDefaultTransaction: true, PrepareStmt: true}

// Redis 필수 설정 (go-redis v9.7+)
redis.Options{Protocol: 3}
```

### Python - QLoRA
```python
# 4-bit 양자화 필수 설정
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # 필수
    bnb_4bit_use_double_quant=True,
)
```

### 주의사항
- `device_map="auto"`는 **추론 전용** (학습 시 금지)
- N+1 방지: GORM `db.Preload()` 사용
- Prometheus 라벨에 동적 값 금지

## 참고 문서

`.claude/docs/` 디렉토리:

| 카테고리 | 문서 |
|----------|------|
| 프로젝트 | `00~09-*.md` (아키텍처, 스택, 환경, ML, MLOps, API, 로드맵, 구조, CI/CD) |
| Go | `go-01~05-*.md` (features, structure, config, gorm-redis, gin-prometheus) |
| Python | `py-00~06-*.md` (overview, uv, pytorch, transformers, quantization, mlflow, patterns) |
| 개발환경 | `dev-01~07-*.md` (makefile, env, quality, templates, mlops-local, security, monitoring) |
| Git | `git-01~02-*.md` (rules, commands) |
