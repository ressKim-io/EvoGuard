# Content Arena (EvoGuard)

> Adversarial Learning 기반 자가 개선 콘텐츠 모더레이션 시스템

## 프로젝트 개요

공격자(Attacker)와 방어자(Defender)가 경쟁하며 진화하는 MLOps 프로젝트입니다.
- **공격자**: Ollama + Mistral 7B로 필터 우회 패턴 생성
- **방어자**: BERT + QLoRA Fine-tuned 모델로 유해 콘텐츠 탐지
- **목표**: 탐지율 60% → 90%+ 자동 개선

## 기술 스택

| 레이어 | 기술 | 버전 |
|--------|------|------|
| Backend API | Go + Gin | Go 1.24, Gin 1.10 |
| ML Service | Python + FastAPI | Python 3.12, FastAPI 0.115 |
| ML Training | Transformers + PEFT | Transformers 4.48, PEFT 0.14 |
| Local LLM | Ollama + Mistral 7B | Q4_K_S 양자화 |
| Database | PostgreSQL | 16.x |
| Cache/Queue | Redis | 7.x |
| ML Tracking | MLflow | 2.22 |
| Monitoring | Prometheus + Grafana | - |
| Container | Docker + k3d | - |

## 프로젝트 구조

```
content-arena/
├── api-service/          # Go API 서버 (Gin)
├── ml-service/           # Python ML 추론 서버 (FastAPI)
├── attacker/             # 공격 전략 모듈 (Python)
├── defender/             # 방어 모델 모듈 (Python)
├── training/             # QLoRA 학습 파이프라인 (Python)
├── mlops/                # MLOps 자동화 모듈 (Python)
├── infra/                # Docker Compose, 설정 파일
├── k8s/                  # Kubernetes 매니페스트
├── data/                 # 데이터셋 (gitignore)
├── models/               # 모델 체크포인트 (gitignore)
├── scripts/              # 유틸리티 스크립트
└── .claude/docs/         # 프로젝트 문서
```

## Git 전략

### 브랜치 전략: GitHub Flow
- `main` = 항상 배포 가능
- feature 브랜치 → PR → main 머지

### 브랜치 네이밍
```
{type}/{ticket}-{description}
```
| Type | 용도 | 예시 |
|------|------|------|
| `feature` | 새 기능 | `feature/JIRA-123-user-login` |
| `fix` | 버그 수정 | `fix/GH-456-auth-error` |
| `hotfix` | 긴급 수정 | `hotfix/ISSUE-789-security` |
| `refactor` | 리팩토링 | `refactor/DEV-101-cleanup` |
| `docs` | 문서 | `docs/JIRA-102-api-readme` |
| `chore` | 설정/빌드 | `chore/JIRA-103-ci` |

### 커밋 메시지 (Conventional Commits)
```
<type>(<scope>): <subject>
```
| Type | 설명 |
|------|------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 포맷팅 |
| `refactor` | 리팩토링 |
| `test` | 테스트 |
| `chore` | 빌드/설정 |
| `perf` | 성능 개선 |

**규칙:**
- 영어, 소문자 시작
- 명령형 (add, fix, update)
- 50자 이내, 마침표 없음

**예시:**
```
feat(auth): add OAuth2 login
fix(api): resolve null pointer exception
docs(readme): update installation guide
```

### 금지 사항
- main 직접 커밋/force push 금지
- 의미없는 메시지 금지 (`fix`, `update` 등)
- 민감정보 커밋 금지 (`.env`, credentials)
- 500줄+ 대규모 커밋 금지

## 개발 컨벤션

### Go (api-service/)
- 파일명: `snake_case.go`
- 패키지명: `lowercase`
- 구조체/인터페이스: `PascalCase`
- 함수/메서드: `PascalCase` (exported), `camelCase` (unexported)
- 테스트: `*_test.go`

### Python (ml-service/, attacker/, defender/, training/, mlops/)
- 파일명: `snake_case.py`
- 클래스명: `PascalCase`
- 함수/변수: `snake_case`
- 상수: `UPPER_SNAKE_CASE`
- 테스트: `test_*.py`

## 주요 API 엔드포인트

### Go API (`:8080`)
- `POST /api/v1/battles` - 배틀 생성
- `GET /api/v1/battles/:id` - 배틀 조회
- `GET /api/v1/models` - 모델 목록
- `POST /api/v1/models/promote` - Challenger 승격

### ML Service (`:8001`)
- `POST /classify` - 텍스트 분류
- `POST /classify/batch` - 배치 분류
- `POST /reload` - 모델 핫 리로드

## 환경 요구사항

- GPU: NVIDIA RTX 4060Ti (8GB VRAM) 이상
- RAM: 32GB 권장
- Docker + Docker Compose
- CUDA 12.4+

## Go 개발 가이드

### 핵심 원칙
- **Clean Architecture**: domain → usecase → adapter → infrastructure
- **Context 필수**: 모든 DB/Redis 호출에 context 전달
- **인터페이스 의존**: domain에서 인터페이스 정의, infrastructure에서 구현

### GORM 성능 설정
```go
gorm.Config{
    SkipDefaultTransaction: true,  // 필수
    PrepareStmt:            true,  // 필수
}
```

### Redis 설정 (go-redis v9.7+)
```go
redis.Options{
    Protocol:       3,      // RESP3
    ReadBufferSize: 32768,  // 32KB
}
```

### 주의사항
- N+1 방지: `db.Preload()` 사용
- Prometheus 라벨에 동적 값 금지 (카디널리티 폭발)
- 에러는 도메인 에러로 래핑 (`fmt.Errorf("%w", err)`)

## 참고 문서

상세 문서는 `.claude/docs/` 디렉토리 참조:

### 프로젝트 문서
- `01-ARCHITECTURE.md` - 시스템 아키텍처
- `02-TECH_STACK.md` - 기술 스택 상세
- `03-ENVIRONMENT_SETUP.md` - 환경 설정 가이드
- `04-ML_PIPELINE.md` - ML 파이프라인
- `05-MLOPS.md` - MLOps 파이프라인
- `06-API_SPEC.md` - API 명세
- `07-DEVELOPMENT_ROADMAP.md` - 개발 로드맵
- `08-PROJECT_STRUCTURE.md` - 프로젝트 구조

### Go 가이드
- `go-01-features.md` - Go 1.24 주요 변경사항
- `go-02-structure.md` - 프로젝트 구조 (Clean Architecture)
- `go-03-config-logging.md` - Viper 설정 & Zap 로깅
- `go-04-gorm-redis.md` - GORM & go-redis Best Practices
- `go-05-gin-prometheus.md` - Gin, Prometheus & DevOps

### Git 가이드
- `git-01-rules.md` - Git 규칙
- `git-02-commands.md` - Git 명령어
