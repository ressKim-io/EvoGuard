# 프로젝트 시작 체크리스트

> EvoGuard (Content Arena) 프로젝트 준비 상태 추적

## 완료 상태 범례
- ✅ 완료
- 🔄 진행 중
- ⬚ 미완료

---

## 1. Git 전략 ✅

| 항목 | 상태 | 관련 문서 |
|------|------|-----------|
| 브랜치 전략 (GitHub Flow) | ✅ | `git-01-rules.md` |
| 커밋 컨벤션 (Conventional Commits) | ✅ | `git-01-rules.md` |
| 커밋 주기 가이드 (Atomic Commits) | ✅ | `git-01-rules.md` |
| Pre-commit hooks (lint + test) | ✅ | `scripts/git-hooks/` |
| PR 템플릿 | ✅ | `.github/PULL_REQUEST_TEMPLATE.md` |
| Issue 템플릿 | ✅ | `.github/ISSUE_TEMPLATE/` |
| 템플릿 가이드 | ✅ | `dev-04-templates.md` |

---

## 2. 개발 환경 표준화

| 항목 | 상태 | 관련 문서/파일 |
|------|------|----------------|
| Makefile (공통 명령어) | ✅ | `Makefile`, `dev-01-makefile.md` |
| 환경 변수 템플릿 | ✅ | `.env.example`, `dev-02-environment.md` |
| EditorConfig | ✅ | `.editorconfig` |
| Docker Compose 개발 환경 | ✅ | `infra/docker-compose.yml` |
| 원커맨드 셋업 | ✅ | `make setup` |

### 왜 필요한가?
- **Makefile**: `make build`, `make test`, `make run` 등 통일된 명령어
- **.env.example**: 팀원이 필요한 환경 변수 파악 가능
- **Docker Compose**: 로컬 개발 환경 표준화 (DB, Redis 등)

---

## 3. CI/CD 파이프라인

| 항목 | 상태 | 관련 문서/파일 |
|------|------|----------------|
| CI/CD 가이드 문서 | ✅ | `09-CI_CD.md` |
| PR 테스트 워크플로우 | ✅ | `.github/workflows/pr-test.yml` |
| 메인 브랜치 빌드 | ✅ | `.github/workflows/build.yml` |
| 배포 워크플로우 | ✅ | `.github/workflows/deploy.yml` |

### 권장 CI 단계
1. Lint (golangci-lint, ruff)
2. Unit Test (go test, pytest)
3. Build (Docker image)
4. Security Scan (선택)

---

## 4. 코드 품질 도구

| 항목 | 상태 | 관련 파일 |
|------|------|-----------|
| Go: golangci-lint 설정 | ✅ | `api-service/.golangci.yml` |
| Python: ruff 설정 | ✅ | `pyproject.toml` |
| Python: mypy 타입 체크 | ✅ | `pyproject.toml` |
| EditorConfig | ✅ | `.editorconfig` |
| 코드 품질 가이드 | ✅ | `dev-03-code-quality.md` |

### 권장 규칙
- **Go**: golangci-lint의 `golangci-lint run ./...`
- **Python**: ruff (flake8 + isort + pyupgrade 통합)
- **공통**: 탭/스페이스, 줄바꿈 통일 → EditorConfig

---

## 5. 테스트 전략

| 항목 | 상태 | 관련 문서 |
|------|------|-----------|
| 테스트 가이드 문서 | ✅ | `10-TESTING.md` |
| Go 테스트 구조 | ✅ | `api-service/*_test.go` |
| Python 테스트 구조 | ✅ | `attacker/tests/` |
| 커버리지 목표 설정 | ✅ | `codecov.yml`, CI 설정 |
| E2E 테스트 계획 | ✅ | `10-TESTING.md` |

### 권장 커버리지 목표
- Unit Test: 70%+
- Integration Test: 핵심 경로
- E2E Test: 주요 시나리오

---

## 6. MLOps 특화

| 항목 | 상태 | 관련 문서 |
|------|------|-----------|
| MLflow 실험 추적 | ✅ | `py-05-mlflow.md`, `dev-05-mlops-local.md` |
| 모델 레지스트리 | ✅ | `05-MLOPS.md` |
| 데이터 버전 관리 (DVC) | ✅ | `.dvc/config` |
| Docker Compose MLOps | ✅ | `infra/docker-compose.yml` |
| Feature Store | ⬚ | 설계 필요 |
| 모델 모니터링 | ⬚ | Drift detection |

### MLOps 성숙도 단계
1. **Level 0**: 수동 프로세스
2. **Level 1**: ML 파이프라인 자동화
3. **Level 2**: CI/CD for ML (목표)

---

## 7. 모니터링 & 로깅

| 항목 | 상태 | 관련 문서 |
|------|------|-----------|
| 구조화된 로깅 표준 | ✅ | `go-03-config-logging.md` |
| Prometheus 메트릭 정의 | ✅ | `dev-07-monitoring.md` |
| Grafana 대시보드 | ✅ | `infra/grafana/provisioning/` |
| 알림 설정 | ✅ | `dev-07-monitoring.md` |

---

## 8. 보안

| 항목 | 상태 | 관련 문서 |
|------|------|-----------|
| 시크릿 관리 방법 | ✅ | `dev-06-security.md` |
| API 인증 방식 | ✅ | `dev-06-security.md` |
| 보안 스캐닝 | ✅ | `dev-06-security.md` |
| .gitignore 검증 | ✅ | `.gitignore` |

---

## 9. 문서화

| 항목 | 상태 | 관련 문서 |
|------|------|-----------|
| 아키텍처 문서 | ✅ | `01-ARCHITECTURE.md` |
| 기술 스택 문서 | ✅ | `02-TECH_STACK.md` |
| 환경 설정 가이드 | ✅ | `03-ENVIRONMENT_SETUP.md` |
| API 명세 | ✅ | `06-API_SPEC.md` |
| 개발 로드맵 | ✅ | `07-DEVELOPMENT_ROADMAP.md` |
| Contributing 가이드 | ✅ | `CONTRIBUTING.md` |

---

## 우선순위 작업 순서

### Phase 1: 개발 환경 ✅
1. ✅ Git 전략 및 hooks
2. ✅ Makefile 생성
3. ✅ .env.example 생성
4. ✅ 코드 품질 설정

### Phase 2: 자동화 ✅
5. ✅ CI/CD 파이프라인
6. ✅ PR/Issue 템플릿

### Phase 3: MLOps ✅
7. ✅ MLflow + Docker Compose 설정
8. ✅ 데이터 버전 관리 (DVC)

### Phase 4: 운영 ✅
9. ✅ 모니터링 설정
10. ✅ 보안 강화
11. ✅ CONTRIBUTING.md

---

## 참고 자료

### 외부 리소스
- [Developer Environment Setup Checklist 2024](https://daily.dev/blog/developer-environment-setup-checklist-2024)
- [MLOps Best Practices - Neptune.ai](https://neptune.ai/blog/mlops-best-practices)
- [Azure MLOps Foundation Checklist](https://microsoft.github.io/azureml-ops-accelerator/1-MLOpsFoundation/checklist.html)
- [Software Project Best Practices](https://kkovacs.eu/software-project-best-practices-checklist/)

### 내부 문서
- `.claude/docs/` - 프로젝트 가이드 문서
- `claude.md` - 프로젝트 요약

---

*마지막 업데이트: 2026-01-18*
