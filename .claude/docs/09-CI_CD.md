# CI/CD 가이드

> GitHub Actions 기반 CI/CD 파이프라인

## 개요

EvoGuard는 GitHub Actions를 사용한 CI/CD 파이프라인을 구성합니다.

### 파이프라인 구조
```
PR 생성/업데이트
    ↓
[Lint] → [Test] → [Build Check]
    ↓
PR 머지 (main)
    ↓
[Build] → [Push Image] → [Deploy]
```

## 워크플로우 파일

```
.github/
├── workflows/
│   ├── pr-test.yml      # PR 테스트 (lint + test)
│   ├── build.yml        # 빌드 및 이미지 푸시
│   └── deploy.yml       # 배포 (수동 트리거)
└── PULL_REQUEST_TEMPLATE.md
```

## PR 테스트 워크플로우

### 트리거
- Pull Request 생성/업데이트
- 특정 경로 변경 시에만 실행 (path filtering)

### 단계
1. **Lint**: golangci-lint (Go), ruff (Python)
2. **Test**: go test, pytest
3. **Build Check**: 빌드 가능 여부 확인

### Path Filtering
```yaml
on:
  pull_request:
    paths:
      - 'api-service/**'  # Go 변경 시
      - 'ml-service/**'   # Python 변경 시
```

## 빌드 워크플로우

### 트리거
- main 브랜치 푸시
- 태그 푸시 (릴리스)

### 단계
1. **Build**: Docker 이미지 빌드
2. **Push**: Container Registry에 푸시
3. **Tag**: 버전 태깅

## 모노레포 최적화

### 1. Path Filtering
변경된 서비스만 빌드/테스트:
```yaml
jobs:
  go-test:
    if: contains(github.event.head_commit.modified, 'api-service/')
```

### 2. Matrix Build
여러 Python 서비스 병렬 실행:
```yaml
strategy:
  matrix:
    service: [ml-service, attacker, defender, training, mlops]
```

### 3. Caching
의존성 캐싱으로 빌드 시간 단축:
```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/go-build
      ~/go/pkg/mod
    key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
```

### 4. Concurrency
중복 실행 취소:
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## 시크릿 관리

### 필요한 시크릿
| Name | 용도 |
|------|------|
| `GHCR_TOKEN` | GitHub Container Registry |
| `DOCKER_USERNAME` | Docker Hub (선택) |
| `DOCKER_PASSWORD` | Docker Hub (선택) |
| `KUBECONFIG` | K8s 배포용 |
| `HF_TOKEN` | Hugging Face (선택) |

### 설정 방법
1. GitHub Repo → Settings → Secrets and variables → Actions
2. New repository secret 클릭
3. 시크릿 추가

## 로컬에서 CI 시뮬레이션

```bash
# Makefile 사용
make ci

# 개별 단계
make lint
make test
make build
```

## Best Practices

### 1. 빠른 피드백
- 린트는 테스트보다 먼저 (빠른 실패)
- 병렬 실행 최대화

### 2. 재현 가능한 빌드
- 의존성 버전 고정 (go.sum, uv.lock)
- 동일한 도구 버전 사용

### 3. 보안
- 시크릿은 GitHub Secrets 사용
- OIDC로 클라우드 인증 (가능하면)
- 최소 권한 원칙

### 4. 비용 관리
- `cancel-in-progress: true` 사용
- macOS 러너는 필요할 때만
- 불필요한 캐시 정리

## 트러블슈팅

### 일반적인 문제

**캐시 미스**
```yaml
# 캐시 키에 OS와 해시 포함
key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
```

**권한 오류**
```yaml
permissions:
  contents: read
  packages: write  # GHCR 푸시 시 필요
```

**타임아웃**
```yaml
timeout-minutes: 30  # 기본 360분은 너무 길음
```

## 참고 자료

- [GitHub Actions in 2026 Guide](https://dev.to/pockit_tools/github-actions-in-2026-the-complete-guide-to-monorepo-cicd-and-self-hosted-runners-1jop)
- [Monorepo with GitHub Actions](https://graphite.com/guides/monorepo-with-github-actions)
- [Vanilla GitHub Actions Monorepo](https://generalreasoning.com/blog/2025/03/22/github-actions-vanilla-monorepo.html)
- [LogRocket: Monorepo CI/CD](https://blog.logrocket.com/creating-separate-monorepo-ci-cd-pipelines-github-actions/)

---

*관련 문서: `00-PROJECT_CHECKLIST.md`, `dev-01-makefile.md`*
