# Contributing to EvoGuard

EvoGuard 프로젝트에 기여해 주셔서 감사합니다! 이 문서는 기여 방법을 안내합니다.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

- 상호 존중하는 커뮤니케이션
- 건설적인 피드백 제공
- 다양한 의견 존중

## Getting Started

### 1. 저장소 Fork & Clone

```bash
# Fork 후 clone
git clone https://github.com/YOUR_USERNAME/EvoGuard.git
cd EvoGuard

# Upstream 설정
git remote add upstream https://github.com/ORIGINAL_OWNER/EvoGuard.git
```

### 2. 개발 환경 설정

```bash
# 환경 변수 설정
cp .env.example .env

# 의존성 설치
make setup

# 개발 서버 시작
make dev
```

자세한 내용은 `.claude/docs/03-ENVIRONMENT_SETUP.md` 참조.

## Development Workflow

### 1. 최신 코드 동기화

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Feature 브랜치 생성

```bash
# 브랜치 네이밍: {type}/{ticket}-{description}
git checkout -b feature/ISSUE-123-add-new-model
```

| Type | 용도 |
|------|------|
| `feature` | 새 기능 |
| `fix` | 버그 수정 |
| `hotfix` | 긴급 수정 |
| `refactor` | 리팩토링 |
| `docs` | 문서 |
| `chore` | 설정/빌드 |

### 3. 개발 & 테스트

```bash
# 코드 작성 후 테스트
make test

# 린트 검사
make lint
```

### 4. 커밋 & Push

```bash
git add .
git commit -m "feat(model): add new classification model"
git push origin feature/ISSUE-123-add-new-model
```

## Commit Guidelines

### Conventional Commits 형식

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Type 목록

| Type | 설명 |
|------|------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 포맷팅 (기능 변경 없음) |
| `refactor` | 리팩토링 |
| `test` | 테스트 추가/수정 |
| `chore` | 빌드/설정 변경 |
| `perf` | 성능 개선 |

### 규칙

- 영어, 소문자 시작
- 명령형 사용 (add, fix, update)
- 50자 이내
- 마침표 없음

### 좋은 예시

```bash
feat(auth): add OAuth2 login support
fix(api): resolve null pointer in battle handler
docs(readme): update installation guide
refactor(ml): simplify prediction pipeline
```

### 나쁜 예시

```bash
# 너무 모호함
fix bug
update

# 대문자/과거형
Fixed the bug
Added feature
```

## Pull Request Process

### 1. PR 생성 전 체크리스트

- [ ] 테스트 통과 (`make test`)
- [ ] 린트 통과 (`make lint`)
- [ ] 문서 업데이트 (필요 시)
- [ ] 커밋 메시지 규칙 준수

### 2. PR 작성

- **제목**: 커밋 메시지와 동일한 형식
- **설명**: 변경 내용, 테스트 방법, 관련 이슈 포함

### 3. 리뷰 프로세스

1. 자동 CI 검사 통과
2. 최소 1명의 리뷰어 승인
3. 모든 코멘트 해결
4. Squash & Merge

### 4. 머지 후

- Feature 브랜치 삭제
- 관련 이슈 닫기

## Coding Standards

### Go (api-service/)

```go
// 파일명: snake_case.go
// 패키지명: lowercase
// 구조체/인터페이스: PascalCase
// 함수/메서드: PascalCase (exported), camelCase (unexported)

// 좋은 예시
type BattleService struct {
    repo BattleRepository
}

func (s *BattleService) CreateBattle(ctx context.Context, req CreateBattleRequest) (*Battle, error) {
    // 구현
}
```

자세한 내용은 `.claude/docs/go-*.md` 참조.

### Python (ml-service/, training/)

```python
# 파일명: snake_case.py
# 클래스명: PascalCase
# 함수/변수: snake_case
# 상수: UPPER_SNAKE_CASE

# 좋은 예시
class TextClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, text: str) -> dict:
        """텍스트 분류 수행."""
        pass
```

자세한 내용은 `.claude/docs/py-*.md` 참조.

## Testing

### Go 테스트

```bash
# 전체 테스트
make test-go

# 특정 패키지
go test ./internal/usecase/...

# 커버리지
go test -coverprofile=coverage.out ./...
```

### Python 테스트

```bash
# 전체 테스트
make test-python

# pytest 직접 실행
cd ml-service && uv run pytest

# 커버리지
uv run pytest --cov=app
```

### 테스트 작성 가이드

- 단위 테스트: 함수/메서드 단위
- 통합 테스트: API 엔드포인트, DB 연동
- 테스트 파일 위치: 소스 파일과 동일 디렉토리

## Documentation

### 문서 위치

| 문서 유형 | 위치 |
|-----------|------|
| 프로젝트 개요 | `README.md` |
| 기여 가이드 | `CONTRIBUTING.md` |
| 상세 문서 | `.claude/docs/` |
| API 명세 | `.claude/docs/06-API_SPEC.md` |

### 문서 업데이트

코드 변경 시 관련 문서도 함께 업데이트해 주세요:

- 새 기능 추가 → README 또는 관련 docs 업데이트
- API 변경 → API 명세 업데이트
- 설정 변경 → 환경 설정 가이드 업데이트

---

질문이 있으시면 이슈를 생성하거나 메인테이너에게 연락해 주세요.
