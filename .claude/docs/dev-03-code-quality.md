# 코드 품질 도구 가이드

> 린터, 포매터, 타입 체커 설정

## 개요

EvoGuard는 다음 코드 품질 도구를 사용합니다:

| 언어 | 린터 | 포매터 | 타입 체커 |
|------|------|--------|-----------|
| Go | golangci-lint | gofmt (내장) | Go 컴파일러 |
| Python | ruff | ruff format | mypy |

## Go: golangci-lint

### 설치
```bash
# macOS
brew install golangci-lint

# go install
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

### 실행
```bash
# 전체 검사
make go-lint

# 직접 실행
cd api-service && golangci-lint run ./...

# 자동 수정 가능한 것 수정
golangci-lint run --fix ./...
```

### 설정 파일 위치
```
api-service/.golangci.yml
```

### 활성화된 린터

#### 기본 활성화
- `errcheck` - 에러 반환값 체크
- `gosimple` - 코드 단순화 제안
- `govet` - go vet 검사
- `ineffassign` - 비효율적 할당 감지
- `staticcheck` - 정적 분석
- `unused` - 미사용 코드 감지

#### 추가 활성화
- `bodyclose` - HTTP 응답 body 닫기 확인
- `contextcheck` - context 사용 검사
- `errname` - 에러 변수 네이밍
- `gocognit` - 인지 복잡도
- `goconst` - 상수로 변환 가능한 문자열
- `gocritic` - 다양한 검사
- `gofmt` - 포맷 검사
- `gosec` - 보안 검사
- `misspell` - 맞춤법 검사
- `prealloc` - 슬라이스 pre-allocation
- `revive` - fast, extensible linter

### 린터 제외 규칙
테스트 파일에서는 일부 린터 제외:
- `gocyclo`, `gocognit` - 테스트는 복잡도 높을 수 있음
- `errcheck` - 테스트에서는 에러 무시 가능
- `gosec` - 테스트 데이터는 하드코딩 가능

## Python: ruff

### 설치
```bash
# uv (권장)
uv pip install ruff

# pip
pip install ruff
```

### 실행
```bash
# 린트 검사
make py-lint

# 직접 실행
ruff check .

# 자동 수정
ruff check --fix .

# 포맷
ruff format .
```

### 설정 파일 위치
```
pyproject.toml (루트 또는 각 서비스)
```

### 활성화된 규칙

#### 핵심 규칙
- `F` - Pyflakes (에러, 미사용 변수 등)
- `E` - pycodestyle 에러
- `W` - pycodestyle 경고
- `I` - isort (import 정렬)

#### 권장 추가 규칙
- `UP` - pyupgrade (Python 버전 업그레이드)
- `B` - flake8-bugbear (버그 패턴)
- `C4` - flake8-comprehensions
- `SIM` - flake8-simplify
- `PTH` - pathlib 사용 권장
- `RUF` - Ruff 전용 규칙

### 무시하는 규칙
- `E501` - 라인 길이 (포매터가 처리)
- `E731` - lambda 할당 (때로 유용)

## Python: mypy

### 설치
```bash
uv pip install mypy
```

### 실행
```bash
make py-type
```

### 설정
`pyproject.toml`에서 설정:
```toml
[tool.mypy]
python_version = "3.12"
strict = true
```

## EditorConfig

에디터 간 일관성을 위한 `.editorconfig`:

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.go]
indent_style = tab

[*.py]
indent_style = space
indent_size = 4

[*.{yml,yaml}]
indent_style = space
indent_size = 2
```

## Pre-commit 통합

Git hooks를 통해 커밋 전 자동 검사:
- `scripts/git-hooks/pre-commit` 사용
- `make setup-hooks`로 설치

## CI 통합

GitHub Actions에서 동일한 도구 사용:
```yaml
- name: Run golangci-lint
  uses: golangci/golangci-lint-action@v6

- name: Run ruff
  run: ruff check .
```

## 트러블슈팅

### golangci-lint가 느림
```yaml
# .golangci.yml
run:
  timeout: 5m
  concurrency: 4
```

### ruff와 기존 설정 충돌
```bash
# 기존 설정 마이그레이션
ruff check --show-settings
```

### mypy 타입 에러가 너무 많음
```toml
# 점진적 적용
[tool.mypy]
strict = false
check_untyped_defs = true
```

## 참고 자료

- [golangci-lint Configuration](https://golangci-lint.run/docs/configuration/)
- [golangci-lint Linters](https://golangci-lint.run/docs/linters/)
- [Golden Config for golangci-lint](https://gist.github.com/maratori/47a4d00457a92aa426dbd48a18776322)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
- [Ruff Settings](https://docs.astral.sh/ruff/settings/)
- [mypy Configuration](https://mypy.readthedocs.io/en/stable/config_file.html)

---

*관련 문서: `dev-01-makefile.md`, `09-CI_CD.md`*
