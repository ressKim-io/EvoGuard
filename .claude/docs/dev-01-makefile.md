# Makefile 가이드

> 프로젝트 빌드 및 개발 자동화

## 개요

Makefile은 프로젝트의 모든 개발 명령어를 통일된 인터페이스로 제공합니다.
- **일관성**: 팀 전체가 동일한 명령어 사용
- **문서화**: 명령어 자체가 문서 역할
- **자동화**: CI/CD에서도 동일하게 사용 가능

## 기본 명령어

```bash
# 전체 명령어 목록 보기
make help

# 개발 환경 설정
make setup

# 린트 실행
make lint

# 테스트 실행
make test

# 빌드
make build

# 실행
make run

# 정리
make clean
```

## 서비스별 명령어

### Go (api-service)
```bash
make go-lint      # golangci-lint 실행
make go-test      # go test 실행
make go-build     # 바이너리 빌드
make go-run       # 로컬 실행
```

### Python (ml-service, attacker, defender, training, mlops)
```bash
make py-lint      # ruff 실행
make py-test      # pytest 실행
make py-type      # mypy 타입 체크
make py-install   # 의존성 설치
```

### Docker
```bash
make docker-build   # 모든 이미지 빌드
make docker-up      # docker-compose up
make docker-down    # docker-compose down
make docker-logs    # 로그 보기
```

## 명령어 상세

### `make setup`
새로운 개발자가 프로젝트를 시작할 때 실행:
1. Git hooks 설치
2. Go 의존성 설치
3. Python 가상환경 생성 및 의존성 설치
4. 환경 변수 템플릿 복사

### `make lint`
모든 언어의 린트를 실행:
- Go: `golangci-lint run ./...`
- Python: `ruff check .`

### `make test`
모든 테스트 실행:
- Go: `go test -v -race ./...`
- Python: `pytest -v`

### `make build`
프로덕션 빌드:
- Go: 바이너리 컴파일
- Docker: 이미지 빌드

## Makefile 구조

```makefile
# 변수 정의
GO_SERVICE := api-service
PY_SERVICES := ml-service attacker defender training mlops

# .PHONY 타겟 (파일이 아닌 명령어)
.PHONY: help setup lint test build run clean

# 기본 타겟
.DEFAULT_GOAL := help

# 도움말
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 실제 타겟들...
```

## Best Practices

### 1. 모든 타겟에 설명 추가
```makefile
lint: ## Run linters for all languages
	@echo "Running linters..."
```

### 2. 의존성 있는 타겟 연결
```makefile
test: lint  ## Run tests (runs lint first)
	go test ./...
```

### 3. 환경별 변수
```makefile
ENV ?= development
ifeq ($(ENV),production)
    GO_BUILD_FLAGS := -ldflags="-s -w"
endif
```

### 4. 색상 출력
```makefile
GREEN  := $(shell tput setaf 2)
RESET  := $(shell tput sgr0)

lint:
	@echo "$(GREEN)Running linters...$(RESET)"
```

### 5. 에러 처리
```makefile
test:
	@go test ./... || (echo "Tests failed!"; exit 1)
```

## CI/CD에서 사용

GitHub Actions에서 동일한 Makefile 사용:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: make test
```

## 모노레포 팁

### 변경된 서비스만 빌드
```makefile
# 마지막 빌드 이후 변경 감지
api-service/.built: $(shell find api-service -name '*.go')
	cd api-service && go build -o bin/api
	touch $@
```

### 서비스별 Makefile
각 서비스에 개별 Makefile을 두고 루트에서 호출:
```makefile
go-build:
	$(MAKE) -C api-service build
```

## 참고 자료

- [Makefile for Monorepos](https://github.com/enspirit/makefile-for-monorepos)
- [Golang Monorepo Makefile](https://github.com/flowerinthenight/golang-monorepo)
- [Python Monorepo Template](https://github.com/niqodea/python-monorepo)
- [Makefile Inheritance](https://tommorris.org/posts/2023/til-makefile-inheritance-and-overriding/)

---

*관련 문서: `00-PROJECT_CHECKLIST.md`, `03-ENVIRONMENT_SETUP.md`*
