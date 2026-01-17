# Go 프로젝트 구조 (Clean Architecture)

## 1. 디렉토리 구조

```
myproject/
├── cmd/api/main.go           # 엔트리포인트
├── internal/
│   ├── domain/               # 엔티티 & 비즈니스 규칙
│   │   ├── entity/
│   │   └── repository/       # 인터페이스 정의
│   ├── usecase/              # 애플리케이션 로직
│   ├── adapter/              # 외부 인터페이스 구현
│   │   ├── http/handler/
│   │   ├── http/middleware/
│   │   └── repository/postgres/
│   └── infrastructure/       # 외부 시스템 연결
│       ├── config/
│       ├── database/
│       ├── cache/
│       └── logger/
├── pkg/                      # 외부 공개 패키지
├── api/openapi.yaml
├── deployments/docker/
└── go.mod
```

## 2. 의존성 방향

```
        cmd/api/main.go (의존성 주입)
                │
                ▼
        adapter/http (핸들러)
                │
                ▼
           usecase (비즈니스 로직)
                │
                ▼
     domain/entity + repository (인터페이스)
                ▲
                │
     infrastructure (구현체: DB, Redis 등)
```

**규칙**: domain은 외부 의존성 없음, usecase는 인터페이스만 의존

## 3. go.mod 설정

```go
module github.com/yourname/myproject

go 1.24

tool (
    golang.org/x/tools/cmd/goimports
    github.com/golangci/golangci-lint/cmd/golangci-lint
    github.com/swaggo/swag/cmd/swag
)

require (
    github.com/gin-gonic/gin v1.10.0
    gorm.io/gorm v1.25.12
    gorm.io/driver/postgres v1.5.9
    github.com/redis/go-redis/v9 v9.7.0
    github.com/prometheus/client_golang v1.20.0
    github.com/spf13/viper v1.19.0
    go.uber.org/zap v1.27.0
)
```

## 4. 핵심 원칙

| 레이어 | 역할 | 의존 대상 |
|-------|------|----------|
| domain | 엔티티, 리포지토리 인터페이스 | 없음 |
| usecase | 비즈니스 로직 | domain 인터페이스 |
| adapter | HTTP 핸들러, DB 구현체 | usecase, domain |
| infrastructure | 설정, 연결 | 외부 라이브러리 |

## 5. internal vs pkg

- `internal/`: 프로젝트 내부 전용 (Go 컴파일러가 외부 import 차단)
- `pkg/`: 다른 프로젝트에서 import 가능한 공개 패키지
