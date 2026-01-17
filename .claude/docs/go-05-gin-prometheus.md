# Gin, Prometheus & DevOps

## 1. Gin 라우터 설정

```go
gin.SetMode(gin.ReleaseMode)
router := gin.New()

router.Use(gin.Recovery())
router.Use(middleware.RequestID())
router.Use(middleware.Logger())
router.Use(middleware.Prometheus())

router.GET("/health", handler.HealthCheck)
router.GET("/metrics", gin.WrapH(promhttp.Handler()))

api := router.Group("/api/v1")
api.Use(authMiddleware.Authenticate())
{
    api.GET("/users/:id", userHandler.GetByID)
}
```

## 2. 핸들러 패턴

```go
func (h *UserHandler) GetByID(c *gin.Context) {
    id := c.Param("id")
    
    user, err := h.usecase.GetByID(c.Request.Context(), id)
    if err != nil {
        handleError(c, err)
        return
    }
    
    c.JSON(http.StatusOK, Response[*User]{
        Success: true,
        Data:    user,
    })
}

type Response[T any] struct {
    Success bool `json:"success"`
    Data    T    `json:"data,omitempty"`
}
```

---

## 3. Prometheus 메트릭스

```go
var (
    HTTPRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
        },
        []string{"method", "endpoint", "status"},
    )
    
    HTTPRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Buckets: []float64{.001, .01, .1, .5, 1, 5},
        },
        []string{"method", "endpoint"},
    )
)
```

### 미들웨어
```go
func Prometheus() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        c.Next()
        
        endpoint := c.FullPath()  // /users/:id (동적 값 X)
        HTTPRequestsTotal.WithLabelValues(
            c.Request.Method, endpoint, strconv.Itoa(c.Writer.Status()),
        ).Inc()
        HTTPRequestDuration.WithLabelValues(
            c.Request.Method, endpoint,
        ).Observe(time.Since(start).Seconds())
    }
}
```

**주의**: 라벨에 `/users/123` 같은 동적 값 넣으면 카디널리티 폭발

---

## 4. 테스트

### 단위 테스트 (Mock)
```go
type MockUserRepo struct { mock.Mock }

func (m *MockUserRepo) FindByID(ctx context.Context, id string) (*User, error) {
    args := m.Called(ctx, id)
    return args.Get(0).(*User), args.Error(1)
}

func TestGetByID(t *testing.T) {
    mockRepo := new(MockUserRepo)
    mockRepo.On("FindByID", mock.Anything, "123").
        Return(&User{ID: "123"}, nil)
    
    uc := NewUserUsecase(mockRepo)
    got, err := uc.GetByID(context.Background(), "123")
    
    assert.NoError(t, err)
    assert.Equal(t, "123", got.ID)
}
```

### 통합 테스트
```go
func TestUserHandler(t *testing.T) {
    gin.SetMode(gin.TestMode)
    router := setupTestRouter()
    
    w := httptest.NewRecorder()
    req, _ := http.NewRequest("GET", "/api/v1/users/123", nil)
    router.ServeHTTP(w, req)
    
    assert.Equal(t, http.StatusOK, w.Code)
}
```

---

## 5. Dockerfile (멀티스테이지)

```dockerfile
FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o server ./cmd/api

FROM alpine:3.19
RUN adduser -D appuser
USER appuser
COPY --from=builder /app/server .
EXPOSE 8080
HEALTHCHECK --interval=30s CMD wget -q --spider http://localhost:8080/health
ENTRYPOINT ["./server"]
```

---

## 6. Makefile

```makefile
build:
	go build -o bin/app ./cmd/api

test:
	go test -race -coverprofile=coverage.out ./...

lint:
	go tool golangci-lint run

swagger:
	go tool swag init -g cmd/api/main.go
```

---

## 7. 체크리스트

### 코드
- [ ] Context 전달
- [ ] 에러 래핑 (`fmt.Errorf("%w", err)`)
- [ ] 구조화된 로깅
- [ ] 인터페이스로 의존성 추상화

### 배포
- [ ] `go vet ./...`
- [ ] `golangci-lint run`
- [ ] 테스트 커버리지
- [ ] Docker 빌드 테스트
