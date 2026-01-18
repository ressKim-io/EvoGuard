# 모니터링 가이드

> Prometheus + Grafana 기반 모니터링

## 개요

EvoGuard는 다음 모니터링 스택을 사용합니다:

| 컴포넌트 | 역할 | 포트 |
|----------|------|------|
| Prometheus | 메트릭 수집 & 저장 | 9090 |
| Grafana | 시각화 & 알림 | 3000 |
| AlertManager | 알림 관리 | 9093 |

## 빠른 시작

```bash
# 모니터링 스택 시작
cd infra
docker compose --profile monitoring up -d

# 접속
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## 핵심 메트릭 (Four Golden Signals)

### 1. Latency (지연 시간)
요청 처리에 걸리는 시간

```promql
# P95 응답 시간
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)
```

### 2. Traffic (트래픽)
시스템에 대한 요청량

```promql
# 초당 요청 수
sum(rate(http_requests_total[5m]))

# 엔드포인트별
sum(rate(http_requests_total[5m])) by (method, path)
```

### 3. Errors (에러)
실패한 요청 비율

```promql
# 에러율 (%)
sum(rate(http_requests_total{status=~"5.."}[5m]))
/ sum(rate(http_requests_total[5m])) * 100
```

### 4. Saturation (포화도)
리소스 사용률

```promql
# CPU 사용률
rate(process_cpu_seconds_total[5m]) * 100

# 메모리 사용량
process_resident_memory_bytes / 1024 / 1024
```

## RED 메트릭 (서비스용)

| 메트릭 | 설명 | 대상 |
|--------|------|------|
| **R**ate | 초당 요청 수 | API, ML Service |
| **E**rrors | 에러 비율 | API, ML Service |
| **D**uration | 응답 시간 | API, ML Service |

## USE 메트릭 (리소스용)

| 메트릭 | 설명 | 대상 |
|--------|------|------|
| **U**tilization | 사용률 | CPU, Memory, Disk |
| **S**aturation | 대기열 길이 | Goroutines, Connections |
| **E**rrors | 에러 수 | Disk I/O, Network |

## Go API 메트릭 구현

### Prometheus 미들웨어

```go
// internal/infrastructure/middleware/metrics.go
package middleware

import (
    "github.com/gin-gonic/gin"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    httpRequests = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )

    httpDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5},
        },
        []string{"method", "path"},
    )
)

func PrometheusMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        timer := prometheus.NewTimer(httpDuration.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
        ))
        defer timer.ObserveDuration()

        c.Next()

        httpRequests.WithLabelValues(
            c.Request.Method,
            c.FullPath(),
            strconv.Itoa(c.Writer.Status()),
        ).Inc()
    }
}
```

### 메트릭 엔드포인트

```go
// cmd/api/main.go
import "github.com/prometheus/client_golang/prometheus/promhttp"

func main() {
    r := gin.Default()

    // 메트릭 엔드포인트
    r.GET("/metrics", gin.WrapH(promhttp.Handler()))

    // 미들웨어 적용
    r.Use(middleware.PrometheusMiddleware())
}
```

## Python ML Service 메트릭 구현

### prometheus-fastapi-instrumentator

```python
# ml-service/app/main.py
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# 자동 계측
Instrumentator().instrument(app).expose(app)

# 커스텀 메트릭
from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model', 'result']
)

INFERENCE_TIME = Histogram(
    'ml_inference_duration_seconds',
    'Model inference duration',
    ['model'],
    buckets=[.01, .025, .05, .1, .25, .5, 1, 2.5, 5]
)
```

### 사용 예시

```python
@app.post("/classify")
async def classify(request: ClassifyRequest):
    with INFERENCE_TIME.labels(model="defender").time():
        result = model.predict(request.text)

    PREDICTIONS.labels(model="defender", result=result).inc()
    return {"result": result}
```

## Grafana 대시보드

### 제공되는 대시보드

| 대시보드 | 설명 | 파일 |
|----------|------|------|
| API Overview | RED 메트릭 요약 | `api-overview.json` |

### 대시보드 프로비저닝

대시보드는 자동으로 프로비저닝됩니다:
```
infra/grafana/provisioning/
├── dashboards/
│   ├── dashboards.yml
│   └── json/
│       └── api-overview.json
└── datasources/
    └── datasources.yml
```

### 커스텀 대시보드 추가

1. Grafana UI에서 대시보드 생성
2. JSON으로 내보내기
3. `json/` 디렉토리에 저장
4. 자동으로 로드됨 (30초 주기)

## 알림 설정

### Grafana Alert Rules

```yaml
# Grafana UI 또는 프로비저닝으로 설정
groups:
  - name: api-alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | printf \"%.2f\" }}%"
```

### AlertManager 설정 (선택)

```yaml
# infra/alertmanager/alertmanager.yml
route:
  receiver: 'slack'
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

receivers:
  - name: 'slack'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'
```

## 권장 알림 규칙

| 메트릭 | 임계값 | 심각도 |
|--------|--------|--------|
| 에러율 | > 5% (5분) | Critical |
| P95 지연 | > 1s (5분) | Warning |
| CPU 사용률 | > 80% (10분) | Warning |
| 메모리 사용률 | > 85% (10분) | Warning |
| Pod 재시작 | > 3회 (1시간) | Critical |

## 트러블슈팅

### 메트릭이 수집되지 않음

```bash
# 1. 서비스에서 /metrics 확인
curl http://localhost:8080/metrics

# 2. Prometheus 타겟 확인
# http://localhost:9090/targets

# 3. 네트워크 확인 (Docker)
docker compose exec prometheus wget -q -O- http://host.docker.internal:8080/metrics
```

### Grafana 데이터 없음

1. Data source 연결 확인
2. 쿼리 시간 범위 확인
3. Prometheus에서 직접 쿼리 테스트

## 참고 자료

- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Four Golden Signals](https://sre.google/sre-book/monitoring-distributed-systems/)
- [RED Method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)

---

*관련 문서: `go-05-gin-prometheus.md`, `dev-05-mlops-local.md`*
