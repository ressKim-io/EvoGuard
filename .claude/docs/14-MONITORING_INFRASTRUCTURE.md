# EvoGuard Monitoring Infrastructure 구축 계획

## 개요

프로덕션급 ML 모니터링 시스템 구축을 위한 Grafana + Prometheus 기반 인프라 설계

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EvoGuard Monitoring Architecture                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │  ml-service │     │ api-service │     │  attacker   │     │  defender   │   │
│  │  (FastAPI)  │     │    (Go)     │     │  (Python)   │     │  (Python)   │   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘   │
│         │ /metrics          │ /metrics          │ /metrics          │ /metrics │
│         └───────────────────┴───────────────────┴───────────────────┘          │
│                                      │                                          │
│                                      ▼                                          │
│                          ┌───────────────────────┐                              │
│                          │     Prometheus        │                              │
│                          │  (Time-Series DB)     │                              │
│                          │  - Scrape metrics     │                              │
│                          │  - Store 15d          │                              │
│                          │  - AlertManager       │                              │
│                          └───────────┬───────────┘                              │
│                                      │                                          │
│                    ┌─────────────────┼─────────────────┐                        │
│                    │                 │                 │                        │
│                    ▼                 ▼                 ▼                        │
│          ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
│          │   Grafana   │   │ AlertManager│   │    Loki     │                   │
│          │ Dashboards  │   │   (Slack)   │   │   (Logs)    │                   │
│          └─────────────┘   └─────────────┘   └─────────────┘                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 모니터링할 메트릭

### 1. ML 모델 성능 메트릭 (Core)

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `model_prediction_total` | Counter | 총 예측 요청 수 |
| `model_prediction_latency_seconds` | Histogram | 추론 지연 시간 |
| `model_prediction_confidence` | Histogram | 신뢰도 분포 |
| `model_prediction_toxic_total` | Counter | Toxic 판정 수 |
| `model_accuracy` | Gauge | 현재 정확도 (ground truth 비교) |
| `model_f1_score` | Gauge | 현재 F1 점수 |

### 2. Adversarial Pipeline 메트릭 (EvoGuard 특화)

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `adversarial_evasion_rate` | Gauge | 현재 회피율 |
| `adversarial_attack_total` | Counter | 총 공격 시도 |
| `adversarial_attack_success_total` | Counter | 성공한 공격 수 |
| `defender_retrain_total` | Counter | 재학습 횟수 |
| `attacker_evolution_total` | Counter | 공격자 진화 횟수 |
| `coevolution_cycle_total` | Counter | Co-evolution 사이클 수 |
| `coevolution_cycle_duration_seconds` | Histogram | 사이클 소요 시간 |

### 3. Data Drift 메트릭

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `data_drift_score` | Gauge | 데이터 드리프트 점수 (KL divergence) |
| `feature_drift_detected` | Gauge | 피처별 드리프트 감지 (0/1) |
| `input_text_length_avg` | Gauge | 평균 입력 텍스트 길이 |

### 4. 시스템 메트릭

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `gpu_memory_used_bytes` | Gauge | GPU 메모리 사용량 |
| `gpu_utilization_percent` | Gauge | GPU 사용률 |
| `model_load_time_seconds` | Gauge | 모델 로드 시간 |

## 기술 스택

### 핵심 구성요소

| 구성요소 | 버전 | 용도 |
|----------|------|------|
| **kube-prometheus-stack** | 81.0.0 | Helm 차트 (Prometheus + Grafana + AlertManager) |
| **Prometheus** | 2.x | 메트릭 수집 및 저장 |
| **Grafana** | 10.x | 시각화 및 대시보드 |
| **AlertManager** | 0.x | 알림 관리 |
| **Loki** | 2.x | 로그 수집 (선택) |

### Python 라이브러리

```python
# requirements.txt 추가
prometheus-client==0.20.0
prometheus-fastapi-instrumentator==7.0.0
```

## 구현 단계

### Phase 1: 로컬 개발 환경 (Docker Compose)

**목표**: 개발 중 빠른 피드백을 위한 로컬 모니터링

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - ./config/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

  alertmanager:
    image: prom/alertmanager:v0.27.0
    ports:
      - "9093:9093"
    volumes:
      - ./config/alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  prometheus_data:
  grafana_data:
```

### Phase 2: FastAPI 메트릭 노출

**목표**: ml-service에서 Prometheus 메트릭 노출

```python
# ml-service/src/ml_service/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

# Custom registry
REGISTRY = CollectorRegistry()

# ML Model Metrics
PREDICTION_COUNTER = Counter(
    'model_prediction_total',
    'Total number of predictions',
    ['model_version', 'label'],
    registry=REGISTRY
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

PREDICTION_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Prediction confidence distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    registry=REGISTRY
)

# Adversarial Metrics
EVASION_RATE = Gauge(
    'adversarial_evasion_rate',
    'Current evasion rate',
    registry=REGISTRY
)

ATTACK_COUNTER = Counter(
    'adversarial_attack_total',
    'Total attack attempts',
    ['strategy'],
    registry=REGISTRY
)

COEVOLUTION_CYCLE = Counter(
    'coevolution_cycle_total',
    'Total co-evolution cycles',
    ['action'],  # retrain_defender, evolve_attacker, balanced
    registry=REGISTRY
)

# Data Drift
DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Data drift score (KL divergence)',
    registry=REGISTRY
)

def setup_metrics(app):
    """Setup Prometheus metrics for FastAPI app."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        excluded_handlers=["/metrics", "/health"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
```

### Phase 3: Kubernetes 배포 (kube-prometheus-stack)

**목표**: 프로덕션 환경 모니터링

```yaml
# k8s/monitoring/values-prometheus-stack.yaml
kube-prometheus-stack:
  grafana:
    enabled: true
    adminPassword: "your-secure-password"
    persistence:
      enabled: true
      size: 10Gi
    dashboardProviders:
      dashboardproviders.yaml:
        apiVersion: 1
        providers:
          - name: 'evoguard'
            orgId: 1
            folder: 'EvoGuard'
            type: file
            disableDeletion: false
            editable: true
            options:
              path: /var/lib/grafana/dashboards/evoguard

  prometheus:
    prometheusSpec:
      retention: 15d
      storageSpec:
        volumeClaimTemplate:
          spec:
            accessModes: ["ReadWriteOnce"]
            resources:
              requests:
                storage: 50Gi
      serviceMonitorSelectorNilUsesHelmValues: false
      podMonitorSelectorNilUsesHelmValues: false

  alertmanager:
    config:
      global:
        slack_api_url: 'https://hooks.slack.com/services/xxx'
      route:
        group_by: ['alertname', 'severity']
        group_wait: 10s
        group_interval: 10s
        repeat_interval: 1h
        receiver: 'slack-notifications'
      receivers:
        - name: 'slack-notifications'
          slack_configs:
            - channel: '#evoguard-alerts'
              send_resolved: true
```

```yaml
# k8s/monitoring/servicemonitor-ml-service.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-service
  labels:
    app: ml-service
spec:
  selector:
    matchLabels:
      app: ml-service
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### Phase 4: Grafana 대시보드

#### Dashboard 1: EvoGuard Overview

```json
{
  "title": "EvoGuard Overview",
  "panels": [
    {
      "title": "Current Evasion Rate",
      "type": "gauge",
      "targets": [{"expr": "adversarial_evasion_rate"}]
    },
    {
      "title": "Predictions per Second",
      "type": "graph",
      "targets": [{"expr": "rate(model_prediction_total[5m])"}]
    },
    {
      "title": "P99 Latency",
      "type": "stat",
      "targets": [{"expr": "histogram_quantile(0.99, rate(model_prediction_latency_seconds_bucket[5m]))"}]
    },
    {
      "title": "Co-Evolution Cycles",
      "type": "graph",
      "targets": [{"expr": "increase(coevolution_cycle_total[1h])"}]
    }
  ]
}
```

#### Dashboard 2: Adversarial Pipeline

```json
{
  "title": "Adversarial Pipeline",
  "panels": [
    {
      "title": "Evasion Rate Over Time",
      "type": "graph",
      "targets": [{"expr": "adversarial_evasion_rate"}]
    },
    {
      "title": "Attack Success by Strategy",
      "type": "piechart",
      "targets": [{"expr": "sum by (strategy) (adversarial_attack_success_total)"}]
    },
    {
      "title": "Defender Retraining Events",
      "type": "graph",
      "targets": [{"expr": "increase(defender_retrain_total[1h])"}]
    },
    {
      "title": "Model F1 Score",
      "type": "graph",
      "targets": [{"expr": "model_f1_score"}]
    }
  ]
}
```

#### Dashboard 3: Model Performance

```json
{
  "title": "Model Performance",
  "panels": [
    {
      "title": "Prediction Confidence Distribution",
      "type": "heatmap",
      "targets": [{"expr": "model_prediction_confidence_bucket"}]
    },
    {
      "title": "Toxic vs Non-Toxic Ratio",
      "type": "piechart",
      "targets": [{"expr": "sum by (label) (model_prediction_total)"}]
    },
    {
      "title": "Latency Distribution",
      "type": "histogram",
      "targets": [{"expr": "model_prediction_latency_seconds_bucket"}]
    },
    {
      "title": "Data Drift Score",
      "type": "graph",
      "targets": [{"expr": "data_drift_score"}]
    }
  ]
}
```

### Phase 5: 알림 규칙

```yaml
# config/prometheus/alerts.yml
groups:
  - name: evoguard-alerts
    rules:
      # 높은 회피율 경고
      - alert: HighEvasionRate
        expr: adversarial_evasion_rate > 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High evasion rate detected"
          description: "Evasion rate is {{ $value | printf \"%.1f\" }}%, consider retraining"

      # 매우 높은 회피율 (긴급)
      - alert: CriticalEvasionRate
        expr: adversarial_evasion_rate > 0.5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical evasion rate!"
          description: "Evasion rate is {{ $value | printf \"%.1f\" }}%, immediate action required"

      # 모델 성능 저하
      - alert: ModelPerformanceDegraded
        expr: model_f1_score < 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model F1 score below threshold"
          description: "F1 score is {{ $value | printf \"%.2f\" }}"

      # 높은 지연 시간
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, rate(model_prediction_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "P99 latency is {{ $value | printf \"%.2f\" }}s"

      # 데이터 드리프트 감지
      - alert: DataDriftDetected
        expr: data_drift_score > 0.1
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Drift score is {{ $value | printf \"%.3f\" }}"

      # 서비스 다운
      - alert: MLServiceDown
        expr: up{job="ml-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML Service is down"
          description: "ml-service has been down for more than 1 minute"
```

## 구현 일정

| 단계 | 작업 | 예상 소요 |
|------|------|----------|
| **Phase 1** | Docker Compose 로컬 환경 | 2-3시간 |
| **Phase 2** | FastAPI 메트릭 통합 | 3-4시간 |
| **Phase 3** | K8s Helm 배포 | 4-6시간 |
| **Phase 4** | Grafana 대시보드 | 3-4시간 |
| **Phase 5** | 알림 규칙 설정 | 2-3시간 |
| **테스트** | 통합 테스트 및 튜닝 | 2-3시간 |

**총 예상 소요: 2-3일**

## 파일 구조

```
EvoGuard/
├── ml-service/
│   └── src/ml_service/
│       └── monitoring/
│           ├── __init__.py
│           ├── metrics.py          # Prometheus 메트릭 정의
│           ├── collectors.py       # 커스텀 콜렉터
│           └── drift_detector.py   # 데이터 드리프트 감지
├── config/
│   ├── prometheus/
│   │   ├── prometheus.yml         # Prometheus 설정
│   │   └── alerts.yml             # 알림 규칙
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   ├── datasources/
│   │   │   │   └── prometheus.yml
│   │   │   └── dashboards/
│   │   │       └── default.yml
│   │   └── dashboards/
│   │       ├── evoguard-overview.json
│   │       ├── adversarial-pipeline.json
│   │       └── model-performance.json
│   └── alertmanager/
│       └── alertmanager.yml
├── k8s/
│   └── monitoring/
│       ├── values-prometheus-stack.yaml
│       ├── servicemonitor-ml-service.yaml
│       └── servicemonitor-api-service.yaml
└── docker-compose.monitoring.yml
```

## 참고 자료

### 공식 문서
- [kube-prometheus-stack Helm Chart](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [Artifact Hub - kube-prometheus-stack](https://artifacthub.io/packages/helm/prometheus-community/kube-prometheus-stack)
- [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)

### 튜토리얼 및 가이드
- [Monitoring FastAPI with Grafana and Prometheus](https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn)
- [ML Monitoring Demo (Jeremy Jordan)](https://github.com/jeremyjordan/ml-monitoring)
- [ML Monitoring Best Practices](https://www.jeremyjordan.me/ml-monitoring/)
- [Kubernetes Monitoring Best Practices](https://slickfinch.com/monitoring-kubernetes-with-prometheus-and-grafana/)

### ML 특화 모니터링
- [MLOps Monitoring with Prometheus & Grafana](https://bowtiedraptor.substack.com/p/mlops-18-monitoring-with-prometheus)
- [BasisAI Model Drift Monitoring](https://grafana.com/blog/2021/08/02/how-basisai-uses-grafana-and-prometheus-to-monitor-model-drift-in-machine-learning-workloads/)
- [Grafana + ClearML ML Monitoring](https://grafana.com/blog/2023/08/18/monitoring-machine-learning-models-in-production-with-grafana-and-clearml/)

## 다음 단계

1. **즉시**: Phase 1 (Docker Compose) 구현하여 로컬에서 테스트
2. **이번 주**: Phase 2 (FastAPI 메트릭) 통합
3. **다음 주**: Phase 3-5 (K8s 배포 및 대시보드)
