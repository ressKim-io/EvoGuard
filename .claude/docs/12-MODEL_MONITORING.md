# 모델 모니터링 설계

> ML 모델 성능 모니터링, Drift Detection, 자동 알림 시스템

## 개요

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│ Production  │───►│   Collector  │───►│   Detector  │───►│   Alerter   │
│   Traffic   │    │  (Metrics)   │    │   (Drift)   │    │  (Grafana)  │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
                          │                   │                   │
                          ▼                   ▼                   ▼
                   ┌──────────────────────────────────────────────────┐
                   │              Prometheus + MLflow                  │
                   └──────────────────────────────────────────────────┘
```

**관련 문서**:
- `dev-07-monitoring.md` - Prometheus + Grafana 기반
- `05-MLOPS.md` - Champion/Challenger 패턴
- `11-FEATURE_STORE.md` - Feature Store 설계

---

## 모니터링 유형

| 유형 | 설명 | 감지 대상 |
|------|------|-----------|
| **Data Drift** | 입력 데이터 분포 변화 | 텍스트 길이, 언어, 패턴 변화 |
| **Concept Drift** | 입력-출력 관계 변화 | 새로운 공격 패턴, 우회 기법 |
| **Model Drift** | 모델 성능 저하 | 정확도, F1, 신뢰도 하락 |
| **Feature Drift** | Feature 분포 변화 | Feature Store 통계 변화 |

---

## 1. Data Drift Detection

### 통계적 방법

| 방법 | 적용 대상 | 임계값 |
|------|-----------|--------|
| **KL Divergence** | 범주형 (언어, 카테고리) | > 0.1 |
| **KS Test** | 연속형 (길이, 비율) | p-value < 0.05 |
| **PSI** | 전체 분포 | > 0.2 |
| **JS Divergence** | 범주형/연속형 | > 0.1 |

### Population Stability Index (PSI)

```python
import numpy as np

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    PSI (Population Stability Index) 계산

    해석:
    - PSI < 0.1: 변화 없음 (안정)
    - 0.1 <= PSI < 0.2: 경미한 변화 (주의)
    - PSI >= 0.2: 유의미한 변화 (경고)
    """
    # 동일 bin 경계 사용
    bins = np.histogram_bin_edges(reference, bins=n_bins)

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    # 비율로 변환 (0 방지)
    ref_pct = (ref_hist + 1) / (len(reference) + n_bins)
    cur_pct = (cur_hist + 1) / (len(current) + n_bins)

    # PSI 계산
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
```

### KS Test (Kolmogorov-Smirnov)

```python
from scipy import stats

def detect_drift_ks(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    KS Test로 분포 변화 감지

    Returns:
        {"drifted": bool, "statistic": float, "p_value": float}
    """
    statistic, p_value = stats.ks_2samp(reference, current)

    return {
        "drifted": p_value < threshold,
        "statistic": float(statistic),
        "p_value": float(p_value),
    }
```

### 텍스트 데이터 Drift 모니터링

```python
class TextDataDriftMonitor:
    """텍스트 입력 데이터 Drift 모니터링"""

    def __init__(self, reference_stats: dict):
        self.reference = reference_stats

    def compute_current_stats(self, texts: list[str]) -> dict:
        """현재 텍스트 배치 통계 계산"""
        lengths = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        unicode_ratios = [
            sum(1 for c in t if ord(c) > 127) / max(len(t), 1)
            for t in texts
        ]

        return {
            "text_length": np.array(lengths),
            "word_count": np.array(word_counts),
            "unicode_ratio": np.array(unicode_ratios),
        }

    def detect_drift(self, texts: list[str]) -> dict:
        """Drift 감지"""
        current = self.compute_current_stats(texts)
        results = {}

        for feature in ["text_length", "word_count", "unicode_ratio"]:
            psi = calculate_psi(
                self.reference[feature],
                current[feature],
            )
            ks = detect_drift_ks(
                self.reference[feature],
                current[feature],
            )

            results[feature] = {
                "psi": psi,
                "psi_alert": psi >= 0.2,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "ks_alert": ks["drifted"],
            }

        return results
```

---

## 2. Concept Drift Detection

### 성능 기반 감지

모델의 실제 예측 성능 모니터링으로 Concept Drift 감지

```python
class ConceptDriftMonitor:
    """Concept Drift 모니터링 (성능 기반)"""

    def __init__(
        self,
        baseline_metrics: dict,
        window_size: int = 1000,
        threshold_drop: float = 0.05,  # 5% 성능 하락 시 알림
    ):
        self.baseline = baseline_metrics
        self.window_size = window_size
        self.threshold = threshold_drop
        self.predictions = []
        self.labels = []

    def add_feedback(self, prediction: int, actual: int):
        """피드백 추가 (라벨링된 데이터)"""
        self.predictions.append(prediction)
        self.labels.append(actual)

        # 윈도우 유지
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.labels.pop(0)

    def check_drift(self) -> dict:
        """Concept Drift 체크"""
        if len(self.predictions) < self.window_size // 2:
            return {"sufficient_data": False}

        from sklearn.metrics import f1_score, precision_score, recall_score

        current_f1 = f1_score(self.labels, self.predictions)
        current_precision = precision_score(self.labels, self.predictions)
        current_recall = recall_score(self.labels, self.predictions)

        f1_drop = self.baseline["f1"] - current_f1
        precision_drop = self.baseline["precision"] - current_precision
        recall_drop = self.baseline["recall"] - current_recall

        return {
            "sufficient_data": True,
            "current_f1": current_f1,
            "current_precision": current_precision,
            "current_recall": current_recall,
            "f1_drop": f1_drop,
            "precision_drop": precision_drop,
            "recall_drop": recall_drop,
            "drift_detected": f1_drop > self.threshold,
            "severity": self._classify_severity(f1_drop),
        }

    def _classify_severity(self, drop: float) -> str:
        if drop > 0.15:
            return "critical"
        elif drop > 0.10:
            return "high"
        elif drop > 0.05:
            return "medium"
        return "low"
```

### ADWIN (Adaptive Windowing)

실시간 스트림에서 Concept Drift 감지

```python
class ADWINDriftDetector:
    """
    ADWIN 알고리즘 기반 Drift 감지
    참고: river 라이브러리 활용 가능
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0

    def update(self, value: float) -> bool:
        """
        새 값 추가 및 Drift 감지

        Returns:
            True if drift detected
        """
        self.window.append(value)
        self.width += 1
        self.total += value

        if self.width > 1:
            # 간소화된 ADWIN 구현
            # 실제로는 river.drift.ADWIN 사용 권장
            return self._check_drift()
        return False

    def _check_drift(self) -> bool:
        """윈도우 분할 후 Drift 체크"""
        if self.width < 10:
            return False

        mean = self.total / self.width

        # 윈도우를 두 부분으로 분할하여 평균 비교
        for split in range(5, self.width - 5):
            mean1 = sum(self.window[:split]) / split
            mean2 = sum(self.window[split:]) / (self.width - split)

            # 평균 차이가 임계값 초과 시 Drift
            if abs(mean1 - mean2) > self.delta * mean:
                # 윈도우 축소
                self.window = self.window[split:]
                self.width = len(self.window)
                self.total = sum(self.window)
                return True

        return False
```

---

## 3. Prediction Monitoring

### 신뢰도 분포 모니터링

```python
from prometheus_client import Histogram, Counter, Gauge

# Prometheus 메트릭 정의
PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Model prediction confidence distribution',
    ['model', 'prediction'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total predictions',
    ['model', 'prediction']
)

LOW_CONFIDENCE_COUNT = Counter(
    'ml_low_confidence_predictions_total',
    'Predictions with confidence below threshold',
    ['model']
)

MODEL_DRIFT_SCORE = Gauge(
    'ml_drift_score',
    'Current drift score',
    ['model', 'drift_type']
)
```

### 예측 로깅

```python
class PredictionLogger:
    """예측 결과 로깅 및 모니터링"""

    def __init__(
        self,
        model_name: str,
        low_confidence_threshold: float = 0.7,
    ):
        self.model_name = model_name
        self.threshold = low_confidence_threshold

    def log_prediction(
        self,
        prediction: int,
        confidence: float,
        input_hash: str | None = None,
    ):
        """예측 결과 로깅"""
        label = "positive" if prediction == 1 else "negative"

        # Prometheus 메트릭 업데이트
        PREDICTION_CONFIDENCE.labels(
            model=self.model_name,
            prediction=label,
        ).observe(confidence)

        PREDICTION_COUNT.labels(
            model=self.model_name,
            prediction=label,
        ).inc()

        if confidence < self.threshold:
            LOW_CONFIDENCE_COUNT.labels(
                model=self.model_name,
            ).inc()

        # MLflow 로깅 (선택적)
        if input_hash:
            self._log_to_mlflow(prediction, confidence, input_hash)

    def _log_to_mlflow(self, prediction: int, confidence: float, input_hash: str):
        """MLflow에 예측 로깅 (샘플링)"""
        import mlflow
        import random

        # 1% 샘플링
        if random.random() < 0.01:
            mlflow.log_metric("sampled_confidence", confidence)
```

### 신뢰도 분포 이상 감지

```python
class ConfidenceAnomalyDetector:
    """신뢰도 분포 이상 감지"""

    def __init__(self, reference_mean: float, reference_std: float):
        self.ref_mean = reference_mean
        self.ref_std = reference_std
        self.current_confidences = []

    def add_confidence(self, confidence: float):
        """신뢰도 값 추가"""
        self.current_confidences.append(confidence)

    def check_anomaly(self, min_samples: int = 100) -> dict:
        """이상 감지"""
        if len(self.current_confidences) < min_samples:
            return {"sufficient_data": False}

        current_mean = np.mean(self.current_confidences)
        current_std = np.std(self.current_confidences)

        # Z-score 기반 이상 감지
        z_score = abs(current_mean - self.ref_mean) / self.ref_std

        return {
            "sufficient_data": True,
            "current_mean": current_mean,
            "current_std": current_std,
            "z_score": z_score,
            "anomaly_detected": z_score > 2.0,  # 2 표준편차 이상
        }

    def reset(self):
        """윈도우 리셋"""
        self.current_confidences = []
```

---

## 4. Feature Drift

### Feature Store 연동

```python
class FeatureDriftMonitor:
    """Feature Store 기반 Feature Drift 모니터링"""

    def __init__(self, feature_store, feature_group: str):
        self.fs = feature_store
        self.feature_group = feature_group
        self.reference_stats = None

    def set_reference(self, reference_data: pd.DataFrame):
        """Reference 분포 설정"""
        self.reference_stats = {}
        for col in reference_data.columns:
            if reference_data[col].dtype in ['int64', 'float64']:
                self.reference_stats[col] = {
                    "mean": reference_data[col].mean(),
                    "std": reference_data[col].std(),
                    "values": reference_data[col].values,
                }

    def check_drift(self, current_data: pd.DataFrame) -> dict:
        """Feature Drift 체크"""
        results = {}

        for col, ref_stats in self.reference_stats.items():
            if col not in current_data.columns:
                continue

            current_values = current_data[col].values

            # PSI 계산
            psi = calculate_psi(ref_stats["values"], current_values)

            # 평균 변화
            mean_shift = abs(
                current_data[col].mean() - ref_stats["mean"]
            ) / max(ref_stats["std"], 1e-6)

            results[col] = {
                "psi": psi,
                "mean_shift_zscore": mean_shift,
                "drift_detected": psi >= 0.2 or mean_shift > 3.0,
            }

        return results
```

---

## 5. Prometheus 메트릭 정의

### 핵심 메트릭

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `ml_prediction_confidence` | Histogram | 예측 신뢰도 분포 |
| `ml_predictions_total` | Counter | 총 예측 수 |
| `ml_low_confidence_predictions_total` | Counter | 저신뢰도 예측 수 |
| `ml_drift_score` | Gauge | Drift 점수 |
| `ml_model_f1_score` | Gauge | 현재 F1 점수 |
| `ml_data_drift_psi` | Gauge | Data Drift PSI |
| `ml_concept_drift_detected` | Gauge | Concept Drift 감지 (0/1) |
| `ml_retrain_recommended` | Gauge | 재학습 권장 (0/1) |

### 메트릭 수집기

```python
from prometheus_client import Gauge, generate_latest

# Drift 메트릭
DATA_DRIFT_PSI = Gauge(
    'ml_data_drift_psi',
    'Population Stability Index for data drift',
    ['model', 'feature']
)

CONCEPT_DRIFT_DETECTED = Gauge(
    'ml_concept_drift_detected',
    'Concept drift detection flag (1=detected)',
    ['model']
)

RETRAIN_RECOMMENDED = Gauge(
    'ml_retrain_recommended',
    'Model retrain recommendation flag',
    ['model']
)

MODEL_F1_SCORE = Gauge(
    'ml_model_f1_score',
    'Current model F1 score',
    ['model', 'dataset']  # dataset: production, validation
)


class DriftMetricsCollector:
    """Drift 메트릭 수집 및 노출"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def update_data_drift(self, feature: str, psi: float):
        """Data Drift PSI 업데이트"""
        DATA_DRIFT_PSI.labels(
            model=self.model_name,
            feature=feature,
        ).set(psi)

    def update_concept_drift(self, detected: bool):
        """Concept Drift 상태 업데이트"""
        CONCEPT_DRIFT_DETECTED.labels(
            model=self.model_name,
        ).set(1 if detected else 0)

    def update_f1_score(self, f1: float, dataset: str = "production"):
        """F1 점수 업데이트"""
        MODEL_F1_SCORE.labels(
            model=self.model_name,
            dataset=dataset,
        ).set(f1)

    def recommend_retrain(self, recommend: bool):
        """재학습 권장 플래그 업데이트"""
        RETRAIN_RECOMMENDED.labels(
            model=self.model_name,
        ).set(1 if recommend else 0)
```

---

## 6. 알림 규칙

### Grafana Alert Rules

```yaml
groups:
  - name: ml-model-alerts
    rules:
      # Data Drift 경고
      - alert: DataDriftDetected
        expr: ml_data_drift_psi > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected for {{ $labels.model }}"
          description: "PSI {{ $value | printf \"%.3f\" }} for feature {{ $labels.feature }}"

      # Concept Drift 경고
      - alert: ConceptDriftDetected
        expr: ml_concept_drift_detected == 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Concept drift detected for {{ $labels.model }}"
          description: "Model performance degradation detected. Consider retraining."

      # F1 점수 하락
      - alert: ModelF1ScoreDrop
        expr: ml_model_f1_score{dataset="production"} < 0.7
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Model F1 score dropped below 0.7"
          description: "Current F1: {{ $value | printf \"%.3f\" }}"

      # 저신뢰도 예측 급증
      - alert: LowConfidencePredictionSpike
        expr: |
          rate(ml_low_confidence_predictions_total[5m])
          / rate(ml_predictions_total[5m]) > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High rate of low confidence predictions"
          description: "{{ $value | printf \"%.1f\" }}% of predictions have low confidence"

      # 재학습 권장
      - alert: ModelRetrainRecommended
        expr: ml_retrain_recommended == 1
        for: 1m
        labels:
          severity: info
        annotations:
          summary: "Model retraining recommended for {{ $labels.model }}"
          description: "Drift metrics suggest model should be retrained"
```

### 알림 심각도 매핑

| 알림 | 조건 | 심각도 | 대응 |
|------|------|--------|------|
| DataDriftDetected | PSI > 0.2 (10분) | Warning | 모니터링 강화 |
| ConceptDriftDetected | Drift Flag = 1 | Critical | 재학습 검토 |
| ModelF1ScoreDrop | F1 < 0.7 (15분) | Critical | 즉시 재학습 |
| LowConfidencePredictionSpike | > 30% 저신뢰도 | Warning | 입력 데이터 검토 |
| ModelRetrainRecommended | Flag = 1 | Info | 재학습 스케줄링 |

---

## 7. 자동 대응 시스템

### 재학습 트리거

```python
class AutoRetrainTrigger:
    """자동 재학습 트리거"""

    def __init__(
        self,
        data_drift_threshold: float = 0.25,
        concept_drift_threshold: float = 0.10,
        min_interval_hours: int = 24,
    ):
        self.data_threshold = data_drift_threshold
        self.concept_threshold = concept_drift_threshold
        self.min_interval = timedelta(hours=min_interval_hours)
        self.last_retrain = None

    def should_retrain(
        self,
        data_drift_psi: float,
        concept_drift_f1_drop: float,
    ) -> dict:
        """재학습 필요 여부 판단"""
        # 최소 간격 체크
        if self.last_retrain:
            if datetime.utcnow() - self.last_retrain < self.min_interval:
                return {
                    "should_retrain": False,
                    "reason": "minimum_interval_not_met",
                }

        reasons = []

        if data_drift_psi > self.data_threshold:
            reasons.append(f"data_drift_psi={data_drift_psi:.3f}")

        if concept_drift_f1_drop > self.concept_threshold:
            reasons.append(f"f1_drop={concept_drift_f1_drop:.3f}")

        should_retrain = len(reasons) > 0

        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "priority": "high" if concept_drift_f1_drop > 0.15 else "normal",
        }

    def trigger_retrain(self):
        """재학습 트리거 및 기록"""
        self.last_retrain = datetime.utcnow()
        # GitHub Actions workflow 트리거 또는 직접 실행
        # ...
```

### Champion 자동 롤백

```python
class ChampionRollback:
    """Champion 모델 자동 롤백"""

    def __init__(self, mlflow_client, model_name: str):
        self.client = mlflow_client
        self.model_name = model_name

    def rollback_to_previous(self) -> dict:
        """이전 Champion으로 롤백"""
        # 현재 Champion 버전 확인
        current = self.client.get_model_version_by_alias(
            name=self.model_name,
            alias="champion",
        )

        # 이전 버전 찾기
        versions = self.client.search_model_versions(
            f"name='{self.model_name}'",
        )

        # 현재보다 이전 버전 중 가장 최신
        previous = None
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            if int(v.version) < int(current.version):
                previous = v
                break

        if not previous:
            return {"success": False, "reason": "no_previous_version"}

        # 롤백 실행
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="champion",
            version=previous.version,
        )

        return {
            "success": True,
            "from_version": current.version,
            "to_version": previous.version,
        }
```

---

## 8. 모듈 구조

```
ml-service/src/ml_service/monitoring/
├── __init__.py
├── drift/                       # Drift Detection
│   ├── __init__.py
│   ├── data_drift.py           # Data Drift (PSI, KS)
│   ├── concept_drift.py        # Concept Drift (ADWIN)
│   └── feature_drift.py        # Feature Drift
├── prediction/                  # Prediction Monitoring
│   ├── __init__.py
│   ├── logger.py               # 예측 로깅
│   └── confidence.py           # 신뢰도 모니터링
├── metrics/                     # Prometheus 메트릭
│   ├── __init__.py
│   └── collector.py            # 메트릭 수집기
├── alerts/                      # 알림 시스템
│   ├── __init__.py
│   ├── rules.py                # 알림 규칙
│   └── handlers.py             # 알림 핸들러
├── actions/                     # 자동 대응
│   ├── __init__.py
│   ├── retrain_trigger.py      # 재학습 트리거
│   └── rollback.py             # Champion 롤백
└── api/                         # API 라우터
    ├── __init__.py
    └── routes.py               # 모니터링 API
```

---

## 9. Grafana 대시보드

### ML Model Monitoring Dashboard

| 패널 | 메트릭 | 시각화 |
|------|--------|--------|
| Prediction Volume | `ml_predictions_total` | Time series |
| Confidence Distribution | `ml_prediction_confidence` | Heatmap |
| Data Drift PSI | `ml_data_drift_psi` | Gauge + Time series |
| F1 Score Trend | `ml_model_f1_score` | Time series |
| Low Confidence Rate | low_conf / total | Stat |
| Drift Alerts | Alert annotations | Timeline |

### 대시보드 JSON 위치

```
infra/grafana/provisioning/dashboards/json/
├── api-overview.json           # 기존 API 대시보드
└── ml-model-monitoring.json    # ML 모니터링 대시보드 (신규)
```

---

## 10. 구현 로드맵

### Phase 1: 기본 모니터링 (MVP) ✅ 완료
- [x] Prometheus 메트릭 정의 (`metrics/collector.py`)
- [x] 예측 로깅 구현 (`prediction/logger.py`)
- [x] 신뢰도 분포 모니터링 (`prediction/confidence.py`)

### Phase 2: Drift Detection
- [ ] Data Drift (PSI, KS Test)
- [ ] Concept Drift (성능 기반)
- [ ] Feature Drift (Feature Store 연동)

### Phase 3: 알림 시스템
- [ ] Grafana Alert Rules 설정
- [ ] AlertManager 연동
- [ ] Slack/Email 알림

### Phase 4: 자동 대응
- [ ] 재학습 자동 트리거
- [ ] Champion 롤백 로직
- [ ] 대시보드 구축

### Phase 5: 고도화
- [ ] ADWIN 실시간 Drift 감지
- [ ] A/B 테스트 연동
- [ ] 리포트 자동 생성

---

## 11. 참고 자료

### 외부 리소스
- [Evidently AI - ML Monitoring](https://www.evidentlyai.com/)
- [NannyML - Performance Estimation](https://nannyml.com/)
- [River - Online ML](https://riverml.xyz/) (ADWIN 구현)
- [Google MLOps - Continuous Evaluation](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### 내부 문서
- `dev-07-monitoring.md` - Prometheus + Grafana
- `05-MLOPS.md` - Champion/Challenger 패턴
- `11-FEATURE_STORE.md` - Feature Store 설계

---

*설계 완료: 2026-01-19*
*Phase 1 MVP 구현 완료: 2026-01-19*
