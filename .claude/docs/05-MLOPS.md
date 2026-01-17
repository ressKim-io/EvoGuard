# ⚙️ 05. MLOps 파이프라인

> Champion/Challenger 패턴, 자동 배포, 모니터링 시스템

---

## 🎯 MLOps 개요

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLOps LIFECYCLE                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
    │  │  Data    │───►│  Train   │───►│ Register │───►│ Compare  │         │
    │  │ Pipeline │    │ Pipeline │    │ (MLflow) │    │ (A/B)    │         │
    │  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘         │
    │                                                       │               │
    │                                       ┌───────────────┴───────────────┐
    │                                       ▼                               ▼
    │                              ┌──────────────┐                ┌──────────────┐
    │                              │   Promote    │                │   Reject     │
    │                              │ (Champion)   │                │ (Archive)    │
    │                              └──────┬───────┘                └──────────────┘
    │                                     │                                       │
    │                                     ▼                                       │
    │                              ┌──────────────┐                              │
    │                              │   Deploy     │                              │
    │                              │ (Hot Reload) │                              │
    │                              └──────┬───────┘                              │
    │                                     │                                       │
    │                                     ▼                                       │
    │                              ┌──────────────┐                              │
    │                              │   Monitor    │◄─────────────────────────────┤
    │                              │ (Metrics)    │        Feedback Loop        │
    │                              └──────────────┘                              │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Champion/Challenger 패턴

### 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                   CHAMPION/CHALLENGER 패턴                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Champion (Production)                                          │
│  ├── 현재 프로덕션 트래픽 100% 처리                            │
│  ├── 검증된 성능 (F1: 0.85)                                    │
│  └── MLflow alias: "champion"                                   │
│                                                                 │
│  Challenger (Shadow)                                            │
│  ├── 프로덕션 트래픽 처리 안 함                                │
│  ├── Shadow 모드로 로깅만                                      │
│  ├── Champion과 동일 입력으로 평가                              │
│  └── MLflow alias: "challenger"                                 │
│                                                                 │
│  승격 조건:                                                     │
│  ├── F1 Score > Champion + 0.01 (1% 개선)                      │
│  ├── Precision >= Champion (정밀도 유지)                        │
│  └── 최소 1000건 평가 완료                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MLflow Model Registry 설정

```python
# mlops/model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class ModelVersion:
    name: str
    version: int
    alias: str
    run_id: str
    metrics: Dict[str, float]
    source: str

class ModelRegistry:
    """MLflow Model Registry 관리"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = "content-filter"
    
    def get_champion(self) -> Optional[ModelVersion]:
        """현재 Champion 모델 조회"""
        try:
            version = self.client.get_model_version_by_alias(
                name=self.model_name,
                alias="champion"
            )
            return self._to_model_version(version, "champion")
        except mlflow.exceptions.MlflowException:
            return None
    
    def get_challenger(self) -> Optional[ModelVersion]:
        """현재 Challenger 모델 조회"""
        try:
            version = self.client.get_model_version_by_alias(
                name=self.model_name,
                alias="challenger"
            )
            return self._to_model_version(version, "challenger")
        except mlflow.exceptions.MlflowException:
            return None
    
    def register_challenger(self, run_id: str) -> ModelVersion:
        """새 모델을 Challenger로 등록"""
        # 1. 모델 버전 생성
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, self.model_name)
        
        # 2. Challenger alias 설정
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="challenger",
            version=mv.version
        )
        
        return self._to_model_version(
            self.client.get_model_version(self.model_name, mv.version),
            "challenger"
        )
    
    def promote_challenger(self) -> bool:
        """
        Challenger를 Champion으로 승격
        
        Returns:
            True if promoted, False otherwise
        """
        challenger = self.get_challenger()
        if not challenger:
            return False
        
        # 1. 기존 Champion alias 제거 (있으면)
        champion = self.get_champion()
        if champion:
            self.client.delete_registered_model_alias(
                name=self.model_name,
                alias="champion"
            )
        
        # 2. Challenger를 Champion으로
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="champion",
            version=challenger.version
        )
        
        # 3. Challenger alias 제거
        self.client.delete_registered_model_alias(
            name=self.model_name,
            alias="challenger"
        )
        
        return True
    
    def reject_challenger(self):
        """Challenger 거부 (보관)"""
        challenger = self.get_challenger()
        if challenger:
            # alias 제거만 (버전은 유지)
            self.client.delete_registered_model_alias(
                name=self.model_name,
                alias="challenger"
            )
            # 태그로 거부 사유 기록
            self.client.set_model_version_tag(
                name=self.model_name,
                version=challenger.version,
                key="status",
                value="rejected"
            )
    
    def _to_model_version(self, mv, alias: str) -> ModelVersion:
        """MLflow ModelVersion을 내부 타입으로 변환"""
        run = self.client.get_run(mv.run_id)
        return ModelVersion(
            name=mv.name,
            version=int(mv.version),
            alias=alias,
            run_id=mv.run_id,
            metrics=run.data.metrics,
            source=mv.source
        )
```

### 비교 평가 시스템

```python
# mlops/evaluator.py
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class ChampionChallengerEvaluator:
    """Champion vs Challenger 비교 평가"""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        defender_service,  # API 클라이언트
        min_samples: int = 1000,
        improvement_threshold: float = 0.01
    ):
        self.registry = model_registry
        self.defender = defender_service
        self.min_samples = min_samples
        self.improvement_threshold = improvement_threshold
    
    def evaluate_on_test_set(
        self,
        test_data: List[Dict]  # [{"text": str, "label": int}, ...]
    ) -> Dict:
        """
        테스트셋으로 Champion/Challenger 비교
        """
        texts = [d["text"] for d in test_data]
        labels = [d["label"] for d in test_data]
        
        # 1. Champion 평가
        champion_preds = self._evaluate_model("champion", texts)
        champion_metrics = self._compute_metrics(champion_preds, labels)
        
        # 2. Challenger 평가
        challenger_preds = self._evaluate_model("challenger", texts)
        challenger_metrics = self._compute_metrics(challenger_preds, labels)
        
        # 3. 비교
        comparison = {
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "improvement": {
                k: challenger_metrics[k] - champion_metrics[k]
                for k in champion_metrics
            },
            "samples_evaluated": len(test_data)
        }
        
        # 4. 승격 판단
        comparison["should_promote"] = self._should_promote(
            champion_metrics, challenger_metrics
        )
        
        return comparison
    
    def _evaluate_model(self, alias: str, texts: List[str]) -> List[int]:
        """모델 추론 실행"""
        results = self.defender.classify_batch(texts, model_alias=alias)
        return [1 if r["is_toxic"] else 0 for r in results]
    
    def _compute_metrics(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Dict[str, float]:
        """메트릭 계산"""
        return {
            "f1": f1_score(labels, predictions, average="binary"),
            "precision": precision_score(labels, predictions, average="binary"),
            "recall": recall_score(labels, predictions, average="binary"),
            "accuracy": np.mean(np.array(predictions) == np.array(labels))
        }
    
    def _should_promote(
        self,
        champion: Dict[str, float],
        challenger: Dict[str, float]
    ) -> bool:
        """승격 여부 판단"""
        # 조건 1: F1 개선
        f1_improved = (
            challenger["f1"] - champion["f1"] > self.improvement_threshold
        )
        
        # 조건 2: Precision 유지 또는 개선
        precision_ok = challenger["precision"] >= champion["precision"] - 0.02
        
        return f1_improved and precision_ok


class ShadowEvaluator:
    """실시간 Shadow 평가 (프로덕션 트래픽 활용)"""
    
    def __init__(self, redis_client, model_registry: ModelRegistry):
        self.redis = redis_client
        self.registry = model_registry
        self.shadow_results_key = "shadow:results"
    
    def record_shadow_result(
        self,
        text: str,
        champion_result: Dict,
        challenger_result: Dict,
        ground_truth: int = None
    ):
        """Shadow 평가 결과 기록"""
        result = {
            "text_hash": hash(text),
            "champion_toxic": champion_result["is_toxic"],
            "champion_score": champion_result["toxic_score"],
            "challenger_toxic": challenger_result["is_toxic"],
            "challenger_score": challenger_result["toxic_score"],
            "ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis.lpush(self.shadow_results_key, json.dumps(result))
        self.redis.ltrim(self.shadow_results_key, 0, 10000)  # 최근 10000건 유지
    
    def get_shadow_comparison(self) -> Dict:
        """Shadow 평가 통계"""
        results = [
            json.loads(r) 
            for r in self.redis.lrange(self.shadow_results_key, 0, -1)
        ]
        
        if not results:
            return {"error": "No shadow results"}
        
        # 일치율 계산
        agreement = sum(
            1 for r in results 
            if r["champion_toxic"] == r["challenger_toxic"]
        ) / len(results)
        
        # Ground truth가 있는 경우 정확도 비교
        labeled = [r for r in results if r["ground_truth"] is not None]
        if labeled:
            champion_acc = sum(
                1 for r in labeled 
                if r["champion_toxic"] == r["ground_truth"]
            ) / len(labeled)
            challenger_acc = sum(
                1 for r in labeled 
                if r["challenger_toxic"] == r["ground_truth"]
            ) / len(labeled)
        else:
            champion_acc = challenger_acc = None
        
        return {
            "total_samples": len(results),
            "agreement_rate": agreement,
            "champion_accuracy": champion_acc,
            "challenger_accuracy": challenger_acc,
            "labeled_samples": len(labeled)
        }
```

---

## 🚀 자동 배포 파이프라인

### 배포 워크플로우

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTO DEPLOYMENT FLOW                        │
└─────────────────────────────────────────────────────────────────┘

  Training          MLflow           Evaluator         Deployer
     │                │                  │                │
     │  train_complete│                  │                │
     │───────────────►│                  │                │
     │                │                  │                │
     │                │  new_challenger  │                │
     │                │─────────────────►│                │
     │                │                  │                │
     │                │                  │ evaluate()     │
     │                │                  │───────────────►│
     │                │                  │                │
     │                │                  │ compare()      │
     │                │                  │◄───────────────│
     │                │                  │                │
     │                │                  │                │
     │                │   if better:     │                │
     │                │   promote()      │                │
     │                │◄─────────────────│                │
     │                │                  │                │
     │                │                  │  reload()      │
     │                │                  │───────────────►│
     │                │                  │                │
```

### 자동 배포 구현

```python
# mlops/deployer.py
import httpx
import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ModelDeployer:
    """모델 자동 배포 관리"""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        evaluator: ChampionChallengerEvaluator,
        inference_service_url: str = "http://localhost:8001",
        notification_webhook: Optional[str] = None
    ):
        self.registry = model_registry
        self.evaluator = evaluator
        self.inference_url = inference_service_url
        self.webhook = notification_webhook
    
    async def deploy_if_better(self, test_data: List[Dict]) -> Dict:
        """
        Challenger가 더 나으면 자동 배포
        """
        # 1. Challenger 존재 확인
        challenger = self.registry.get_challenger()
        if not challenger:
            return {"status": "no_challenger"}
        
        # 2. 비교 평가
        comparison = self.evaluator.evaluate_on_test_set(test_data)
        logger.info(f"Comparison result: {comparison}")
        
        # 3. 승격 판단
        if comparison["should_promote"]:
            return await self._promote_and_deploy(comparison)
        else:
            return await self._reject_challenger(comparison)
    
    async def _promote_and_deploy(self, comparison: Dict) -> Dict:
        """Challenger 승격 및 배포"""
        try:
            # 1. Registry에서 승격
            self.registry.promote_challenger()
            logger.info("Challenger promoted to Champion")
            
            # 2. Inference 서비스 핫 리로드
            await self._reload_inference_service()
            logger.info("Inference service reloaded")
            
            # 3. 알림 발송
            await self._send_notification({
                "event": "model_promoted",
                "improvement": comparison["improvement"],
                "new_champion_metrics": comparison["challenger"]
            })
            
            return {
                "status": "promoted",
                "comparison": comparison
            }
            
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return {
                "status": "promotion_failed",
                "error": str(e)
            }
    
    async def _reject_challenger(self, comparison: Dict) -> Dict:
        """Challenger 거부"""
        self.registry.reject_challenger()
        
        await self._send_notification({
            "event": "challenger_rejected",
            "reason": "Performance not improved",
            "comparison": comparison
        })
        
        return {
            "status": "rejected",
            "comparison": comparison
        }
    
    async def _reload_inference_service(self):
        """Inference 서비스 핫 리로드"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.inference_url}/reload",
                timeout=30.0
            )
            response.raise_for_status()
    
    async def _send_notification(self, payload: Dict):
        """알림 발송 (Slack/Discord 등)"""
        if not self.webhook:
            return
        
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook, json=payload)


class DeploymentPipeline:
    """전체 배포 파이프라인 오케스트레이션"""
    
    def __init__(
        self,
        trainer: QLoRATrainer,
        data_preparator: DatasetPreparator,
        registry: ModelRegistry,
        evaluator: ChampionChallengerEvaluator,
        deployer: ModelDeployer
    ):
        self.trainer = trainer
        self.data_prep = data_preparator
        self.registry = registry
        self.evaluator = evaluator
        self.deployer = deployer
    
    async def run_full_pipeline(self) -> Dict:
        """
        전체 MLOps 파이프라인 실행
        
        1. 데이터 준비
        2. 학습
        3. Challenger 등록
        4. 평가 & 배포 판단
        """
        logger.info("Starting full MLOps pipeline")
        
        # 1. 데이터 준비
        dataset = self.data_prep.prepare_training_data()
        logger.info(f"Dataset prepared: {len(dataset['train'])} train, {len(dataset['test'])} test")
        
        # 2. 학습
        run_id = self.trainer.train(
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"]
        )
        logger.info(f"Training completed: {run_id}")
        
        # 3. Challenger 등록
        self.registry.register_challenger(run_id)
        logger.info("Challenger registered")
        
        # 4. 평가 & 배포
        result = await self.deployer.deploy_if_better(dataset["test"])
        logger.info(f"Deployment result: {result['status']}")
        
        return result
```

---

## 📊 모니터링 & 알림

### Prometheus 메트릭 정의

```python
# mlops/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# 모델 정보
model_info = Info(
    "content_arena_model",
    "Current model information"
)

# 모델 버전
model_version = Gauge(
    "content_arena_model_version",
    "Current model version",
    ["alias"]  # champion, challenger
)

# F1 Score
model_f1_score = Gauge(
    "content_arena_model_f1_score",
    "Model F1 score",
    ["alias"]
)

# 추론 요청
inference_requests = Counter(
    "content_arena_inference_requests_total",
    "Total inference requests",
    ["model_alias", "result"]  # result: toxic, clean
)

# 추론 지연 시간
inference_latency = Histogram(
    "content_arena_inference_latency_seconds",
    "Inference latency in seconds",
    ["model_alias"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

# 배틀 결과
battle_detection_rate = Gauge(
    "content_arena_battle_detection_rate",
    "Current battle detection rate"
)

# 재학습 이벤트
retrain_events = Counter(
    "content_arena_retrain_events_total",
    "Total retrain events",
    ["trigger_reason", "result"]  # result: success, failed
)

# Champion 교체
champion_changes = Counter(
    "content_arena_champion_changes_total",
    "Total champion model changes"
)


class MetricsCollector:
    """메트릭 수집기"""
    
    def update_model_info(self, alias: str, version: int, f1: float):
        model_version.labels(alias=alias).set(version)
        model_f1_score.labels(alias=alias).set(f1)
    
    def record_inference(self, alias: str, is_toxic: bool, latency: float):
        result = "toxic" if is_toxic else "clean"
        inference_requests.labels(model_alias=alias, result=result).inc()
        inference_latency.labels(model_alias=alias).observe(latency)
    
    def update_detection_rate(self, rate: float):
        battle_detection_rate.set(rate)
    
    def record_retrain(self, reason: str, success: bool):
        result = "success" if success else "failed"
        retrain_events.labels(trigger_reason=reason, result=result).inc()
    
    def record_champion_change(self):
        champion_changes.inc()
```

### Grafana 대시보드 설정

```json
// grafana/dashboards/content-arena.json
{
  "title": "Content Arena MLOps",
  "panels": [
    {
      "title": "Detection Rate Over Time",
      "type": "timeseries",
      "targets": [
        {
          "expr": "content_arena_battle_detection_rate",
          "legendFormat": "Detection Rate"
        }
      ]
    },
    {
      "title": "Model F1 Score",
      "type": "gauge",
      "targets": [
        {
          "expr": "content_arena_model_f1_score{alias=\"champion\"}",
          "legendFormat": "Champion F1"
        },
        {
          "expr": "content_arena_model_f1_score{alias=\"challenger\"}",
          "legendFormat": "Challenger F1"
        }
      ]
    },
    {
      "title": "Inference Latency (p99)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, rate(content_arena_inference_latency_seconds_bucket[5m]))",
          "legendFormat": "p99 Latency"
        }
      ]
    },
    {
      "title": "Champion Changes",
      "type": "stat",
      "targets": [
        {
          "expr": "increase(content_arena_champion_changes_total[24h])",
          "legendFormat": "Changes (24h)"
        }
      ]
    },
    {
      "title": "Inference Requests by Result",
      "type": "piechart",
      "targets": [
        {
          "expr": "sum by (result) (increase(content_arena_inference_requests_total[1h]))",
          "legendFormat": "{{result}}"
        }
      ]
    }
  ]
}
```

### 알림 규칙

```yaml
# prometheus/alerts.yml
groups:
  - name: content-arena
    rules:
      # 탐지율 급락
      - alert: DetectionRateDrop
        expr: content_arena_battle_detection_rate < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Detection rate dropped below 50%"
          description: "Current detection rate: {{ $value }}"
      
      # 모델 F1 저하
      - alert: ModelF1ScoreDrop
        expr: content_arena_model_f1_score{alias="champion"} < 0.7
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Champion model F1 score dropped"
          description: "Current F1: {{ $value }}"
      
      # 추론 지연
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.99, rate(content_arena_inference_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "p99 latency: {{ $value }}s"
      
      # Challenger 장기 대기
      - alert: ChallengerStale
        expr: time() - content_arena_challenger_created_at > 86400
        for: 1h
        labels:
          severity: info
        annotations:
          summary: "Challenger model waiting for over 24 hours"
```

---

## 🔄 CI/CD 통합

### GitHub Actions 워크플로우

```yaml
# .github/workflows/mlops.yml
name: MLOps Pipeline

on:
  push:
    paths:
      - 'ml-service/**'
      - 'training/**'
  schedule:
    # 매일 새벽 2시 재학습 (선택적)
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force retrain regardless of threshold'
        type: boolean
        default: false

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          cd ml-service
          pip install -r requirements-test.txt
      - name: Run tests
        run: |
          cd ml-service
          pytest tests/

  train:
    needs: test
    runs-on: [self-hosted, gpu]  # GPU 러너 필요
    if: github.event_name == 'schedule' || github.event.inputs.force_retrain == 'true'
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r ml-service/requirements.txt
      - name: Run training
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python training/train.py
      - name: Evaluate and deploy
        run: |
          python training/evaluate_and_deploy.py

  notify:
    needs: train
    runs-on: ubuntu-latest
    if: always()
    steps:
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          fields: repo,message,commit,author
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 📁 파일 구조

```
mlops/
├── __init__.py
├── model_registry.py      # MLflow 모델 레지스트리 관리
├── evaluator.py           # Champion/Challenger 평가
├── deployer.py            # 자동 배포
├── metrics.py             # Prometheus 메트릭
├── alerting.py            # 알림 시스템
└── config.py              # MLOps 설정

training/
├── __init__.py
├── data_preparation.py    # 데이터셋 준비
├── qlora_trainer.py       # QLoRA 학습
├── auto_retrain.py        # 자동 재학습 트리거
├── train.py               # 학습 엔트리포인트
└── evaluate_and_deploy.py # 평가 & 배포 스크립트
```
