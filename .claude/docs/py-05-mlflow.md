# 05. MLflow Experiment Tracking

## MLflow 개요

ML 실험의 파라미터, 메트릭, 모델을 추적하는 플랫폼.

```bash
uv add mlflow
mlflow ui --port 5000  # http://localhost:5000
```

## 기본 사용법

```python
import mlflow

mlflow.set_experiment("my-llm-finetuning")

with mlflow.start_run(run_name="qlora-llama3-v1"):
    # 파라미터 로깅
    mlflow.log_params({
        "model": "Llama-3.1-8B",
        "lora_r": 16,
        "learning_rate": 2e-4,
    })
    
    # 학습 루프
    for epoch in range(3):
        train_loss = train_one_epoch()
        mlflow.log_metrics({"train_loss": train_loss}, step=epoch)
    
    # 모델 저장
    mlflow.pytorch.log_model(model, "model")
```

## PyTorch Manual Logging

```python
import mlflow
import torch.nn as nn

mlflow.set_experiment("pytorch-training")

with mlflow.start_run():
    params = {"hidden_size": 512, "lr": 1e-3, "epochs": 10}
    mlflow.log_params(params)
    
    model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
    
    for epoch in range(params["epochs"]):
        loss = train_epoch(model, train_loader)
        mlflow.log_metrics({"loss": loss}, step=epoch)
    
    mlflow.pytorch.log_model(model, "model")
```

## Transformers Trainer 연동

```python
from transformers import TrainingArguments, Trainer
import mlflow

training_args = TrainingArguments(
    output_dir="./outputs",
    report_to="mlflow",  # MLflow 자동 연동
    logging_steps=10,
)

with mlflow.start_run():
    mlflow.log_params({"model": model_id, "lora_r": 16})
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
```

## Artifact 저장

```python
with mlflow.start_run():
    mlflow.log_artifact("config.yaml")              # 파일
    mlflow.log_artifacts("./outputs", "outputs")    # 디렉토리
    mlflow.log_dict({"accuracy": 0.95}, "result.json")
```

## 모델 Registry

```python
# 등록
mlflow.pytorch.log_model(model, "model", registered_model_name="llama3-ft")

# 로드
model = mlflow.pytorch.load_model("models:/llama3-ft/1")
# 또는
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
```

## 실험 검색

```python
experiment = mlflow.get_experiment_by_name("my-experiment")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.val_loss < 0.5",
    order_by=["metrics.val_loss ASC"],
)
best_run = runs.iloc[0]
```

## Tag 활용

```python
with mlflow.start_run():
    mlflow.set_tags({
        "model_type": "causal_lm",
        "quantization": "4bit",
    })
```

## Best Practices

1. 실험명: `{project}-{task}-{date}` 형식
2. run_name: 하이퍼파라미터 요약 포함
3. step 지정: 시계열 메트릭에 필수
4. artifact 저장: config, 체크포인트 포함

## 주의사항

- `mlflow.pytorch.autolog()`는 **Lightning 전용**
- vanilla PyTorch는 manual logging 필요
- 대용량 artifact는 S3/GCS 백엔드 권장
