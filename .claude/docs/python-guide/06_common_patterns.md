# 06. Common Patterns & Tips

## 프로젝트 구조 (권장)

```
ml-project/
├── pyproject.toml
├── uv.lock
├── .python-version
├── configs/
│   └── train.yaml
├── src/
│   ├── data/dataset.py
│   ├── models/model.py
│   └── training/trainer.py
├── scripts/
│   ├── train.py
│   └── inference.py
└── tests/
```

## 설정 관리 (OmegaConf)

```yaml
# configs/train.yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  quantization: "4bit"
lora:
  r: 16
  alpha: 32
training:
  batch_size: 4
  learning_rate: 2e-4
```

```python
from omegaconf import OmegaConf
config = OmegaConf.load("configs/train.yaml")
print(config.model.name)
```

## 재현성 보장

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

## 메모리 디버깅

```python
import torch, gc

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU Memory: {allocated:.2f}GB")

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
```

## 학습 루프 템플릿

```python
from tqdm import tqdm

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

## 추론 템플릿

```python
@torch.inference_mode()
def generate(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 환경 변수 (.env)

```bash
HF_HOME=/data/huggingface
MLFLOW_TRACKING_URI=http://localhost:5000
CUDA_VISIBLE_DEVICES=0,1
```

```python
from dotenv import load_dotenv
load_dotenv()
```

## 자주 발생하는 문제

| 문제 | 해결책 |
|------|--------|
| CUDA OOM | batch↓, gradient_checkpointing, 4-bit |
| 느린 학습 | num_workers↑, torch.compile() |
| 재현 불가 | seed 고정, uv.lock 커밋 |
| 모델 로드 실패 | trust_remote_code=True |

## DevOps 체크리스트

```
□ uv.lock이 git에 커밋되어 있는가?
□ Python 버전이 .python-version에 고정되어 있는가?
□ 모든 실험이 MLflow에 기록되는가?
□ seed가 고정되어 재현 가능한가?
□ CI/CD에서 테스트가 실행되는가?
```

## 유용한 명령어

```bash
# GPU 모니터링
watch -n 1 nvidia-smi

# MLflow UI
mlflow ui --port 5000

# 캐시 정리
rm -rf ~/.cache/huggingface/hub
```
