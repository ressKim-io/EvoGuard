# 04. Quantization with bitsandbytes

## 양자화 개요

양자화는 모델 가중치를 낮은 정밀도로 표현하여 메모리 사용량을 줄입니다.

| 정밀도 | 메모리 (7B 모델) | 용도 |
|--------|-----------------|------|
| FP32 | ~28GB | 기준 |
| FP16/BF16 | ~14GB | 학습/추론 |
| INT8 | ~7GB | 추론 |
| INT4 (NF4) | ~3.5GB | QLoRA 학습/추론 |

## 4-bit 양자화 (권장)

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4가 FP4보다 좋음
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 정밀도
    bnb_4bit_use_double_quant=True,     # 추가 0.4bit 절약
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

## 8-bit 양자화

```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # outlier 임계값
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
```

## 특정 모듈 제외

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["lm_head"],  # 출력층 제외
)
```

## QLoRA 전체 설정

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# 1. 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# 3. k-bit 학습 준비
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 4. LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

## CPU Offload (VRAM 부족 시)

```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

# device_map으로 레이어 분배
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": "cpu",  # CPU로 오프로드
    # ...
    "lm_head": 0,
}
```

## 메모리 확인

```python
# 모델 메모리 사용량
print(f"Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

# GPU 메모리
print(f"GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Max: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Hub에 양자화 모델 Push

```python
# 양자화 설정 포함하여 저장
model.push_to_hub("my-quantized-model")

# 로드 시 자동 적용
model = AutoModelForCausalLM.from_pretrained(
    "username/my-quantized-model",
    device_map="auto",
)
```

## 양자화 선택 가이드

```
추론만 필요?
├─ Yes → 8-bit (빠름, 정확도 좋음)
└─ No (학습 필요)
   ├─ VRAM 충분 (>24GB) → FP16/BF16 + LoRA
   └─ VRAM 부족 (<24GB) → 4-bit QLoRA
```

## 주의사항

- 4-bit 양자화 후 **순수 학습 불가** → LoRA 필수
- `bnb_4bit_compute_dtype`은 반드시 bfloat16 권장
- 8-bit이 4-bit보다 추론 속도 빠름
- `device_map="auto"`는 학습 시 사용 금지
- CUDA 11.8+ 필요, 12.x 권장
