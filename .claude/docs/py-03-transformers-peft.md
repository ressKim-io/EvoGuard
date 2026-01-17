# 03. Transformers & PEFT Guide

## Transformers 기본 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # 추론 전용!
)
```

## PEFT (Parameter-Efficient Fine-Tuning)

전체 모델 대신 일부 파라미터만 학습하여 메모리 절약.

### LoRA 설정

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                      # rank (8-64 권장)
    lora_alpha=32,             # scaling factor
    lora_dropout=0.1,
    target_modules=[           # 적용할 레이어
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 6.5M || all params: 8B || 0.08%
```

### QLoRA (4-bit + LoRA)

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# k-bit 학습 준비
model = prepare_model_for_kbit_training(model)

# LoRA 적용
model = get_peft_model(model, lora_config)
```

## Dataset 준비

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data.jsonl")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, num_proc=4)
```

## Trainer로 학습

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,                    # bfloat16 사용
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch_fused",    # Fused optimizer
    gradient_checkpointing=True,  # 메모리 절약
    report_to="mlflow",           # MLflow 연동
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

## 모델 저장/로드

```python
# LoRA 어댑터만 저장 (작은 용량)
model.save_pretrained("./lora-adapter")

# 로드
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# 병합 (선택사항)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

## 추론 최적화

```python
# Flash Attention 2 사용
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# 생성
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    use_cache=True,
)
```

## LoRA Hyperparameter 가이드

| 파라미터 | 작은 모델 (<7B) | 큰 모델 (>13B) |
|---------|----------------|----------------|
| r | 8-16 | 16-64 |
| lora_alpha | r * 2 | r * 2 |
| dropout | 0.05-0.1 | 0.05 |
| target_modules | attention만 | attention + MLP |

## 주의사항

- `device_map="auto"`는 학습 시 사용 금지 (추론 전용)
- 학습 시 명시적으로 `.to(device)` 사용
- `gradient_checkpointing`은 메모리↓, 속도↓ 트레이드오프
- Flash Attention 2는 Ampere 이상 GPU 필요
