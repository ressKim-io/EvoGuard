# 🤖 04. ML 파이프라인 상세

> 공격자 모델, 방어자 모델, 학습 파이프라인의 상세 구현

---

## 📊 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ML PIPELINE OVERVIEW                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   데이터셋    │───►│   전처리     │───►│   학습       │───►│   평가       │
│  (Jigsaw +   │    │  (정제,      │    │  (QLoRA      │    │  (F1, AUC,   │
│   Battle)    │    │   토큰화)    │    │   Fine-tune) │    │   정확도)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                               ┌────────────────────┘
                                               ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   배포       │◄───│  비교 평가   │◄───│  모델 등록   │
│  (Champion   │    │  (Champion   │    │  (MLflow     │
│   교체)      │    │   vs         │    │   Registry)  │
│              │    │   Challenger)│    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 🎯 공격자 (Attacker) 파이프라인

### 1. 공격 전략 아키텍처

```python
# attacker/strategies/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class EvasionResult:
    original: str
    evasion: str
    strategy: str
    confidence: float  # 우회 성공 예상 확률

class AttackStrategy(ABC):
    """공격 전략 베이스 클래스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def generate(self, text: str, num_variants: int = 1) -> List[EvasionResult]:
        """
        주어진 텍스트에 대해 우회 변형을 생성
        
        Args:
            text: 원본 유해 텍스트
            num_variants: 생성할 변형 수
            
        Returns:
            EvasionResult 리스트
        """
        pass
```

### 2. 규칙 기반 전략들

```python
# attacker/strategies/rule_based.py
import random
import re
from typing import List

class UnicodeEvasionStrategy(AttackStrategy):
    """유니코드 문자 변형 전략"""
    
    name = "unicode_evasion"
    
    # 한글 자음/모음 분리 매핑
    JAMO_MAP = {
        '가': 'ㄱㅏ', '나': 'ㄴㅏ', '다': 'ㄷㅏ',
        '바': 'ㅂㅏ', '사': 'ㅅㅏ', '자': 'ㅈㅏ',
        # ... 확장
    }
    
    # 유사 문자 매핑
    SIMILAR_CHARS = {
        'a': ['а', 'ɑ', 'α'],  # 키릴, IPA, 그리스
        'e': ['е', 'ε', 'ɛ'],
        'o': ['о', 'ο', '0'],
        'i': ['і', 'ι', '1', 'l'],
        # ... 확장
    }
    
    def generate(self, text: str, num_variants: int = 5) -> List[EvasionResult]:
        results = []
        
        for _ in range(num_variants):
            evasion = self._apply_random_transform(text)
            results.append(EvasionResult(
                original=text,
                evasion=evasion,
                strategy=self.name,
                confidence=0.6  # 규칙 기반은 보통 성공률
            ))
        
        return results
    
    def _apply_random_transform(self, text: str) -> str:
        transforms = [
            self._space_insertion,
            self._jamo_decompose,
            self._similar_char_replace,
            self._zero_width_insert,
        ]
        
        # 1-3개의 변형을 무작위 적용
        num_transforms = random.randint(1, 3)
        for transform in random.sample(transforms, num_transforms):
            text = transform(text)
        
        return text
    
    def _space_insertion(self, text: str) -> str:
        """글자 사이에 공백 삽입: 바보 → 바 보"""
        chars = list(text)
        for i in range(len(chars) - 1, 0, -1):
            if random.random() < 0.3:
                chars.insert(i, ' ')
        return ''.join(chars)
    
    def _jamo_decompose(self, text: str) -> str:
        """한글 자모 분리: 바보 → ㅂㅏㅂㅗ"""
        result = []
        for char in text:
            if char in self.JAMO_MAP and random.random() < 0.5:
                result.append(self.JAMO_MAP[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def _similar_char_replace(self, text: str) -> str:
        """유사 문자 치환: hello → hеllo (키릴 'е')"""
        result = []
        for char in text.lower():
            if char in self.SIMILAR_CHARS and random.random() < 0.3:
                result.append(random.choice(self.SIMILAR_CHARS[char]))
            else:
                result.append(char)
        return ''.join(result)
    
    def _zero_width_insert(self, text: str) -> str:
        """보이지 않는 문자 삽입"""
        zero_widths = ['\u200b', '\u200c', '\u200d', '\ufeff']
        chars = list(text)
        for i in range(len(chars) - 1, 0, -1):
            if random.random() < 0.2:
                chars.insert(i, random.choice(zero_widths))
        return ''.join(chars)


class HomoglyphStrategy(AttackStrategy):
    """동형 문자(Homoglyph) 치환 전략"""
    
    name = "homoglyph"
    
    # 시각적으로 유사한 문자 매핑 (더 광범위)
    HOMOGLYPHS = {
        'A': ['Α', 'А', 'Ꭺ'],
        'B': ['Β', 'В', 'Ᏼ'],
        'C': ['С', 'Ϲ', 'Ꮯ'],
        # ... 전체 알파벳
    }
    
    def generate(self, text: str, num_variants: int = 5) -> List[EvasionResult]:
        # 구현
        pass


class LeetSpeakStrategy(AttackStrategy):
    """리트스피크 변형: hello → h3ll0"""
    
    name = "leetspeak"
    
    LEET_MAP = {
        'a': ['4', '@'],
        'e': ['3'],
        'i': ['1', '!'],
        'o': ['0'],
        's': ['5', '$'],
        't': ['7'],
        # ...
    }
    
    def generate(self, text: str, num_variants: int = 5) -> List[EvasionResult]:
        # 구현
        pass
```

### 3. LLM 기반 전략

```python
# attacker/strategies/llm_based.py
import httpx
from typing import List

class LLMEvasionStrategy(AttackStrategy):
    """Ollama LLM을 이용한 창의적 우회"""
    
    name = "llm_evasion"
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "mistral:7b-instruct-v0.2-q4_K_S"
    
    def generate(self, text: str, num_variants: int = 5) -> List[EvasionResult]:
        results = []
        
        for _ in range(num_variants):
            prompt = self._build_prompt(text)
            response = self._call_ollama(prompt)
            
            if response:
                results.append(EvasionResult(
                    original=text,
                    evasion=response,
                    strategy=self.name,
                    confidence=0.8  # LLM은 더 창의적
                ))
        
        return results
    
    def _build_prompt(self, text: str) -> str:
        return f"""당신은 콘텐츠 필터 테스트를 위한 연구원입니다.
다음 텍스트의 의미는 유지하면서, 자동화된 욕설 필터를 우회할 수 있는 변형을 생성하세요.

원본 텍스트: "{text}"

변형 규칙:
1. 의미는 동일하게 유지
2. 특수문자, 이모지, 유니코드 활용 가능
3. 발음이 비슷한 다른 단어로 대체 가능
4. 띄어쓰기, 줄바꿈 활용 가능

변형된 텍스트만 출력하세요 (설명 없이):"""
    
    def _call_ollama(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,  # 창의성 높게
                        "top_p": 0.95,
                        "max_tokens": 100
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Ollama 호출 실패: {e}")
            return None


class AdversarialLLMStrategy(AttackStrategy):
    """방어 모델의 약점을 학습하는 적대적 전략"""
    
    name = "adversarial_llm"
    
    def __init__(self, ollama_url: str, failed_evasions: List[str] = None):
        self.ollama_url = ollama_url
        self.failed_evasions = failed_evasions or []  # 탐지된 패턴들
    
    def generate(self, text: str, num_variants: int = 5) -> List[EvasionResult]:
        # 이전에 탐지된 패턴을 피하면서 새로운 변형 생성
        prompt = self._build_adversarial_prompt(text)
        # ...
    
    def _build_adversarial_prompt(self, text: str) -> str:
        failed_examples = "\n".join([f"- {e}" for e in self.failed_evasions[-10:]])
        
        return f"""이전에 탐지된 우회 패턴들:
{failed_examples}

위 패턴들은 모두 탐지되었습니다.
다음 텍스트를 완전히 새로운 방식으로 변형하세요.

원본: "{text}"
새로운 변형:"""
```

### 4. 공격자 오케스트레이터

```python
# attacker/orchestrator.py
from typing import List, Dict
import random

class AttackerOrchestrator:
    """여러 공격 전략을 조합하여 실행"""
    
    def __init__(self, strategies: List[AttackStrategy]):
        self.strategies = {s.name: s for s in strategies}
    
    def attack(
        self, 
        text: str, 
        strategy: str = None,
        num_variants: int = 10
    ) -> List[EvasionResult]:
        """
        공격 실행
        
        Args:
            text: 원본 텍스트
            strategy: 특정 전략 지정 (None이면 무작위)
            num_variants: 생성할 변형 수
        """
        if strategy:
            return self.strategies[strategy].generate(text, num_variants)
        
        # 전략 조합
        results = []
        per_strategy = num_variants // len(self.strategies) + 1
        
        for s in self.strategies.values():
            results.extend(s.generate(text, per_strategy))
        
        return results[:num_variants]
    
    def evolve_strategy(self, battle_results: List[Dict]):
        """
        배틀 결과를 분석하여 전략 가중치 조정
        (어떤 전략이 더 효과적인지 학습)
        """
        success_by_strategy = {}
        
        for result in battle_results:
            strategy = result["attack_strategy"]
            detected = result["is_detected"]
            
            if strategy not in success_by_strategy:
                success_by_strategy[strategy] = {"success": 0, "total": 0}
            
            success_by_strategy[strategy]["total"] += 1
            if not detected:  # 탐지 안 됨 = 우회 성공
                success_by_strategy[strategy]["success"] += 1
        
        # 성공률 계산 및 로깅
        for strategy, stats in success_by_strategy.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"Strategy {strategy}: {success_rate:.2%} evasion rate")
```

---

## 🛡️ 방어자 (Defender) 파이프라인

### 1. 모델 아키텍처

```python
# defender/model.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

class ContentFilter:
    """콘텐츠 필터 모델"""
    
    def __init__(
        self, 
        base_model: str = "bert-base-multilingual-cased",
        lora_weights: str = None,
        device: str = "cuda"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # 기본 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2  # 0: clean, 1: toxic
        )
        
        # LoRA 가중치가 있으면 적용
        if lora_weights:
            self.model = PeftModel.from_pretrained(self.model, lora_weights)
            self.model = self.model.merge_and_unload()  # 추론 최적화
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def classify(self, text: str) -> Dict:
        """
        단일 텍스트 분류
        
        Returns:
            {
                "toxic_score": 0.85,
                "is_toxic": True,
                "confidence": 0.85
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        toxic_score = probs[0][1].item()
        
        return {
            "toxic_score": toxic_score,
            "is_toxic": toxic_score > 0.5,
            "confidence": max(probs[0]).item()
        }
    
    @torch.no_grad()
    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """배치 분류"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            for j, prob in enumerate(probs):
                toxic_score = prob[1].item()
                results.append({
                    "toxic_score": toxic_score,
                    "is_toxic": toxic_score > 0.5,
                    "confidence": max(prob).item()
                })
        
        return results
```

### 2. 추론 API 서버

```python
# defender/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import mlflow

app = FastAPI(title="Content Filter API")

# 모델 인스턴스 (싱글톤)
filter_model = None

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class ClassifyBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class ClassifyResponse(BaseModel):
    toxic_score: float
    is_toxic: bool
    confidence: float
    model_version: str

@app.on_event("startup")
async def load_model():
    """서버 시작 시 Champion 모델 로드"""
    global filter_model
    
    # MLflow에서 Champion 모델 로드
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(
        name="content-filter",
        alias="champion"
    )
    
    filter_model = ContentFilter(
        base_model="bert-base-multilingual-cased",
        lora_weights=model_version.source
    )

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """단일 텍스트 분류"""
    result = filter_model.classify(request.text)
    result["model_version"] = get_current_version()
    return result

@app.post("/classify/batch", response_model=List[ClassifyResponse])
async def classify_batch(request: ClassifyBatchRequest):
    """배치 텍스트 분류"""
    results = filter_model.classify_batch(request.texts)
    version = get_current_version()
    for r in results:
        r["model_version"] = version
    return results

@app.post("/reload")
async def reload_model():
    """Champion 모델 재로드 (핫 리로드)"""
    global filter_model
    await load_model()
    return {"status": "reloaded"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": filter_model is not None}
```

---

## 📚 학습 파이프라인

### 1. 데이터셋 준비

```python
# training/data_preparation.py
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from typing import List, Dict

class DatasetPreparator:
    """학습 데이터셋 준비"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def prepare_training_data(self) -> Dataset:
        """
        학습 데이터 통합
        1. Jigsaw 베이스 데이터셋
        2. Battle에서 수집된 데이터
        """
        # 1. Jigsaw 데이터셋 (베이스)
        jigsaw = self._load_jigsaw_dataset()
        
        # 2. Battle 수집 데이터
        battle_data = self._load_battle_data()
        
        # 3. 통합
        combined = concatenate_datasets([jigsaw, battle_data])
        
        # 4. 셔플 및 분할
        combined = combined.shuffle(seed=42)
        split = combined.train_test_split(test_size=0.1)
        
        return split
    
    def _load_jigsaw_dataset(self) -> Dataset:
        """Jigsaw Toxic Comment Dataset 로드"""
        # Kaggle에서 다운로드 필요
        # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
        
        df = pd.read_csv("data/jigsaw_train.csv")
        
        # toxic 컬럼들을 하나로 통합
        df["label"] = (df[["toxic", "severe_toxic", "obscene", 
                          "threat", "insult", "identity_hate"]].sum(axis=1) > 0).astype(int)
        
        return Dataset.from_pandas(df[["comment_text", "label"]].rename(
            columns={"comment_text": "text"}
        ))
    
    def _load_battle_data(self) -> Dataset:
        """Battle에서 수집된 우회 패턴 데이터"""
        # 우회 성공한 패턴 = toxic으로 레이블링
        query = """
            SELECT evasion_text as text, 1 as label
            FROM battle_rounds
            WHERE is_detected = false
            
            UNION ALL
            
            SELECT evasion_text as text, 1 as label
            FROM battle_rounds
            WHERE is_detected = true
        """
        
        df = pd.read_sql(query, self.db)
        
        if len(df) == 0:
            return Dataset.from_dict({"text": [], "label": []})
        
        return Dataset.from_pandas(df)
    
    def _augment_data(self, dataset: Dataset) -> Dataset:
        """데이터 증강 (선택적)"""
        # 백번역, 동의어 치환 등
        pass


def prepare_tokenized_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 512
) -> Dataset:
    """토크나이징된 데이터셋 생성"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    
    return tokenized
```

### 2. QLoRA 학습

```python
# training/qlora_trainer.py
import torch
import mlflow
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

class QLoRATrainer:
    """QLoRA Fine-tuning 파이프라인"""
    
    def __init__(
        self,
        base_model: str = "bert-base-multilingual-cased",
        output_dir: str = "./results",
        mlflow_experiment: str = "content-filter-training"
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.mlflow_experiment = mlflow_experiment
        
        # MLflow 설정
        mlflow.set_experiment(mlflow_experiment)
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        lora_r: int = 16,
        lora_alpha: int = 32,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-4
    ) -> str:
        """
        QLoRA Fine-tuning 실행
        
        Returns:
            MLflow run_id
        """
        with mlflow.start_run() as run:
            # 파라미터 로깅
            mlflow.log_params({
                "base_model": self.base_model,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset)
            })
            
            # 1. 4-bit 양자화 설정
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            # 2. 모델 & 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                num_labels=2
            )
            
            # 3. k-bit 학습 준비
            model = prepare_model_for_kbit_training(model)
            
            # 4. LoRA 설정
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value", "key", "dense"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS"
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # 5. 학습 설정
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size * 2,
                gradient_accumulation_steps=8,
                learning_rate=learning_rate,
                warmup_ratio=0.1,
                
                # 메모리 최적화
                bf16=True,
                optim="adamw_8bit",
                gradient_checkpointing=True,
                
                # 평가 & 로깅
                eval_strategy="steps",
                eval_steps=100,
                logging_steps=10,
                save_strategy="steps",
                save_steps=100,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                
                # MLflow
                report_to="mlflow"
            )
            
            # 6. Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=self._compute_metrics
            )
            
            # 7. 학습 실행
            trainer.train()
            
            # 8. 최종 평가
            eval_results = trainer.evaluate()
            mlflow.log_metrics({
                f"final_{k}": v for k, v in eval_results.items()
            })
            
            # 9. 모델 저장
            model_path = f"{self.output_dir}/final_model"
            trainer.save_model(model_path)
            
            # 10. MLflow에 모델 등록
            mlflow.peft.log_model(
                model,
                artifact_path="model",
                registered_model_name="content-filter"
            )
            
            return run.info.run_id
    
    def _compute_metrics(self, eval_pred):
        """평가 메트릭 계산"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="binary"),
            "precision": precision_score(labels, predictions, average="binary"),
            "recall": recall_score(labels, predictions, average="binary")
        }
```

### 3. 자동 재학습 트리거

```python
# training/auto_retrain.py
import redis
from datetime import datetime
import threading
import time

class AutoRetrainTrigger:
    """자동 재학습 트리거 시스템"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        trainer: QLoRATrainer,
        data_preparator: DatasetPreparator,
        evasion_threshold: float = 0.3,  # 30% 우회율 넘으면 재학습
        min_new_samples: int = 100       # 최소 새 샘플 수
    ):
        self.redis = redis_client
        self.trainer = trainer
        self.data_preparator = data_preparator
        self.evasion_threshold = evasion_threshold
        self.min_new_samples = min_new_samples
        
        self._running = False
        self._lock_key = "lock:training"
    
    def start_monitoring(self):
        """이벤트 모니터링 시작"""
        self._running = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
    
    def _monitor_loop(self):
        """이벤트 큐 모니터링"""
        pubsub = self.redis.pubsub()
        pubsub.subscribe("battle_completed")
        
        for message in pubsub.listen():
            if not self._running:
                break
            
            if message["type"] == "message":
                battle_id = message["data"]
                self._check_retrain_condition(battle_id)
    
    def _check_retrain_condition(self, battle_id: str):
        """재학습 조건 확인"""
        # 1. 최근 배틀 통계 조회
        stats = self._get_recent_stats()
        
        # 2. 조건 확인
        if stats["evasion_rate"] > self.evasion_threshold:
            if stats["new_samples"] >= self.min_new_samples:
                self._trigger_retrain(
                    reason=f"High evasion rate: {stats['evasion_rate']:.2%}"
                )
    
    def _get_recent_stats(self) -> Dict:
        """최근 배틀 통계"""
        # Redis 또는 DB에서 조회
        pass
    
    def _trigger_retrain(self, reason: str):
        """재학습 실행"""
        # 분산 락 획득
        if not self._acquire_lock():
            print("Another training in progress, skipping")
            return
        
        try:
            print(f"Triggering retrain: {reason}")
            
            # 1. 데이터 준비
            dataset = self.data_preparator.prepare_training_data()
            
            # 2. 학습 실행
            run_id = self.trainer.train(
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"]
            )
            
            # 3. Challenger로 등록
            self._register_as_challenger(run_id)
            
            print(f"Training completed: {run_id}")
            
        finally:
            self._release_lock()
    
    def _acquire_lock(self) -> bool:
        """분산 락 획득"""
        return self.redis.set(
            self._lock_key, 
            "training", 
            nx=True, 
            ex=3600  # 1시간 타임아웃
        )
    
    def _release_lock(self):
        """분산 락 해제"""
        self.redis.delete(self._lock_key)
    
    def _register_as_challenger(self, run_id: str):
        """새 모델을 Challenger로 등록"""
        client = mlflow.tracking.MlflowClient()
        
        # 최신 버전 가져오기
        versions = client.search_model_versions(
            f"name='content-filter' and run_id='{run_id}'"
        )
        
        if versions:
            client.set_registered_model_alias(
                name="content-filter",
                alias="challenger",
                version=versions[0].version
            )
```

---

## 📈 평가 메트릭

### 수집할 메트릭

```python
# 1. 분류 성능 메트릭
metrics = {
    "accuracy": 0.92,
    "f1_score": 0.87,
    "precision": 0.85,
    "recall": 0.89,
    "auc_roc": 0.94
}

# 2. 배틀 성능 메트릭
battle_metrics = {
    "detection_rate": 0.75,    # 탐지율
    "evasion_rate": 0.25,      # 우회율
    "false_positive_rate": 0.08,
    "false_negative_rate": 0.17
}

# 3. 라운드별 추이
round_metrics = [
    {"round": 1, "detection_rate": 0.60},
    {"round": 2, "detection_rate": 0.65},
    {"round": 3, "detection_rate": 0.72},
    # ...
]
```

### 평가 파이프라인

```python
# training/evaluation.py

class ModelEvaluator:
    """모델 평가 파이프라인"""
    
    def evaluate_model(
        self,
        model: ContentFilter,
        test_dataset: Dataset
    ) -> Dict:
        """
        모델 종합 평가
        """
        # 1. 기본 분류 평가
        predictions = model.classify_batch([ex["text"] for ex in test_dataset])
        labels = [ex["label"] for ex in test_dataset]
        
        # 2. 메트릭 계산
        metrics = self._compute_classification_metrics(predictions, labels)
        
        # 3. 우회 패턴 평가 (선택적)
        evasion_metrics = self._evaluate_evasion_resistance(model)
        
        return {**metrics, **evasion_metrics}
    
    def compare_models(
        self,
        champion: ContentFilter,
        challenger: ContentFilter,
        test_dataset: Dataset
    ) -> Dict:
        """
        Champion vs Challenger 비교
        """
        champion_metrics = self.evaluate_model(champion, test_dataset)
        challenger_metrics = self.evaluate_model(challenger, test_dataset)
        
        comparison = {
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "improvement": {
                k: challenger_metrics[k] - champion_metrics[k]
                for k in champion_metrics.keys()
            },
            "should_promote": challenger_metrics["f1"] > champion_metrics["f1"]
        }
        
        return comparison
```
