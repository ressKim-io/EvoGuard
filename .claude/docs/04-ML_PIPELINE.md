# ML 파이프라인

> 공격자/방어자 모델 및 학습 파이프라인 개요

## 파이프라인 개요

```
데이터셋 → 전처리 → 학습 (QLoRA) → 평가 → 모델 등록 → Champion/Challenger 비교 → 배포
```

**관련 문서**:
- `py-03-transformers-peft.md` - Transformers & PEFT
- `py-04-quantization.md` - bitsandbytes 양자화
- `py-05-mlflow.md` - MLflow 실험 추적

## 공격자 (Attacker) 파이프라인

### 전략 아키텍처

```
attacker/
├── strategies/
│   ├── base.py              # AttackStrategy 추상 클래스
│   ├── rule_based.py        # 규칙 기반 전략
│   └── llm_based.py         # LLM 기반 전략
└── orchestrator.py          # 전략 조합 및 실행
```

### 공격 전략 종류

| 전략 | 설명 | 예시 |
|------|------|------|
| `unicode_evasion` | 유니코드 문자 변형 | 바보 → ㅂㅏㅂㅗ |
| `homoglyph` | 동형 문자 치환 | hello → hеllo (키릴 е) |
| `leetspeak` | 리트스피크 | hello → h3ll0 |
| `llm_evasion` | Ollama LLM 창의적 우회 | 문맥 유지하며 변형 |
| `adversarial_llm` | 방어 모델 약점 학습 | 탐지 실패 패턴 분석 |

### 핵심 인터페이스

```python
# attacker/strategies/base.py
@dataclass
class EvasionResult:
    original: str
    evasion: str
    strategy: str
    confidence: float

class AttackStrategy(ABC):
    @abstractmethod
    def generate(self, text: str, num_variants: int = 1) -> List[EvasionResult]:
        pass
```

### 규칙 기반 변형 기법

1. **공백 삽입**: `바보` → `바 보`
2. **자모 분리**: `바보` → `ㅂㅏㅂㅗ`
3. **유사 문자 치환**: `a` → `α` (그리스), `а` (키릴)
4. **Zero-width 삽입**: 보이지 않는 유니코드 삽입

### LLM 기반 전략

```python
# Ollama API 호출
prompt = f"""다음 텍스트의 의미는 유지하면서 욕설 필터를 우회할 수 있는 변형을 생성하세요.
원본: "{text}"
변형된 텍스트만 출력:"""

response = httpx.post(f"{ollama_url}/api/generate", json={
    "model": "mistral:7b-instruct-v0.2-q4_K_S",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.9}
})
```

## 방어자 (Defender) 파이프라인

### 모델 아키텍처

```
defender/
├── model.py      # ContentFilter 클래스
└── api.py        # FastAPI 추론 서버
```

**베이스 모델**: `bert-base-multilingual-cased`
**Fine-tuning**: QLoRA (4-bit 양자화 + LoRA)

### 핵심 인터페이스

```python
# defender/model.py
class ContentFilter:
    def __init__(self, base_model: str, lora_weights: str = None):
        # BERT + LoRA 어댑터 로드
        pass

    def classify(self, text: str) -> Dict:
        """단일 텍스트 분류"""
        return {
            "toxic_score": 0.85,
            "is_toxic": True,
            "confidence": 0.85
        }

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """배치 분류"""
        pass
```

### 추론 API

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/classify` | POST | 단일 텍스트 분류 |
| `/classify/batch` | POST | 배치 분류 |
| `/reload` | POST | 모델 핫 리로드 |
| `/health` | GET | 헬스 체크 |

상세: `06-API_SPEC.md`

## 학습 파이프라인

### 데이터셋 구성

| 소스 | 설명 | 라벨링 |
|------|------|--------|
| Jigsaw Toxic | 베이스 데이터셋 | toxic 컬럼 통합 |
| Battle 수집 | 우회 성공 패턴 | 모두 toxic=1 |

### QLoRA 학습 설정

```python
# 4-bit 양자화 (8GB VRAM에서 7B 모델 학습 가능)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key", "dense"],
    lora_dropout=0.05,
    task_type="SEQ_CLS"
)
```

상세: `py-03-transformers-peft.md`, `py-04-quantization.md`

### 학습 파라미터 (8GB VRAM)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| batch_size | 2 | 작은 배치 |
| gradient_accumulation | 8 | 실효 배치 = 16 |
| bf16 | True | 4060Ti 최적화 |
| optim | adamw_8bit | 메모리 절약 |
| gradient_checkpointing | True | 메모리-속도 트레이드오프 |

### MLflow 연동

```python
with mlflow.start_run():
    mlflow.log_params({...})
    trainer.train()
    mlflow.log_metrics({...})
    mlflow.peft.log_model(model, "model", registered_model_name="content-filter")
```

상세: `py-05-mlflow.md`

## 자동 재학습 트리거

### 트리거 조건

| 조건 | 임계값 | 설명 |
|------|--------|------|
| 우회율 | > 30% | 최근 배틀 기준 |
| 새 샘플 수 | >= 100 | 충분한 학습 데이터 |

### 재학습 플로우

```
battle_completed 이벤트 → 우회율 체크 → 임계값 초과 → 분산 락 획득 → 학습 → Challenger 등록
```

**분산 락**: Redis `SET NX` (중복 학습 방지)

## 평가 메트릭

### 분류 성능

| 메트릭 | 목표 |
|--------|------|
| F1 Score | > 0.85 |
| Precision | > 0.85 |
| Recall | > 0.85 |
| AUC-ROC | > 0.90 |

### 배틀 성능

| 메트릭 | 설명 |
|--------|------|
| Detection Rate | 탐지율 (목표: 90%+) |
| Evasion Rate | 우회율 (목표: < 10%) |
| False Positive Rate | 오탐율 |
| False Negative Rate | 미탐율 |

### Champion vs Challenger 비교

```python
comparison = {
    "champion": {"f1": 0.82},
    "challenger": {"f1": 0.87},
    "improvement": {"f1": 0.05},
    "should_promote": True  # challenger.f1 > champion.f1
}
```

Challenger가 Champion보다 F1 높으면 자동 승격.

상세: `05-MLOPS.md`
