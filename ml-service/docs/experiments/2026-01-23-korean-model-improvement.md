# 한국어 독성 텍스트 분류 모델 개선 방향

> 작성일: 2026-01-23
> 현재 최고 성능: F1 0.9594 (앙상블)

## 1. 현재 모델 아키텍처

### 1.1 KcELECTRA-base-v2022 구조

```
┌─────────────────────────────────────────────────────────┐
│  Input: 한국어 텍스트 (max 256 tokens)                   │
├─────────────────────────────────────────────────────────┤
│  Embedding Layer                                         │
│    ├─ Token Embedding (54,343 vocab)                    │
│    ├─ Position Embedding (512 max)                      │
│    └─ Segment Embedding                                  │
│    → Output: [batch, seq_len, 768]                      │
├─────────────────────────────────────────────────────────┤
│  Transformer Encoder × 12 Layers                         │
│    ├─ Multi-Head Self-Attention (12 heads, 64 dim each) │
│    │   └─ Attention(Q,K,V) = softmax(QK^T/√d)V          │
│    ├─ Feed-Forward Network                               │
│    │   └─ FFN(x) = GELU(xW₁+b₁)W₂+b₂                    │
│    │   └─ 768 → 3072 → 768                              │
│    └─ Layer Norm + Residual Connection                  │
│    → Output: [batch, seq_len, 768]                      │
├─────────────────────────────────────────────────────────┤
│  Pooling: [CLS] token extraction                         │
│    → Output: [batch, 768]                               │
├─────────────────────────────────────────────────────────┤
│  Classification Head                                     │
│    └─ Dropout(0.1) → Linear(768→2) → Softmax            │
│    → Output: [batch, 2] (정상/독성 확률)                 │
└─────────────────────────────────────────────────────────┘

총 파라미터: ~110M
학습 가능: 전체 (Fine-tuning)
```

### 1.2 현재 앙상블 구성

```
Phase 2 Model (KcELECTRA)     Phase 4 Model (KcELECTRA)
        │ weight=0.6                  │ weight=0.4
        ▼                             ▼
   [prob_clean, prob_toxic]    [prob_clean, prob_toxic]
        │                             │
        └──────────┬──────────────────┘
                   ▼
         Weighted Average
                   │
                   ▼
         threshold > 0.5 → Toxic
```

## 2. 현재 성능 지표

| 모델 | F1 | Precision | Recall | FP | FN |
|------|-----|-----------|--------|-----|-----|
| Phase 2 (단독) | 0.9597 | - | - | 80 | 164 |
| Phase 4 (단독) | 0.9580 | - | - | 98 | 137 |
| **앙상블 (0.6:0.4)** | **0.9594** | - | - | **78** | **150** |

### 2.1 에러 분석

**False Negative 주요 패턴:**
1. 맥락 의존적 표현: "백린탄이 필요하다", "앞차 최소 전라도"
2. 암시적 혐오: "여판사네", "땅크 부릉부릉"
3. 난독화 변종: ㅅㅂ, 시ㅂ, 씨ㄹ 등

**False Positive 주요 패턴:**
1. 무기/폭력 단어의 정상 문맥 사용
2. 유사 욕설 패턴 오탐

## 3. 최신 연구 동향 (2025)

### 3.1 PMF (Parallel Model Fusion) - Nature Scientific Reports

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│   BERT   │  │DistilBERT│  │ RoBERTa  │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   ▼
           ┌──────────────┐
           │  Meta-Learner │
           │  (RF, SVM)    │
           └──────────────┘
```

- **성능**: 한국어 89% accuracy, 영어 85%
- **특징**: Thompson Sampling으로 동적 가중치 조정
- **참고**: https://www.nature.com/articles/s41598-025-88960-y

### 3.2 CNN + Transformer 결합

```
Transformer Output [batch, seq_len, 768]
        │
        ▼
   Conv1D Layers (다양한 kernel size)
   ├─ kernel=2 (bigram 패턴)
   ├─ kernel=3 (trigram 패턴)
   └─ kernel=4 (4-gram 패턴)
        │
        ▼
   MaxPooling + Concatenate
        │
        ▼
   Classification Head
```

- **장점**: CNN이 로컬 n-gram 패턴(욕설) 포착, Transformer가 전역 맥락 이해
- **참고**: https://arxiv.org/html/2511.06051v1

### 3.3 LoRA 기반 경량화

- 전체 모델 대신 3개 레이어만 학습
- 메모리 효율적
- 참고: https://arxiv.org/html/2511.06051v1

## 4. 개선 방향

### 4.1 Phase 5: CNN-Enhanced Model (권장)

```python
class CNNEnhancedClassifier(nn.Module):
    """
    Transformer + CNN 결합 모델

    장점:
    - CNN이 욕설 n-gram 패턴 직접 포착
    - Transformer가 맥락 이해
    - 두 정보 결합으로 FN/FP 동시 감소 기대
    """
    def __init__(self, base_model, hidden_size=768):
        self.transformer = base_model

        # Multi-scale CNN for n-gram patterns
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, 128, kernel_size=2),  # bigram
            nn.Conv1d(hidden_size, 128, kernel_size=3),  # trigram
            nn.Conv1d(hidden_size, 128, kernel_size=4),  # 4-gram
        ])

        # Combined classifier
        # 768 (CLS) + 128*3 (CNN) = 1152
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 + 128*3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
```

**예상 효과:**
- F1: 0.9594 → 0.965+ (+0.5~1%)
- FN 감소: 난독화 욕설 패턴 직접 포착
- FP 감소: 맥락 정보와 결합으로 오탐 감소

### 4.2 Phase 6: Meta-Learner Ensemble

```python
# Phase 2, 4, 5 모델 출력을 입력으로
from sklearn.ensemble import RandomForestClassifier

meta_features = np.column_stack([
    phase2_probs,  # [N, 2]
    phase4_probs,  # [N, 2]
    phase5_probs,  # [N, 2]
])

meta_learner = RandomForestClassifier(n_estimators=100)
meta_learner.fit(meta_features, labels)
```

### 4.3 Phase 7: 멀티태스크 학습

```
Input → Transformer → Shared Representation
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      Binary Task    Type Task     Severity Task
      (독성 여부)    (혐오 유형)    (심각도 1-5)
```

## 5. 우선순위 및 일정

| 단계 | 작업 | 예상 효과 | 난이도 | 우선순위 |
|------|------|----------|--------|---------|
| Phase 5 | CNN 레이어 추가 | F1 +0.5~1% | 중간 | 🔴 높음 |
| Phase 6 | Meta-Learner | F1 +0.3~0.5% | 낮음 | 🟡 중간 |
| Phase 7 | 멀티태스크 | 세밀한 분류 | 높음 | 🟢 낮음 |
| - | 정상 데이터 증강 | FP -20~30% | 낮음 | 🔴 높음 |

## 6. 참고 자료

### 논문
- [Adaptive ensemble techniques (Nature 2025)](https://www.nature.com/articles/s41598-025-88960-y)
- [Korean Political Hate Speech (Springer 2024)](https://link.springer.com/article/10.1007/s10579-024-09797-x)
- [3-Layer LoRA BERTweet (arXiv 2025)](https://arxiv.org/html/2511.06051v1)
- [K-HATERS Corpus (EMNLP 2023)](https://aclanthology.org/2023.findings-emnlp.952.pdf)

### 데이터셋
- [BEEP! Korean Toxic Speech](https://github.com/kocohub/korean-hate-speech)
- [K-MHaS Multi-label Hate Speech](https://github.com/adlnlp/K-MHaS)
- [Korean Hate Speech (HuggingFace)](https://huggingface.co/datasets/nayohan/korean-hate-speech)

### 모델
- [KcELECTRA](https://github.com/Beomi/KcELECTRA)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
