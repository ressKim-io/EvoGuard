---
description: 모델의 FP/FN 에러 패턴 심층 분석
---

# 에러 패턴 분석

모델의 오분류(FP/FN) 패턴을 분석하여 개선 방향을 도출합니다.

## 대상 모델
$ARGUMENTS (기본: models/coevolution-latest)

## 분석 절차

### 1. 오분류 샘플 추출
모델 추론 후 FP(정상→유해 오분류)와 FN(유해→정상 오분류)을 분리합니다.

```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate
python -c "
import torch, pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = '$ARGUMENTS'.strip() or 'models/coevolution-latest'
TEST_CSV = 'data/korean/standard_v1/korean_standard_v1_test.csv'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).cuda().eval()
df = pd.read_csv(TEST_CSV)

results = []
with torch.no_grad():
    for i in range(0, len(df), 64):
        batch_texts = df['text'].iloc[i:i+64].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        probs = torch.softmax(model(**inputs).logits, dim=-1)[:, 1].cpu().tolist()
        for j, p in enumerate(probs):
            idx = i + j
            results.append({'text': df['text'].iloc[idx], 'label': df['label'].iloc[idx], 'prob': p, 'pred': 1 if p >= 0.5 else 0})

import json
rdf = pd.DataFrame(results)
fp = rdf[(rdf['label']==0) & (rdf['pred']==1)].sort_values('prob', ascending=False)
fn = rdf[(rdf['label']==1) & (rdf['pred']==0)].sort_values('prob', ascending=True)
boundary = rdf[rdf['prob'].between(0.4, 0.6)]

print(f'=== FP ({len(fp)}개): 정상인데 유해로 판단 ===')
for _, r in fp.head(20).iterrows():
    print(f'  [{r[\"prob\"]:.3f}] {r[\"text\"][:80]}')

print(f'\n=== FN ({len(fn)}개): 유해인데 정상으로 판단 ===')
for _, r in fn.head(20).iterrows():
    print(f'  [{r[\"prob\"]:.3f}] {r[\"text\"][:80]}')

print(f'\n=== 경계 케이스 ({len(boundary)}개, confidence 0.4~0.6) ===')
for _, r in boundary.head(10).iterrows():
    print(f'  [{r[\"prob\"]:.3f}] label={r[\"label\"]} {r[\"text\"][:80]}')
"
```

### 2. 패턴 분석
위 결과를 기반으로:
1. **FP 패턴 분류**: 어떤 유형의 정상 텍스트가 오탐되는지 (비속어 포함 중립문, 뉴스 인용 등)
2. **FN 패턴 분류**: 어떤 유형의 유해 텍스트가 미탐되는지 (우회 표현, 은유적 혐오 등)
3. **경계 케이스**: 모델이 확신 없는 텍스트들의 공통점
4. **개선 제안**: 데이터 증강, 학습 전략 등 구체적 개선 방안

### 3. 결과 요약
| 에러 유형 | 건수 | 주요 패턴 | 개선 방안 |
|-----------|------|-----------|-----------|
| FP | N | ... | ... |
| FN | N | ... | ... |
| 경계 | N | ... | ... |
