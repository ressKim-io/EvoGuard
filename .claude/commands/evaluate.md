---
description: 모델 평가 및 역대 성능 비교 (표준 테스트셋 기준)
---

# 모델 평가

지정된 모델을 표준 테스트셋(korean_standard_v1_test.csv, 6,207 samples)으로 평가하고 역대 성능과 비교합니다.

## 평가 대상
$ARGUMENTS

인자가 없으면 `models/coevolution-latest/` (현재 프로덕션)를 평가합니다.

## 평가 실행
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate
python -c "
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
from torch.utils.data import DataLoader, Dataset

MODEL_PATH = '$ARGUMENTS' if '$ARGUMENTS'.strip() else 'models/coevolution-latest'
TEST_CSV = 'data/korean/standard_v1/korean_standard_v1_test.csv'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).cuda().eval()

df = pd.read_csv(TEST_CSV)
texts, labels = df['text'].tolist(), df['label'].tolist()

all_preds, all_probs = [], []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        preds = [1 if p >= 0.5 else 0 for p in probs]
        all_preds.extend(preds)
        all_probs.extend(probs)

f1 = f1_score(labels, all_preds)
cm = confusion_matrix(labels, all_preds)
tn, fp, fn, tp = cm.ravel()
print(f'F1={f1:.4f} FP={fp} FN={fn} TP={tp} TN={tn}')
print(classification_report(labels, all_preds, target_names=['정상', '유해']))
"
```

## 역대 성능 비교표
현재 모델의 결과를 아래 기준과 비교해주세요:

| 모델 | F1 | FP | FN | 비고 |
|------|----|----|-----|------|
| 12h Pipeline Soup | 0.9844 | 56 | 74 | 역대 최고 |
| v2 FocalLoss+R3F | 0.9766 | 81 | 64 | |
| v1 프로덕션 (공진화) | 0.9621 | 182 | 51 | |

## 분석 포인트
1. F1 변화 (개선/하락)
2. FP/FN 트레이드오프
3. 각 클래스별 precision/recall
4. 프로덕션 교체 추천 여부
