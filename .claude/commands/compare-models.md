---
description: 두 모델의 성능을 테스트셋에서 직접 비교
---

# 모델 비교

두 모델을 동일한 테스트셋으로 평가하여 비교합니다.

## 비교 대상
$ARGUMENTS

예시: `models/pipeline_12h/final_model models/coevolution-latest`

인자가 없으면 pipeline_12h/final_model vs coevolution-latest를 비교합니다.

## 비교 실행
```bash
cd /home/resshome/project/EvoGuard/ml-service && source .venv/bin/activate
python -c "
import torch, pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix

TEST_CSV = 'data/korean/standard_v1/korean_standard_v1_test.csv'
df = pd.read_csv(TEST_CSV)
texts, labels = df['text'].tolist(), df['label'].tolist()

args = '$ARGUMENTS'.split() if '$ARGUMENTS'.strip() else ['models/pipeline_12h/final_model', 'models/coevolution-latest']
models = args[:2] if len(args) >= 2 else args + ['models/coevolution-latest']

for model_path in models:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda().eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            inputs = tokenizer(texts[i:i+64], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
            probs = torch.softmax(model(**inputs).logits, dim=-1)[:, 1].cpu().tolist()
            all_preds.extend([1 if p >= 0.5 else 0 for p in probs])
            all_probs.extend(probs)
    f1 = f1_score(labels, all_preds)
    cm = confusion_matrix(labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f'{model_path}: F1={f1:.4f} FP={fp} FN={fn} TP={tp} TN={tn}')
    del model; torch.cuda.empty_cache()
"
```

## 분석
1. F1/FP/FN 비교표
2. 어떤 모델이 어떤 케이스에서 더 나은지
3. 프로덕션 교체 추천 여부
