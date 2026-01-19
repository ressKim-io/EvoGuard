# EVOGUARD Training Experiments Report

**Generated:** 2026-01-19 20:58:07
**Total Experiments:** 1

## 🏆 Best Experiment

| Metric | Value |
|--------|-------|
| **Name** | qlora-bert-base-uncased |
| **F1 Score** | 0.9131 |
| **Accuracy** | 0.9130 |
| **Date** | 2026-01-19 |

## 📊 Experiments Summary

| # | Name | Dataset | F1 | Accuracy | Date |
|---|------|---------|-----|----------|------|
| 1 | qlora-bert-base-uncased | jigsaw (Arsive/toxicity_classification_jigsaw) | 0.9131 | 0.9130 | 2026-01-19 |

## 📝 Experiment Details

### Experiment #1: qlora-bert-base-uncased

**Timestamp:** 2026-01-19T20:58:07.360139

**Dataset:** jigsaw (Arsive/toxicity_classification_jigsaw)
**Tags:** qlora, toxic-classification

#### Configuration
```json
{
  "model_name": "bert-base-uncased",
  "max_length": 256,
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 0.0002,
  "gradient_accumulation_steps": 4,
  "use_4bit_quantization": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "max_samples": 10000
}
```

#### Metrics
| Metric | Value |
|--------|-------|
| train_loss | 0.213300 |
| eval_loss | 0.258200 |
| eval_accuracy | 0.913000 |
| eval_f1 | 0.913100 |
| eval_precision | 0.913300 |
| eval_recall | 0.913000 |
| test_eval_loss | 1.136800 |
| test_eval_accuracy | 0.727900 |
| test_eval_f1 | 0.808500 |
| test_eval_precision | 0.959700 |
| test_eval_recall | 0.727900 |

**Model Path:** `models/checkpoints`

---
