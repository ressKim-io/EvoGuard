#!/usr/bin/env python3
"""Error analysis for Phase 2 model."""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Settings
MODEL_PATH = Path("models/phase2-combined/best_model")
DATA_DIR = Path("data/korean")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256
BATCH_SIZE = 32


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": str(self.texts[idx]),
        }


def load_validation_data():
    """Load all validation datasets."""
    all_texts = []
    all_labels = []
    sources = []

    # KOTOX
    kotox_valid = DATA_DIR / "KOTOX" / "data" / "KOTOX_classification" / "total" / "valid.csv"
    if kotox_valid.exists():
        df = pd.read_csv(kotox_valid)
        all_texts.extend(df["text"].tolist())
        all_labels.extend(df["label"].tolist())
        sources.extend(["KOTOX"] * len(df))
        print(f"KOTOX valid: {len(df)}")

    # BEEP
    beep_dev = DATA_DIR / "beep_dev.tsv"
    if beep_dev.exists():
        df = pd.read_csv(beep_dev, sep="\t")
        df["label"] = df["hate"].apply(lambda x: 0 if x == "none" else 1)
        all_texts.extend(df["comments"].tolist())
        all_labels.extend(df["label"].tolist())
        sources.extend(["BEEP"] * len(df))
        print(f"BEEP dev: {len(df)}")

    # UnSmile
    unsmile_valid = DATA_DIR / "unsmile_valid.tsv"
    if unsmile_valid.exists():
        df = pd.read_csv(unsmile_valid, sep="\t")
        df["label"] = 1 - df["clean"]
        all_texts.extend(df["문장"].tolist())
        all_labels.extend(df["label"].tolist())
        sources.extend(["UnSmile"] * len(df))
        print(f"UnSmile valid: {len(df)}")

    return all_texts, all_labels, sources


def main():
    print("=" * 70)
    print("ERROR ANALYSIS - Phase 2 Model")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    # Load data
    print("\nLoading validation data...")
    texts, labels, sources = load_validation_data()
    print(f"Total validation samples: {len(texts)}")
    print()

    # Predict
    print("Running predictions...")
    predictions = []
    confidences = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i:i+BATCH_SIZE]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            predictions.extend(preds.cpu().tolist())
            confidences.extend(probs.max(dim=-1).values.cpu().tolist())

    # Analysis
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(labels, predictions, target_names=["Clean", "Toxic"]))

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(labels, predictions)
    print(f"                 Predicted")
    print(f"                 Clean  Toxic")
    print(f"Actual Clean     {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"Actual Toxic     {cm[1][0]:5d}  {cm[1][1]:5d}")

    # Error analysis
    false_positives = []  # Clean -> Toxic (잘못된 차단)
    false_negatives = []  # Toxic -> Clean (놓친 독성)

    for i, (text, label, pred, conf, src) in enumerate(zip(texts, labels, predictions, confidences, sources)):
        if label == 0 and pred == 1:
            false_positives.append({
                "text": text,
                "confidence": conf,
                "source": src
            })
        elif label == 1 and pred == 0:
            false_negatives.append({
                "text": text,
                "confidence": conf,
                "source": src
            })

    print("\n" + "=" * 70)
    print(f"FALSE POSITIVES (정상인데 독성으로 예측): {len(false_positives)}건")
    print("=" * 70)
    print("\n[상위 20개 - 높은 confidence 순]")
    for i, fp in enumerate(sorted(false_positives, key=lambda x: -x["confidence"])[:20]):
        print(f"\n{i+1}. [{fp['source']}] (conf: {fp['confidence']:.3f})")
        print(f"   \"{fp['text'][:100]}{'...' if len(fp['text']) > 100 else ''}\"")

    print("\n" + "=" * 70)
    print(f"FALSE NEGATIVES (독성인데 정상으로 예측): {len(false_negatives)}건")
    print("=" * 70)
    print("\n[상위 20개 - 높은 confidence 순]")
    for i, fn in enumerate(sorted(false_negatives, key=lambda x: -x["confidence"])[:20]):
        print(f"\n{i+1}. [{fn['source']}] (conf: {fn['confidence']:.3f})")
        print(f"   \"{fn['text'][:100]}{'...' if len(fn['text']) > 100 else ''}\"")

    # Source-wise analysis
    print("\n" + "=" * 70)
    print("SOURCE-WISE ERROR DISTRIBUTION")
    print("=" * 70)

    fp_sources = Counter([fp["source"] for fp in false_positives])
    fn_sources = Counter([fn["source"] for fn in false_negatives])
    source_counts = Counter(sources)

    print(f"\n{'Source':<12} {'Total':>8} {'FP':>8} {'FN':>8} {'FP%':>8} {'FN%':>8}")
    print("-" * 56)
    for src in ["KOTOX", "BEEP", "UnSmile"]:
        total = source_counts.get(src, 0)
        fp = fp_sources.get(src, 0)
        fn = fn_sources.get(src, 0)
        fp_pct = (fp / total * 100) if total > 0 else 0
        fn_pct = (fn / total * 100) if total > 0 else 0
        print(f"{src:<12} {total:>8} {fp:>8} {fn:>8} {fp_pct:>7.2f}% {fn_pct:>7.2f}%")

    # Pattern analysis for false negatives
    print("\n" + "=" * 70)
    print("FALSE NEGATIVE PATTERN ANALYSIS (놓친 독성)")
    print("=" * 70)

    # Check for obfuscation patterns
    obfuscation_patterns = ["ㅅㅂ", "ㅂㅅ", "ㅈㄴ", "ㄱㅅㄲ", "ㅄ", "ㅗ", "ㅜ", "ㄲ", "tlqkf", "qkf"]
    obfuscated_fn = [fn for fn in false_negatives if any(p in fn["text"] for p in obfuscation_patterns)]
    print(f"\n난독화 패턴 포함: {len(obfuscated_fn)}건 / {len(false_negatives)}건")

    # Check text length
    short_fn = [fn for fn in false_negatives if len(fn["text"]) < 10]
    medium_fn = [fn for fn in false_negatives if 10 <= len(fn["text"]) < 30]
    long_fn = [fn for fn in false_negatives if len(fn["text"]) >= 30]
    print(f"\n길이별 분포:")
    print(f"  - 짧은 텍스트 (<10자): {len(short_fn)}건")
    print(f"  - 중간 텍스트 (10-30자): {len(medium_fn)}건")
    print(f"  - 긴 텍스트 (30자+): {len(long_fn)}건")

    # Save results
    results = {
        "total_samples": len(texts),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "fp_rate": len(false_positives) / len(texts) * 100,
        "fn_rate": len(false_negatives) / len(texts) * 100,
        "fp_by_source": dict(fp_sources),
        "fn_by_source": dict(fn_sources),
        "fp_examples": [{"text": fp["text"], "conf": fp["confidence"], "src": fp["source"]}
                       for fp in sorted(false_positives, key=lambda x: -x["confidence"])[:50]],
        "fn_examples": [{"text": fn["text"], "conf": fn["confidence"], "src": fn["source"]}
                       for fn in sorted(false_negatives, key=lambda x: -x["confidence"])[:50]],
    }

    output_path = Path("logs/error_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
