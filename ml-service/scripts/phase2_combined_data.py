#!/usr/bin/env python3
"""Phase 2: Train with combined Korean hate speech datasets.

Combines KOTOX, beep, unsmile, curse, and korean_hate_speech datasets.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToxicDataset(Dataset):
    """Dataset for toxic text classification."""

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
        }


def load_kotox(data_dir: Path):
    """Load KOTOX dataset."""
    kotox_dir = data_dir / "KOTOX" / "data" / "KOTOX_classification" / "total"
    train_df = pd.read_csv(kotox_dir / "train.csv")
    valid_df = pd.read_csv(kotox_dir / "valid.csv")
    return train_df[["text", "label"]], valid_df[["text", "label"]]


def load_beep(data_dir: Path):
    """Load beep dataset."""
    train_path = data_dir / "beep_train.tsv"
    dev_path = data_dir / "beep_dev.tsv"

    if not train_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(train_path, sep="\t")
    dev_df = pd.read_csv(dev_path, sep="\t")

    # Convert hate column to binary (hate/offensive -> 1, none -> 0)
    def to_binary(x):
        return 0 if x == "none" else 1

    train_df["label"] = train_df["hate"].apply(to_binary)
    dev_df["label"] = dev_df["hate"].apply(to_binary)

    train_df = train_df.rename(columns={"comments": "text"})[["text", "label"]]
    dev_df = dev_df.rename(columns={"comments": "text"})[["text", "label"]]

    return train_df, dev_df


def load_unsmile(data_dir: Path):
    """Load unsmile dataset (multi-label -> binary)."""
    train_path = data_dir / "unsmile_train.tsv"
    valid_path = data_dir / "unsmile_valid.tsv"

    if not train_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(train_path, sep="\t")
    valid_df = pd.read_csv(valid_path, sep="\t")

    # Use 'clean' column: 1 = clean, 0 = toxic
    train_df["label"] = 1 - train_df["clean"]
    valid_df["label"] = 1 - valid_df["clean"]

    train_df = train_df.rename(columns={"문장": "text"})[["text", "label"]]
    valid_df = valid_df.rename(columns={"문장": "text"})[["text", "label"]]

    return train_df, valid_df


def load_curse(data_dir: Path):
    """Load curse dataset."""
    curse_path = data_dir / "curse_dataset.txt"

    if not curse_path.exists():
        return pd.DataFrame()

    with open(curse_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # All curse words are toxic (label=1)
    df = pd.DataFrame({"text": lines, "label": [1] * len(lines)})
    return df


def load_korean_hate(data_dir: Path):
    """Load korean hate speech balanced dataset."""
    path = data_dir / "korean_hate_speech_balanced.csv"

    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 90/10
    split_idx = int(len(df) * 0.9)
    train_df = df[:split_idx]
    valid_df = df[split_idx:]

    return train_df[["text", "label"]], valid_df[["text", "label"]]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Combined Data Training")
    parser.add_argument("--model", default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="models/phase2-combined")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(__file__).parent.parent / "data" / "korean"

    # Load all datasets
    logger.info("Loading datasets...")

    kotox_train, kotox_valid = load_kotox(data_dir)
    beep_train, beep_valid = load_beep(data_dir)
    unsmile_train, unsmile_valid = load_unsmile(data_dir)
    curse_df = load_curse(data_dir)
    khate_train, khate_valid = load_korean_hate(data_dir)

    logger.info(f"KOTOX: {len(kotox_train)} train, {len(kotox_valid)} valid")
    logger.info(f"Beep: {len(beep_train)} train, {len(beep_valid)} valid")
    logger.info(f"Unsmile: {len(unsmile_train)} train, {len(unsmile_valid)} valid")
    logger.info(f"Curse: {len(curse_df)} samples")
    logger.info(f"Korean Hate: {len(khate_train)} train, {len(khate_valid)} valid")

    # Combine training data
    train_dfs = [kotox_train, beep_train, unsmile_train, khate_train]
    if not curse_df.empty:
        train_dfs.append(curse_df)

    combined_train = pd.concat([df for df in train_dfs if not df.empty], ignore_index=True)
    combined_valid = pd.concat([kotox_valid, beep_valid, unsmile_valid, khate_valid], ignore_index=True)

    # Remove duplicates and NaN
    combined_train = combined_train.dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined_valid = combined_valid.dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_valid = combined_valid.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Combined Train: {len(combined_train)}")
    logger.info(f"Combined Valid: {len(combined_valid)}")
    logger.info(f"Train label distribution: {combined_train['label'].value_counts().to_dict()}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    # Create datasets
    train_dataset = ToxicDataset(
        combined_train["text"].tolist(),
        combined_train["label"].tolist(),
        tokenizer
    )
    valid_dataset = ToxicDataset(
        combined_valid["text"].tolist(),
        combined_valid["label"].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # Training
    from ml_service.training.losses import FocalLoss

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 2: COMBINED DATA TRAINING")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())

        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = accuracy_score(all_labels, all_preds)

        logger.info(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, f1={f1:.4f}, acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"New best model saved (F1={best_f1:.4f})")

    logger.info("=" * 60)
    logger.info(f"PHASE 2 COMPLETE - Best F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({"best_f1": best_f1, "epochs": args.epochs, "train_size": len(combined_train)}, f)


if __name__ == "__main__":
    main()
