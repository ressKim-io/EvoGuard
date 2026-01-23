#!/usr/bin/env python3
"""Phase 4: Train with augmented data for improved accuracy.

Based on error analysis, adds:
1. KOTOX-style obfuscation patterns
2. Context-dependent hate speech patterns
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
from sklearn.metrics import f1_score, classification_report

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


def load_all_data(data_dir: Path):
    """Load all datasets including augmented data."""
    all_train_texts = []
    all_train_labels = []
    all_valid_texts = []
    all_valid_labels = []

    # 1. KOTOX
    kotox_dir = data_dir / "KOTOX" / "data" / "KOTOX_classification" / "total"
    if kotox_dir.exists():
        train_df = pd.read_csv(kotox_dir / "train.csv")
        valid_df = pd.read_csv(kotox_dir / "valid.csv")
        all_train_texts.extend(train_df["text"].tolist())
        all_train_labels.extend(train_df["label"].tolist())
        all_valid_texts.extend(valid_df["text"].tolist())
        all_valid_labels.extend(valid_df["label"].tolist())
        logger.info(f"KOTOX: train={len(train_df)}, valid={len(valid_df)}")

    # 2. BEEP
    beep_train = data_dir / "beep_train.tsv"
    beep_dev = data_dir / "beep_dev.tsv"
    if beep_train.exists():
        train_df = pd.read_csv(beep_train, sep="\t")
        dev_df = pd.read_csv(beep_dev, sep="\t")
        train_df["label"] = train_df["hate"].apply(lambda x: 0 if x == "none" else 1)
        dev_df["label"] = dev_df["hate"].apply(lambda x: 0 if x == "none" else 1)
        all_train_texts.extend(train_df["comments"].tolist())
        all_train_labels.extend(train_df["label"].tolist())
        all_valid_texts.extend(dev_df["comments"].tolist())
        all_valid_labels.extend(dev_df["label"].tolist())
        logger.info(f"BEEP: train={len(train_df)}, valid={len(dev_df)}")

    # 3. UnSmile
    unsmile_train = data_dir / "unsmile_train.tsv"
    unsmile_valid = data_dir / "unsmile_valid.tsv"
    if unsmile_train.exists():
        train_df = pd.read_csv(unsmile_train, sep="\t")
        valid_df = pd.read_csv(unsmile_valid, sep="\t")
        train_df["label"] = 1 - train_df["clean"]
        valid_df["label"] = 1 - valid_df["clean"]
        all_train_texts.extend(train_df["문장"].tolist())
        all_train_labels.extend(train_df["label"].tolist())
        all_valid_texts.extend(valid_df["문장"].tolist())
        all_valid_labels.extend(valid_df["label"].tolist())
        logger.info(f"UnSmile: train={len(train_df)}, valid={len(valid_df)}")

    # 4. Curse dataset
    curse_file = data_dir / "curse_dataset.txt"
    if curse_file.exists():
        with open(curse_file, "r", encoding="utf-8") as f:
            curses = [line.strip() for line in f if line.strip()]
        all_train_texts.extend(curses)
        all_train_labels.extend([1] * len(curses))
        logger.info(f"Curse dataset: {len(curses)}")

    # 5. Korean hate speech
    khs_dir = data_dir / "korean-hate-speech-dataset"
    if khs_dir.exists():
        for split in ["train", "dev"]:
            file_path = khs_dir / f"{split}.tsv"
            if file_path.exists():
                df = pd.read_csv(file_path, sep="\t")
                if "comments" in df.columns and "hate" in df.columns:
                    df["label"] = df["hate"].apply(lambda x: 0 if x == "none" else 1)
                    if split == "train":
                        all_train_texts.extend(df["comments"].tolist())
                        all_train_labels.extend(df["label"].tolist())
                    else:
                        all_valid_texts.extend(df["comments"].tolist())
                        all_valid_labels.extend(df["label"].tolist())
                    logger.info(f"KHS {split}: {len(df)}")

    # 6. AUGMENTED DATA (NEW!)
    augmented_file = data_dir / "augmented" / "augmented_toxic.tsv"
    if augmented_file.exists():
        aug_df = pd.read_csv(augmented_file, sep="\t")
        all_train_texts.extend(aug_df["text"].tolist())
        all_train_labels.extend(aug_df["label"].tolist())
        logger.info(f"Augmented data: {len(aug_df)}")
    else:
        logger.warning(f"Augmented data not found: {augmented_file}")

    return all_train_texts, all_train_labels, all_valid_texts, all_valid_labels


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    return f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--base_model", type=str, default="models/phase2-combined/best_model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Paths
    data_dir = Path("data/korean")
    output_dir = Path("models/phase4-augmented")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model (from phase2)
    logger.info(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    model.to(device)

    # Load data
    logger.info("Loading data...")
    train_texts, train_labels, valid_texts, valid_labels = load_all_data(data_dir)

    logger.info(f"Train size: {len(train_texts)}")
    logger.info(f"Valid size: {len(valid_texts)}")
    logger.info(f"Train toxic ratio: {sum(train_labels)/len(train_labels):.2%}")

    # Create datasets
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    valid_dataset = ToxicDataset(valid_texts, valid_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_f1 = 0.0
    logger.info("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Evaluate
        avg_loss = total_loss / len(train_loader)
        f1, preds, labels = evaluate(model, valid_loader, device)

        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, f1={f1:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"New best model saved! F1: {best_f1:.4f}")

            # Save detailed report
            report = classification_report(labels, preds, target_names=["Clean", "Toxic"])
            logger.info(f"\n{report}")

    # Save results
    results = {
        "best_f1": best_f1,
        "epochs": args.epochs,
        "train_size": len(train_texts),
        "augmented_included": True,
        "base_model": args.base_model,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()
