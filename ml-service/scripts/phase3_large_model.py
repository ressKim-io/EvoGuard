#!/usr/bin/env python3
"""Phase 3: Train with larger model (KoELECTRA-large or similar).

Uses a larger pretrained model for better performance.
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


def load_all_data(data_dir: Path):
    """Load and combine all available datasets."""
    all_train = []
    all_valid = []

    # KOTOX
    kotox_dir = data_dir / "KOTOX" / "data" / "KOTOX_classification" / "total"
    if kotox_dir.exists():
        train_df = pd.read_csv(kotox_dir / "train.csv")[["text", "label"]]
        valid_df = pd.read_csv(kotox_dir / "valid.csv")[["text", "label"]]
        all_train.append(train_df)
        all_valid.append(valid_df)
        logger.info(f"KOTOX: {len(train_df)} train")

    # Beep
    beep_train = data_dir / "beep_train.tsv"
    if beep_train.exists():
        df = pd.read_csv(beep_train, sep="\t")
        df["label"] = df["hate"].apply(lambda x: 0 if x == "none" else 1)
        df = df.rename(columns={"comments": "text"})[["text", "label"]]
        all_train.append(df)
        logger.info(f"Beep: {len(df)} train")

    # Unsmile
    unsmile_train = data_dir / "unsmile_train.tsv"
    if unsmile_train.exists():
        df = pd.read_csv(unsmile_train, sep="\t")
        df["label"] = 1 - df["clean"]
        df = df.rename(columns={"문장": "text"})[["text", "label"]]
        all_train.append(df)
        logger.info(f"Unsmile: {len(df)} train")

    # Korean Hate Speech
    khate = data_dir / "korean_hate_speech_balanced.csv"
    if khate.exists():
        df = pd.read_csv(khate)[["text", "label"]]
        split_idx = int(len(df) * 0.9)
        all_train.append(df[:split_idx])
        all_valid.append(df[split_idx:])
        logger.info(f"Korean Hate: {len(df)} total")

    # Combine
    combined_train = pd.concat(all_train, ignore_index=True)
    combined_valid = pd.concat(all_valid, ignore_index=True)

    # Clean
    combined_train = combined_train.dropna().drop_duplicates(subset=["text"])
    combined_valid = combined_valid.dropna().drop_duplicates(subset=["text"])

    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_valid = combined_valid.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined_train, combined_valid


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Large Model Training")
    # Try different large Korean models
    parser.add_argument("--model", default="monologg/koelectra-base-v3-discriminator",
                       help="Large model to use")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)  # Smaller batch for large model
    parser.add_argument("--lr", type=float, default=1e-5)  # Lower LR for large model
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--output_dir", default="models/phase3-large")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {gpu_mem:.1f} GB")

    data_dir = Path(__file__).parent.parent / "data" / "korean"

    # Load data
    logger.info("Loading all datasets...")
    combined_train, combined_valid = load_all_data(data_dir)
    logger.info(f"Combined Train: {len(combined_train)}")
    logger.info(f"Combined Valid: {len(combined_valid)}")

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

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

    # Training with mixed precision for memory efficiency
    from ml_service.training.losses import FocalLoss

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 3: LARGE MODEL TRAINING")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size} x {args.gradient_accumulation} (grad accum)")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / args.gradient_accumulation

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation
            progress_bar.set_postfix({"loss": f"{total_loss/(step+1):.4f}"})

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast('cuda'):
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
    logger.info(f"PHASE 3 COMPLETE - Best F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "best_f1": best_f1,
            "epochs": args.epochs,
            "model": args.model,
            "params_millions": total_params / 1e6
        }, f)


if __name__ == "__main__":
    main()
