#!/usr/bin/env python3
"""Phase 6: Train with combined Korean datasets including K-HATERS and K-MHaS.

Combines all available Korean hate speech datasets (~370K samples):
- K-HATERS: 192K samples
- K-MHaS: 109K samples
- KOTOX, BEEP, UnSmile, curse, korean_hate_speech

Target:
- F1: 0.9696 -> 0.98+
- FP: 60 -> 30
- FN: 168 -> 100
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ToxicDataset(Dataset):
    """Dataset for toxic text classification."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 256):
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


def compute_metrics(labels: list, preds: list, probs: list = None) -> dict:
    """Compute classification metrics including FP/FN counts."""
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    # Count FP and FN
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)

    return {
        "f1": f1,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Combined Korean Dataset Training")
    parser.add_argument("--model", default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", default="models/phase6-korean")
    parser.add_argument("--train_file", default="data/korean/korean_combined_v2_train.csv")
    parser.add_argument("--valid_file", default="data/korean/korean_combined_v2_valid.csv")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(__file__).parent.parent

    # Load datasets
    train_path = data_dir / args.train_file
    valid_path = data_dir / args.valid_file

    if not train_path.exists():
        logger.error(f"Training file not found: {train_path}")
        logger.error("Run integrate_korean_datasets.py first.")
        return

    logger.info("Loading datasets...")
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    logger.info(f"Train: {len(train_df):,} samples")
    logger.info(f"Valid: {len(valid_df):,} samples")
    logger.info(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    # Create datasets
    train_dataset = ToxicDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )
    valid_dataset = ToxicDataset(
        valid_df["text"].tolist(),
        valid_df["label"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        num_workers=4,
        pin_memory=True,
    )

    # Setup training
    from ml_service.training.losses import FocalLoss

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_metrics = {}

    logger.info("=" * 60)
    logger.info("PHASE 6: COMBINED KOREAN DATASET TRAINING")
    logger.info(f"  Train samples: {len(train_df):,}")
    logger.info(f"  Valid samples: {len(valid_df):,}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size} x {args.gradient_accumulation}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Mixed precision: {args.fp16}")
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

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) / args.gradient_accumulation
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) / args.gradient_accumulation
                loss.backward()

            total_loss += loss.item() * args.gradient_accumulation

            if (step + 1) % args.gradient_accumulation == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation:.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())
                all_probs.extend(probs)

        metrics = compute_metrics(all_labels, all_preds, all_probs)

        logger.info(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, f1={metrics['f1']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            best_metrics["epoch"] = epoch + 1

            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"New best model saved (F1={best_f1:.4f})")

    # Save final model
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")

    logger.info("=" * 60)
    logger.info("PHASE 6 COMPLETE")
    logger.info(f"  Best F1: {best_f1:.4f}")
    logger.info(f"  Best FP: {best_metrics.get('fp', 'N/A')}")
    logger.info(f"  Best FN: {best_metrics.get('fn', 'N/A')}")
    logger.info(f"  Best Epoch: {best_metrics.get('epoch', 'N/A')}")
    logger.info("=" * 60)

    # Save results
    results = {
        "best_f1": best_f1,
        "best_metrics": best_metrics,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "epochs": args.epochs,
        "model": args.model,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Update MODEL_REGISTRY.json
    registry_path = Path(__file__).parent.parent / "models" / "MODEL_REGISTRY.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)

        registry["production_models"]["phase6-korean"] = {
            "path": str(output_dir),
            "description": "Phase 6 Combined Korean - K-HATERS + K-MHaS 포함",
            "f1_score": best_f1,
            "fp_count": best_metrics.get("fp", 0),
            "fn_count": best_metrics.get("fn", 0),
            "status": "experimental",
        }
        registry["last_updated"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info("MODEL_REGISTRY.json updated")


if __name__ == "__main__":
    main()
