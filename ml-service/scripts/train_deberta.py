#!/usr/bin/env python3
"""Train DeBERTa-v3 model for Korean hate speech classification.

Uses team-lucid/deberta-v3-base-korean (Korean-tuned DeBERTa) with
disentangled attention for improved contextual understanding.

Follows the same training pipeline as train_multi_model.py for fair comparison.

Usage:
    # Train with default settings (standard dataset, 10 epochs)
    python scripts/train_deberta.py

    # Custom model
    python scripts/train_deberta.py --model lighthouse/mdeberta-v3-base-kor-further

    # Quick test
    python scripts/train_deberta.py --epochs 2 --batch-size 8

    # Evaluate only (no training)
    python scripts/train_deberta.py --evaluate-only

    # Compare with KcELECTRA baseline
    python scripts/train_deberta.py --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_service.training.losses import FocalLoss
from ml_service.training.standard_config import (
    STANDARD_CONFIG,
    evaluate_model,
    get_data_paths,
    is_better_model,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DeBERTa models to try
DEBERTA_MODELS = {
    "team-lucid": "team-lucid/deberta-v3-base-korean",
    "lighthouse": "lighthouse/mdeberta-v3-base-kor-further",
    "mdeberta": "microsoft/mdeberta-v3-base",
}

DEFAULT_MODEL = "team-lucid/deberta-v3-base-korean"
DEFAULT_OUTPUT = "models/deberta-v3"


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


def train_deberta(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    use_amp: bool = True,
    max_length: int = 256,
    device: torch.device | None = None,
) -> dict:
    """Train DeBERTa model on standard dataset.

    Args:
        model_name: HuggingFace model name.
        train_df: Training data with 'text' and 'label' columns.
        valid_df: Validation data.
        output_dir: Output directory for saved model.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        warmup_ratio: Warmup ratio for scheduler.
        use_amp: Use automatic mixed precision.
        max_length: Max sequence length.
        device: Training device.

    Returns:
        Dict with training results and best metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Training DeBERTa-v3: {model_name}")
    logger.info(f"Device: {device}, Epochs: {epochs}, BS: {batch_size}, LR: {lr}")
    logger.info("=" * 60)

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create datasets
    train_dataset = ToxicDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    valid_dataset = ToxicDataset(valid_df["text"].tolist(), valid_df["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Training setup
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_metrics = {}
    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast(dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Validation
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())

        metrics = evaluate_model(all_labels, all_preds, all_probs)
        metrics["epoch"] = epoch + 1

        logger.info(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, f1={metrics['f1_weighted']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}"
        )

        if is_better_model(metrics, best_metrics):
            best_f1 = metrics["f1_weighted"]
            best_metrics = metrics.copy()

            model.save_pretrained(output_path / "best_model")
            tokenizer.save_pretrained(output_path / "best_model")
            logger.info(f"  -> New best model saved (F1={best_f1:.4f})")

    elapsed = time.time() - start_time

    # Save results
    results = {
        "model_name": model_name,
        "best_metrics": best_metrics,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_seconds": elapsed,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "device": str(device),
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete: Best F1={best_f1:.4f} in {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()

    return results


def evaluate_on_test(
    model_path: str,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict:
    """Evaluate a trained model on test set.

    Returns:
        Evaluation metrics dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = ToxicDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    del model
    torch.cuda.empty_cache()

    return evaluate_model(all_labels, all_preds, all_probs)


def compare_with_baseline(
    deberta_metrics: dict,
    baseline_path: str = "models/production",
    test_df: pd.DataFrame | None = None,
    device: torch.device | None = None,
) -> None:
    """Compare DeBERTa results with KcELECTRA baseline."""
    baseline_model_path = Path(baseline_path)
    if not baseline_model_path.exists():
        baseline_model_path = Path(__file__).parent.parent / baseline_path

    print("\n" + "=" * 60)
    print("MODEL COMPARISON: DeBERTa-v3 vs KcELECTRA (Production)")
    print("=" * 60)

    if test_df is not None and baseline_model_path.exists():
        logger.info("Evaluating baseline model on test set...")
        baseline_metrics = evaluate_on_test(str(baseline_model_path), test_df, device=device)
    else:
        # Use recorded metrics
        baseline_metrics = {"f1_weighted": 0.9621, "fp": 182, "fn": 51}

    metrics_to_show = ["f1_weighted", "accuracy", "precision_weighted", "recall_weighted", "fp", "fn"]

    print(f"\n{'Metric':<25} {'KcELECTRA':>12} {'DeBERTa-v3':>12} {'Delta':>12}")
    print("-" * 65)
    for metric in metrics_to_show:
        base_val = baseline_metrics.get(metric, "N/A")
        deb_val = deberta_metrics.get(metric, "N/A")

        if isinstance(base_val, float) and isinstance(deb_val, float):
            delta = deb_val - base_val
            sign = "+" if delta > 0 else ""
            if metric in ("fp", "fn"):
                sign = "+" if delta > 0 else ""  # Lower is better for these
                better = "←" if delta < 0 else "→" if delta > 0 else "="
            else:
                better = "←" if delta > 0 else "→" if delta < 0 else "="
            print(f"  {metric:<23} {base_val:>12.4f} {deb_val:>12.4f} {sign}{delta:>11.4f} {better}")
        elif isinstance(base_val, int) and isinstance(deb_val, int):
            delta = deb_val - base_val
            sign = "+" if delta > 0 else ""
            better = "←" if delta < 0 else ""
            print(f"  {metric:<23} {base_val:>12d} {deb_val:>12d} {sign}{delta:>11d} {better}")
        else:
            print(f"  {metric:<23} {str(base_val):>12} {str(deb_val):>12}")


def main():
    parser = argparse.ArgumentParser(
        description="Train DeBERTa-v3 model for Korean hate speech classification"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default="korean_standard_v1")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only evaluate existing model, skip training")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with KcELECTRA baseline after training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    paths = get_data_paths(data_dir, args.dataset_version)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output_dir

    # Load data
    train_df = pd.read_csv(paths["train"])
    valid_df = pd.read_csv(paths["valid"])
    test_df = pd.read_csv(paths["test"]) if paths["test"].exists() else None

    logger.info(f"Dataset: train={len(train_df)}, valid={len(valid_df)}"
                + (f", test={len(test_df)}" if test_df is not None else ""))

    if args.evaluate_only:
        model_path = output_dir / "best_model"
        if not model_path.exists():
            logger.error(f"No trained model found at {model_path}. Train first.")
            sys.exit(1)

        if test_df is not None:
            metrics = evaluate_on_test(str(model_path), test_df)
        else:
            metrics = evaluate_on_test(str(model_path), valid_df)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        if args.compare:
            compare_with_baseline(metrics, test_df=test_df)
        return

    # Train
    results = train_deberta(
        model_name=args.model,
        train_df=train_df,
        valid_df=valid_df,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        use_amp=not args.no_amp,
        max_length=args.max_length,
    )

    # Evaluate on test set
    if test_df is not None:
        best_model_path = output_dir / "best_model"
        if best_model_path.exists():
            logger.info("Evaluating on test set...")
            test_metrics = evaluate_on_test(str(best_model_path), test_df)
            results["test_metrics"] = test_metrics

            print("\n" + "=" * 60)
            print("TEST SET RESULTS")
            print("=" * 60)
            print(f"  F1: {test_metrics['f1_weighted']:.4f}")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")

            # Update results file
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            if args.compare:
                compare_with_baseline(test_metrics, test_df=test_df)


if __name__ == "__main__":
    main()
