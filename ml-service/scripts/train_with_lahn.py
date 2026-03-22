#!/usr/bin/env python3
"""Train hate speech classifier with LAHN (Label-Aware Hard Negative) contrastive loss.

Combines classification (Focal Loss) with contrastive learning (LAHN)
for improved implicit hate speech detection.

Usage:
    # Train with LAHN + Focal Loss
    python scripts/train_with_lahn.py

    # Custom alpha (classification vs contrastive weight)
    python scripts/train_with_lahn.py --alpha 0.6

    # Use specific base model
    python scripts/train_with_lahn.py --model beomi/KcELECTRA-base-v2022

    # Use DeBERTa as base
    python scripts/train_with_lahn.py --model team-lucid/deberta-v3-base-korean

    # Evaluate existing LAHN model
    python scripts/train_with_lahn.py --evaluate-only
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_service.training.lahn_loss import CombinedLAHNLoss
from ml_service.training.standard_config import (
    evaluate_model,
    get_data_paths,
    is_better_model,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "beomi/KcELECTRA-base-v2022"
DEFAULT_OUTPUT = "models/lahn"


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


def get_hidden_states(model, input_ids, attention_mask):
    """Extract [CLS] hidden states from transformer model.

    Works with both ELECTRA and DeBERTa architectures.

    Returns:
        (N, D) hidden state tensor from the last layer's [CLS] token.
    """
    # Get base model outputs with hidden states
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # Last hidden state, [CLS] token
    hidden = outputs.hidden_states[-1][:, 0, :]
    return hidden


def train_with_lahn(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    alpha: float = 0.7,
    queue_size: int = 4096,
    temperature: float = 0.07,
    hard_negative_k: int = 256,
    momentum: float = 0.999,
    warmup_ratio: float = 0.1,
    use_amp: bool = True,
    max_length: int = 256,
    device: torch.device | None = None,
) -> dict:
    """Train model with combined LAHN + Focal Loss.

    Args:
        model_name: HuggingFace model name.
        train_df: Training data.
        valid_df: Validation data.
        output_dir: Output directory.
        epochs: Number of epochs.
        batch_size: Batch size.
        lr: Learning rate.
        alpha: Weight for classification loss (1-alpha for contrastive).
        queue_size: LAHN queue size.
        temperature: Contrastive temperature.
        hard_negative_k: Number of hard negatives.
        momentum: EMA momentum for momentum encoder.
        warmup_ratio: LR warmup ratio.
        use_amp: Use mixed precision.
        max_length: Max sequence length.
        device: Training device.

    Returns:
        Training results dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"LAHN Training: {model_name}")
    logger.info(f"Alpha: {alpha} (cls={alpha:.1%}, contrastive={1-alpha:.1%})")
    logger.info(f"Queue: {queue_size}, Temperature: {temperature}, K: {hard_negative_k}")
    logger.info("=" * 60)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    ).to(device)

    # Create momentum encoder (copy of main model, no gradient)
    momentum_model = copy.deepcopy(model)
    for param in momentum_model.parameters():
        param.requires_grad = False
    momentum_model.to(device)
    momentum_model.eval()

    # Detect embedding dimension from model
    # Try common attribute names
    hidden_size = getattr(model.config, "hidden_size", 768)
    logger.info(f"Hidden size: {hidden_size}")

    # Combined loss
    criterion = CombinedLAHNLoss(
        embedding_dim=hidden_size,
        alpha=alpha,
        queue_size=queue_size,
        temperature=temperature,
        hard_negative_k=hard_negative_k,
    ).to(device)

    # Datasets
    train_dataset = ToxicDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    valid_dataset = ToxicDataset(valid_df["text"].tolist(), valid_df["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    # Include LAHN projector parameters
    all_params = list(model.parameters()) + list(criterion.lahn_loss.projector.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

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
        model.train()
        criterion.train()
        total_cls_loss = 0
        total_con_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast(dtype=torch.float16):
                    # Main model forward
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    logits = outputs.logits
                    hidden = outputs.hidden_states[-1][:, 0, :]  # [CLS]

                    # Momentum model forward (no gradient)
                    with torch.no_grad():
                        mom_outputs = momentum_model(
                            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                        )
                        mom_hidden = mom_outputs.hidden_states[-1][:, 0, :]

                    # Combined loss
                    loss, components = criterion(logits, labels, hidden, mom_hidden)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                logits = outputs.logits
                hidden = outputs.hidden_states[-1][:, 0, :]

                with torch.no_grad():
                    mom_outputs = momentum_model(
                        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                    )
                    mom_hidden = mom_outputs.hidden_states[-1][:, 0, :]

                loss, components = criterion(logits, labels, hidden, mom_hidden)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

            scheduler.step()

            # Update momentum encoder
            with torch.no_grad():
                criterion.lahn_loss.update_momentum_encoder(model, momentum_model)

            total_cls_loss += components["cls_loss"]
            total_con_loss += components["contrastive_loss"]
            n_batches += 1

        avg_cls = total_cls_loss / n_batches
        avg_con = total_con_loss / n_batches

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
            f"Epoch {epoch + 1}: cls_loss={avg_cls:.4f}, con_loss={avg_con:.4f}, "
            f"f1={metrics['f1_weighted']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}"
        )

        if is_better_model(metrics, best_metrics):
            best_f1 = metrics["f1_weighted"]
            best_metrics = metrics.copy()

            model.save_pretrained(output_path / "best_model")
            tokenizer.save_pretrained(output_path / "best_model")
            logger.info(f"  -> New best model saved (F1={best_f1:.4f})")

    elapsed = time.time() - start_time

    results = {
        "model_name": model_name,
        "method": "LAHN",
        "best_metrics": best_metrics,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "alpha": alpha,
        "queue_size": queue_size,
        "temperature": temperature,
        "hard_negative_k": hard_negative_k,
        "momentum": momentum,
        "training_time_seconds": elapsed,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"LAHN training complete: Best F1={best_f1:.4f} in {elapsed:.1f}s")

    del model, momentum_model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train with LAHN contrastive learning"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Classification loss weight (1-alpha for contrastive)")
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negative-k", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default="korean_standard_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    paths = get_data_paths(data_dir, args.dataset_version)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output_dir

    train_df = pd.read_csv(paths["train"])
    valid_df = pd.read_csv(paths["valid"])
    test_df = pd.read_csv(paths["test"]) if paths["test"].exists() else None

    logger.info(f"Dataset: train={len(train_df)}, valid={len(valid_df)}")

    if args.evaluate_only:
        model_path = output_dir / "best_model"
        if not model_path.exists():
            logger.error(f"No model at {model_path}")
            sys.exit(1)

        sys.path.insert(0, str(Path(__file__).parent))
        from train_deberta import evaluate_on_test
        eval_df = test_df if test_df is not None else valid_df
        metrics = evaluate_on_test(str(model_path), eval_df)
        print("\nLAHN Evaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return

    results = train_with_lahn(
        model_name=args.model,
        train_df=train_df,
        valid_df=valid_df,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        queue_size=args.queue_size,
        temperature=args.temperature,
        hard_negative_k=args.hard_negative_k,
        momentum=args.momentum,
        warmup_ratio=args.warmup_ratio,
        use_amp=not args.no_amp,
        max_length=args.max_length,
    )

    # Test set evaluation
    if test_df is not None:
        best_path = output_dir / "best_model"
        if best_path.exists():
            sys.path.insert(0, str(Path(__file__).parent))
            from train_deberta import evaluate_on_test
            test_metrics = evaluate_on_test(str(best_path), test_df)
            results["test_metrics"] = test_metrics

            print("\n" + "=" * 60)
            print("LAHN TEST RESULTS")
            print("=" * 60)
            print(f"  F1: {test_metrics['f1_weighted']:.4f}")
            print(f"  FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")

            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
