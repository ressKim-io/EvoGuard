#!/usr/bin/env python3
"""Train EXAONE 1.2B for toxic text classification using QLoRA.

EXAONE is a Korean-specialized language model. This script fine-tunes
the 1.2B version using QLoRA for efficient training on consumer GPUs.

Requirements:
    - ~3-4GB VRAM (RTX 4060 Ti compatible)
    - peft, bitsandbytes, accelerate packages

Usage:
    python scripts/train_exaone.py
    python scripts/train_exaone.py --epochs 5 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml_service.training.standard_config import (
    STANDARD_CONFIG,
    get_data_paths,
    evaluate_model,
    is_better_model,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# EXAONE model
EXAONE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUTPUT_DIR = "models/pmf/exaone"


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


def train_exaone(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_4bit: bool = True,
) -> dict:
    """Train EXAONE with QLoRA.

    Args:
        train_df: Training data.
        valid_df: Validation data.
        epochs: Number of epochs.
        batch_size: Batch size.
        lr: Learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.
        use_4bit: Use 4-bit quantization (QLoRA).

    Returns:
        Training results dict.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit QLoRA quantization")
        except ImportError:
            logger.warning("bitsandbytes not available, using full precision")
            bnb_config = None
            use_4bit = False
    else:
        bnb_config = None

    logger.info(f"Loading EXAONE model: {EXAONE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EXAONE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "num_labels": 2,
        "trust_remote_code": True,
    }
    if use_4bit and bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForSequenceClassification.from_pretrained(
        EXAONE_MODEL,
        **model_kwargs,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not use_4bit:
        model.to(device)

    # Datasets
    train_dataset = ToxicDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    valid_dataset = ToxicDataset(
        valid_df["text"].tolist(),
        valid_df["label"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Training
    from ml_service.training.losses import FocalLoss

    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
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

        avg_loss = total_loss / len(train_loader)

        # Evaluate
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
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, "
            f"f1={metrics['f1_weighted']:.4f}, "
            f"FP={metrics['fp']}, FN={metrics['fn']}"
        )

        if is_better_model(metrics, best_metrics):
            best_f1 = metrics["f1_weighted"]
            best_metrics = metrics.copy()
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"  -> New best model (F1={best_f1:.4f})")

    # Save results
    results = {
        "model": EXAONE_MODEL,
        "best_metrics": best_metrics,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "use_4bit": use_4bit,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete. Best F1={best_f1:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train EXAONE with QLoRA")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default="korean_standard_v1")
    args = parser.parse_args()

    set_seed(STANDARD_CONFIG.seed)

    # Load data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    paths = get_data_paths(data_dir, args.dataset_version)

    if not paths["train"].exists():
        logger.error(f"Data not found: {paths['train']}")
        sys.exit(1)

    train_df = pd.read_csv(paths["train"])
    valid_df = pd.read_csv(paths["valid"])

    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}")

    train_exaone(
        train_df=train_df,
        valid_df=valid_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
