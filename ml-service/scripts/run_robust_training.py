#!/usr/bin/env python3
"""Robust Training Script with Focal Loss + TRADES + Adaptive Precision.

Usage:
    python scripts/run_robust_training.py --epochs 5 --batch_size 8
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ToxicDataset(Dataset):
    """Dataset for toxic text classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_kotox_data(data_dir: Path, split: str = "train", shuffle: bool = True) -> tuple[list[str], list[int]]:
    """Load KOTOX classification dataset."""
    file_path = data_dir / "KOTOX" / "data" / "KOTOX_classification" / "total" / f"{split}.csv"

    if not file_path.exists():
        logger.warning(f"KOTOX {split} not found at {file_path}")
        return [], []

    df = pd.read_csv(file_path)

    # Shuffle data to ensure balanced sampling when limiting
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    logger.info(f"Loaded {len(texts)} samples from KOTOX {split}")
    logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    return texts, labels


def load_korean_hate_data(data_dir: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load Korean hate speech dataset."""
    file_path = data_dir / "korean_hate_speech_balanced.csv"

    if not file_path.exists():
        logger.warning(f"Korean hate speech not found at {file_path}")
        return [], [], [], []

    df = pd.read_csv(file_path)

    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.9)

    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # Get text and label columns
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]

    train_texts = train_df[text_col].tolist()
    train_labels = train_df[label_col].tolist()
    val_texts = val_df[text_col].tolist()
    val_labels = val_df[label_col].tolist()

    logger.info(f"Loaded {len(train_texts)} train, {len(val_texts)} val from Korean hate speech")
    return train_texts, train_labels, val_texts, val_labels


def main():
    parser = argparse.ArgumentParser(description="Robust Adversarial Training")
    parser.add_argument("--model", default="beomi/KcELECTRA-base-v2022", help="Model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--trades_beta", type=float, default=6.0, help="TRADES beta")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal gamma")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples (for quick test)")
    parser.add_argument("--output_dir", default="models/robust-kotox", help="Output directory")
    parser.add_argument("--no_adversarial", action="store_true", help="Disable adversarial training")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
    ).to(device)

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "korean"

    # Try KOTOX first, then Korean hate speech
    train_texts, train_labels = load_kotox_data(data_dir, "train")
    val_texts, val_labels = load_kotox_data(data_dir, "valid")

    if not train_texts:
        # Fallback to Korean hate speech
        train_texts, train_labels, val_texts, val_labels = load_korean_hate_data(data_dir)

    if not train_texts:
        logger.error("No data found!")
        return

    # Limit samples for quick test
    if args.max_samples and len(train_texts) > args.max_samples:
        train_texts = train_texts[:args.max_samples]
        train_labels = train_labels[:args.max_samples]
        val_texts = val_texts[:args.max_samples // 5]
        val_labels = val_labels[:args.max_samples // 5]
        logger.info(f"Limited to {len(train_texts)} train, {len(val_texts)} val samples")

    # Create datasets
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Import training modules
    from ml_service.training.losses import RobustLoss, FocalLoss
    from ml_service.training.robust_trainer import (
        RobustTrainer,
        RobustTrainingConfig,
        TextAdversarialAttacker,
    )

    # Create config
    config = RobustTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        trades_beta=args.trades_beta,
        focal_gamma=args.focal_gamma,
        use_adaptive_precision=True,
        adversarial_training=not args.no_adversarial,
        output_dir=Path(args.output_dir),
    )

    # Create trainer
    trainer = RobustTrainer(model, tokenizer, config)

    # Train
    logger.info("=" * 60)
    logger.info("STARTING ROBUST TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"TRADES beta: {args.trades_beta}")
    logger.info(f"Focal gamma: {args.focal_gamma}")
    logger.info(f"Adversarial: {not args.no_adversarial}")
    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Val samples: {len(val_texts)}")
    logger.info("=" * 60)

    results = trainer.train(train_loader, val_loader)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best F1: {results['best_f1']:.4f}")
    logger.info(f"Model saved to: {results['model_path']}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / "training_results.json", "w") as f:
        # Convert non-serializable items
        results_json = {
            "best_f1": results["best_f1"],
            "final_metrics": results["final_metrics"],
            "model_path": str(results["model_path"]),
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "trades_beta": args.trades_beta,
                "focal_gamma": args.focal_gamma,
                "adversarial": not args.no_adversarial,
            }
        }
        json.dump(results_json, f, indent=2)

    logger.info(f"Results saved to: {output_dir / 'training_results.json'}")


if __name__ == "__main__":
    main()
