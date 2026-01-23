#!/usr/bin/env python3
"""Phase 5: CNN-Enhanced Model Training.

This phase adds CNN layers to capture local n-gram patterns (curse words, slang)
while the Transformer captures global context.

Expected improvement:
    - Better detection of obfuscated expressions (ㅅㅂ, 시ㅂ, etc.)
    - Reduced False Negatives through pattern matching
    - Target: F1 0.965+ (from 0.9594)

Usage:
    python scripts/phase5_cnn_enhanced.py
    python scripts/phase5_cnn_enhanced.py --epochs 15 --batch-size 16
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ml_service.models.cnn_enhanced import CNNEnhancedClassifier


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
        text = str(self.texts[idx])
        label = int(self.labels[idx])

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


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def load_datasets(data_dir: Path):
    """Load all training datasets."""
    print("\n📁 Loading datasets...")

    all_texts = []
    all_labels = []

    # 1. KOTOX dataset
    kotox_path = data_dir / "korean/KOTOX/data/KOTOX_classification/total/train.csv"
    if kotox_path.exists():
        df = pd.read_csv(kotox_path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  KOTOX: {len(texts):,} samples")

    # 2. BEEP dataset
    beep_path = data_dir / "korean/beep_train.tsv"
    if beep_path.exists():
        df = pd.read_csv(beep_path, sep="\t")
        texts = df["comments"].tolist()
        labels = [1 if l == "offensive" else 0 for l in df["hate"].tolist()]
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  BEEP: {len(texts):,} samples")

    # 3. UnSmile dataset
    unsmile_path = data_dir / "korean/unsmile_train.tsv"
    if unsmile_path.exists():
        df = pd.read_csv(unsmile_path, sep="\t")
        texts = df["문장"].tolist()
        # Multi-label to binary
        label_cols = [c for c in df.columns if c not in ["문장", "clean"]]
        labels = (df[label_cols].sum(axis=1) > 0).astype(int).tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  UnSmile: {len(texts):,} samples")

    # 4. Curse dataset
    curse_path = data_dir / "korean/curse_dataset.txt"
    if curse_path.exists():
        with open(curse_path, "r", encoding="utf-8") as f:
            curse_words = [line.strip() for line in f if line.strip()]
        all_texts.extend(curse_words)
        all_labels.extend([1] * len(curse_words))
        print(f"  Curse: {len(curse_words):,} samples")

    # 5. Korean hate speech balanced
    hate_path = data_dir / "korean/KOTOX/data/korean_hate_speech_balanced.csv"
    if hate_path.exists():
        df = pd.read_csv(hate_path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  Korean Hate: {len(texts):,} samples")

    # 6. Augmented data
    aug_path = data_dir / "korean/augmented/augmented_toxic.tsv"
    if aug_path.exists():
        df = pd.read_csv(aug_path, sep="\t")
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  Augmented: {len(texts):,} samples")

    print(f"\n  Total: {len(all_texts):,} samples")
    print(f"  Toxic: {sum(all_labels):,} ({sum(all_labels)/len(all_labels)*100:.1f}%)")
    print(f"  Clean: {len(all_labels) - sum(all_labels):,}")

    return all_texts, all_labels


def load_validation_data(data_dir: Path):
    """Load validation datasets."""
    print("\n📁 Loading validation data...")

    all_texts = []
    all_labels = []

    # KOTOX valid
    kotox_valid = data_dir / "korean/KOTOX/data/KOTOX_classification/total/valid.csv"
    if kotox_valid.exists():
        df = pd.read_csv(kotox_valid)
        all_texts.extend(df["text"].tolist())
        all_labels.extend(df["label"].tolist())
        print(f"  KOTOX Valid: {len(df):,}")

    # BEEP dev
    beep_dev = data_dir / "korean/beep_dev.tsv"
    if beep_dev.exists():
        df = pd.read_csv(beep_dev, sep="\t")
        texts = df["comments"].tolist()
        labels = [1 if l == "offensive" else 0 for l in df["hate"].tolist()]
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  BEEP Dev: {len(df):,}")

    # UnSmile valid
    unsmile_valid = data_dir / "korean/unsmile_valid.tsv"
    if unsmile_valid.exists():
        df = pd.read_csv(unsmile_valid, sep="\t")
        texts = df["문장"].tolist()
        label_cols = [c for c in df.columns if c not in ["문장", "clean"]]
        labels = (df[label_cols].sum(axis=1) > 0).astype(int).tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  UnSmile Valid: {len(df):,}")

    print(f"\n  Total Validation: {len(all_texts):,}")

    return all_texts, all_labels


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    loss_fn = FocalLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    # Count FP and FN
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    return {
        "loss": avg_loss,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fp": fp,
        "fn": fn,
    }


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Phase 5: CNN-Enhanced Model Training")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load datasets
    train_texts, train_labels = load_datasets(data_dir)
    val_texts, val_labels = load_validation_data(data_dir)

    # Create datasets
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print(f"\n🏗️  Creating CNN-Enhanced Model...")
    model = CNNEnhancedClassifier(
        model_name=args.model_name,
        num_labels=2,
        cnn_filters=args.cnn_filters,
        kernel_sizes=args.kernel_sizes,
        dropout=args.dropout,
        freeze_transformer_layers=args.freeze_layers,
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer with different learning rates
    transformer_params = list(model.transformer.parameters())
    cnn_params = list(model.conv_layers.parameters()) + list(model.bn_layers.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": transformer_params, "lr": args.lr},
        {"params": cnn_params, "lr": args.lr * 5},  # Higher LR for new CNN layers
        {"params": classifier_params, "lr": args.lr * 5},
    ], weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Loss function
    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # Mixed precision
    scaler = GradScaler() if args.fp16 else None

    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  CNN Filters: {args.cnn_filters}")
    print(f"  Kernel sizes: {args.kernel_sizes}")

    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if args.fp16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs["logits"], labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate
        metrics = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {metrics['loss']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  FP: {metrics['fp']}, FN: {metrics['fn']}")

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch + 1
            print(f"  ✅ New best F1! Saving model...")

            # Save model
            model_save_path = output_dir / "best_model"
            model_save_path.mkdir(parents=True, exist_ok=True)

            # Save model weights
            torch.save(model.state_dict(), model_save_path / "pytorch_model.bin")

            # Save tokenizer
            tokenizer.save_pretrained(model_save_path)

            # Save config
            model.config.save_pretrained(model_save_path)

            # Save training info
            with open(model_save_path / "training_info.txt", "w") as f:
                f.write(f"Phase 5: CNN-Enhanced Model\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Best F1: {best_f1:.4f}\n")
                f.write(f"FP: {metrics['fp']}, FN: {metrics['fn']}\n")
                f.write(f"CNN Filters: {args.cnn_filters}\n")
                f.write(f"Kernel Sizes: {args.kernel_sizes}\n")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {output_dir / 'best_model'}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: CNN-Enhanced Training")

    # Data
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="models/phase5-cnn-enhanced",
                        help="Output directory")

    # Model
    parser.add_argument("--model-name", type=str, default="beomi/KcELECTRA-base-v2022",
                        help="Base transformer model")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--cnn-filters", type=int, default=128,
                        help="Number of CNN filters per kernel")
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="CNN kernel sizes")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Number of transformer layers to freeze")

    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    # Loss
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal loss alpha")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
