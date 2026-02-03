#!/usr/bin/env python3
"""Train multiple base models for PMF (Parallel Model Fusion) ensemble.

Trains 3 different Korean transformer models on the same dataset:
- beomi/KcELECTRA-base-v2022: Comment-specialized Korean ELECTRA
- klue/bert-base: General-purpose Korean BERT
- monologg/koelectra-base-v3-discriminator: KoELECTRA v3

Each model is fine-tuned on korean_standard_v1 dataset and saved to models/pmf/.

Uses standardized training configuration from ml_service.training.standard_config
for fair and reproducible model comparison.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from ml_service.training.standard_config import (
    STANDARD_CONFIG,
    PMF_MODELS as STANDARD_PMF_MODELS,
    get_data_paths,
    evaluate_model,
    is_better_model,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a base model."""
    name: str
    pretrained: str
    output_dir: str
    description: str


# PMF 앙상블에 사용할 3개 모델 정의
PMF_MODELS = [
    ModelConfig(
        name="kcelectra",
        pretrained="beomi/KcELECTRA-base-v2022",
        output_dir="models/pmf/kcelectra",
        description="Comment-specialized Korean ELECTRA (current baseline)"
    ),
    ModelConfig(
        name="klue-bert",
        pretrained="klue/bert-base",
        output_dir="models/pmf/klue-bert",
        description="General-purpose Korean BERT from KLUE"
    ),
    ModelConfig(
        name="koelectra-v3",
        pretrained="monologg/koelectra-base-v3-discriminator",
        output_dir="models/pmf/koelectra-v3",
        description="KoELECTRA v3 discriminator"
    ),
]


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


def load_data(data_dir: Path | None = None, version: str = "korean_standard_v1"):
    """Load standardized training and validation data.

    Args:
        data_dir: Data directory path. If None, uses default.
        version: Dataset version prefix.

    Returns:
        Tuple of (train_df, valid_df)
    """
    paths = get_data_paths(data_dir, version)

    if not paths["train"].exists():
        raise FileNotFoundError(
            f"Standard dataset not found: {paths['train']}\n"
            "Run: python scripts/create_standard_dataset.py"
        )

    logger.info(f"Loading standardized dataset: {version}")
    train_df = pd.read_csv(paths["train"])
    valid_df = pd.read_csv(paths["valid"])

    logger.info(f"  Train: {len(train_df)}, Valid: {len(valid_df)}")
    return train_df, valid_df


def train_model(
    config: ModelConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """Train a single model.

    Args:
        config: Model configuration
        train_df: Training dataframe with 'text' and 'label' columns
        valid_df: Validation dataframe
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        use_amp: Use automatic mixed precision
        device: Device to train on

    Returns:
        Dictionary with training results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Training: {config.name}")
    logger.info(f"  Model: {config.pretrained}")
    logger.info(f"  Description: {config.description}")
    logger.info("=" * 60)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained,
        num_labels=2
    ).to(device)

    # Create datasets
    train_dataset = ToxicDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer
    )
    valid_dataset = ToxicDataset(
        valid_df["text"].tolist(),
        valid_df["label"].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Training setup
    from ml_service.training.losses import FocalLoss
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler() if use_amp else None

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_metrics = {}

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp and scaler:
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

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluation
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

        # Calculate metrics using standard evaluation
        metrics = evaluate_model(all_labels, all_preds, all_probs)
        metrics["epoch"] = epoch + 1

        logger.info(
            f"Epoch {epoch+1}: loss={avg_loss:.4f}, f1={metrics['f1_weighted']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}"
        )

        if is_better_model(metrics, best_metrics):
            best_f1 = metrics["f1_weighted"]
            best_metrics = metrics.copy()

            # Save best model
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info(f"  -> New best model saved (F1={best_f1:.4f})")

    # Save final results
    results = {
        "model_name": config.name,
        "pretrained": config.pretrained,
        "description": config.description,
        "best_metrics": best_metrics,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete for {config.name}: Best F1={best_f1:.4f}")

    return results


def evaluate_trained_model(config: ModelConfig, valid_df: pd.DataFrame, device: Optional[torch.device] = None):
    """Evaluate a trained model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(config.output_dir) / "best_model"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None

    logger.info(f"Evaluating {config.name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = ToxicDataset(
        valid_df["text"].tolist(),
        valid_df["label"].tolist(),
        tokenizer
    )
    loader = DataLoader(dataset, batch_size=32)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {config.name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    # Use standard evaluation
    metrics = evaluate_model(all_labels, all_preds, all_probs)

    logger.info(f"  {config.name}: F1={metrics['f1_weighted']:.4f}, Acc={metrics['accuracy']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}")

    return {
        "model_name": config.name,
        **metrics,
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multiple models for PMF ensemble")
    parser.add_argument("--models", nargs="+", choices=["kcelectra", "klue-bert", "koelectra-v3", "all"],
                        default=["all"], help="Models to train")
    parser.add_argument("--epochs", type=int, default=STANDARD_CONFIG.epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=STANDARD_CONFIG.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=STANDARD_CONFIG.learning_rate, help="Learning rate")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing models")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--dataset-version", type=str, default=STANDARD_CONFIG.dataset_version,
                        help="Dataset version (e.g., korean_standard_v1)")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(STANDARD_CONFIG.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {STANDARD_CONFIG.seed}")

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    # Load standardized data
    train_df, valid_df = load_data(data_dir, version=args.dataset_version)
    logger.info(f"Dataset: {args.dataset_version}")
    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
    logger.info(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")

    # Select models to train
    if "all" in args.models:
        models_to_train = PMF_MODELS
    else:
        models_to_train = [m for m in PMF_MODELS if m.name in args.models]

    if args.evaluate_only:
        # Evaluate existing models
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION ONLY MODE")
        logger.info("=" * 60)

        results = []
        for config in models_to_train:
            result = evaluate_trained_model(config, valid_df, device)
            if result:
                results.append(result)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        for r in results:
            logger.info(f"  {r['model_name']}: F1={r['f1_weighted']:.4f}, FP={r['fp']}, FN={r['fn']}")

        # Save summary
        summary_path = Path("models/pmf/evaluation_summary.json")
        summary = [{k: v for k, v in r.items() if k not in ["probs", "preds", "labels"]} for r in results]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved to {summary_path}")

    else:
        # Train all models
        logger.info("\n" + "=" * 60)
        logger.info("PMF MULTI-MODEL TRAINING")
        logger.info("=" * 60)

        all_results = []
        for config in models_to_train:
            result = train_model(
                config=config,
                train_df=train_df,
                valid_df=valid_df,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_amp=not args.no_amp,
                device=device,
            )
            all_results.append(result)

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        for r in all_results:
            m = r["best_metrics"]
            logger.info(f"  {r['model_name']}: F1={m['f1']:.4f}, FP={m['fp']}, FN={m['fn']}")

        # Save summary
        summary_path = Path("models/pmf/training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
