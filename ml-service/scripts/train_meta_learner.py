#!/usr/bin/env python3
"""Train meta-learner for PMF ensemble.

Uses validation set predictions from all base models to train an optimal
combination model (XGBoost or Logistic Regression).

The meta-learner learns the best way to combine individual model predictions
based on their performance on the validation set.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Using Logistic Regression as meta-learner.")


class TextDataset(Dataset):
    """Simple text dataset for prediction."""

    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
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
        }


def get_model_predictions(
    model_path: str,
    texts: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Get predictions from a single model.

    Args:
        model_path: Path to model
        texts: List of texts
        device: Device to use
        batch_size: Batch size

    Returns:
        Array of toxic probabilities
    """
    model_path = Path(model_path)
    model_name = model_path.parent.name if model_path.name == "best_model" else model_path.name

    logger.info(f"Getting predictions from {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=model_name, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return np.array(all_probs)


def train_xgboost_meta_learner(
    X: np.ndarray,
    y: np.ndarray,
    model_names: List[str],
) -> "xgb.XGBClassifier":
    """Train XGBoost meta-learner.

    Args:
        X: Feature matrix (n_samples, n_models)
        y: Labels
        model_names: Names of base models

    Returns:
        Trained XGBoost classifier
    """
    logger.info("Training XGBoost meta-learner...")

    meta_learner = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    meta_learner.fit(X, y)

    # Feature importance
    logger.info("Feature importance (model weights):")
    for name, importance in zip(model_names, meta_learner.feature_importances_):
        logger.info(f"  {name}: {importance:.4f}")

    return meta_learner


def train_logistic_meta_learner(
    X: np.ndarray,
    y: np.ndarray,
    model_names: List[str],
) -> LogisticRegression:
    """Train Logistic Regression meta-learner.

    Args:
        X: Feature matrix (n_samples, n_models)
        y: Labels
        model_names: Names of base models

    Returns:
        Trained LogisticRegression classifier
    """
    logger.info("Training Logistic Regression meta-learner...")

    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )

    meta_learner.fit(X, y)

    # Model coefficients as weights
    logger.info("Model coefficients (effective weights):")
    for name, coef in zip(model_names, meta_learner.coef_[0]):
        logger.info(f"  {name}: {coef:.4f}")

    return meta_learner


def evaluate_ensemble(
    meta_learner,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate meta-learner ensemble.

    Args:
        meta_learner: Trained meta-learner
        X: Feature matrix
        y: Labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    probs = meta_learner.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)

    f1 = f1_score(y, preds, average="weighted")
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, average="weighted")
    recall = recall_score(y, preds, average="weighted")
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "auc_roc": float(auc),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def compare_strategies(
    probs_matrix: np.ndarray,
    labels: np.ndarray,
    model_names: List[str],
    meta_learner,
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """Compare different ensemble strategies.

    Args:
        probs_matrix: Matrix of probabilities (n_samples, n_models)
        labels: True labels
        model_names: Names of models
        meta_learner: Trained meta-learner
        threshold: Classification threshold

    Returns:
        Dictionary of strategy -> metrics
    """
    strategies = {}

    # Individual models
    for i, name in enumerate(model_names):
        probs = probs_matrix[:, i]
        preds = (probs > threshold).astype(int)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        strategies[f"individual_{name}"] = {
            "f1": float(f1_score(labels, preds, average="weighted")),
            "fp": int(fp),
            "fn": int(fn),
        }

    # Simple average
    avg_probs = probs_matrix.mean(axis=1)
    avg_preds = (avg_probs > threshold).astype(int)
    cm = confusion_matrix(labels, avg_preds)
    tn, fp, fn, tp = cm.ravel()
    strategies["simple_avg"] = {
        "f1": float(f1_score(labels, avg_preds, average="weighted")),
        "fp": int(fp),
        "fn": int(fn),
    }

    # Weighted average (0.4, 0.3, 0.3)
    weights = np.array([0.4, 0.3, 0.3])
    weighted_probs = np.average(probs_matrix, axis=1, weights=weights)
    weighted_preds = (weighted_probs > threshold).astype(int)
    cm = confusion_matrix(labels, weighted_preds)
    tn, fp, fn, tp = cm.ravel()
    strategies["weighted_avg"] = {
        "f1": float(f1_score(labels, weighted_preds, average="weighted")),
        "fp": int(fp),
        "fn": int(fn),
    }

    # AND strategy (all must agree)
    and_positive = np.all(probs_matrix > threshold, axis=1)
    and_preds = and_positive.astype(int)
    cm = confusion_matrix(labels, and_preds)
    tn, fp, fn, tp = cm.ravel()
    strategies["and"] = {
        "f1": float(f1_score(labels, and_preds, average="weighted")),
        "fp": int(fp),
        "fn": int(fn),
    }

    # OR strategy (any positive)
    or_positive = np.any(probs_matrix > threshold, axis=1)
    or_preds = or_positive.astype(int)
    cm = confusion_matrix(labels, or_preds)
    tn, fp, fn, tp = cm.ravel()
    strategies["or"] = {
        "f1": float(f1_score(labels, or_preds, average="weighted")),
        "fp": int(fp),
        "fn": int(fn),
    }

    # Voting (majority)
    votes = (probs_matrix > threshold).sum(axis=1)
    voting_preds = (votes >= 2).astype(int)  # Majority (2/3)
    cm = confusion_matrix(labels, voting_preds)
    tn, fp, fn, tp = cm.ravel()
    strategies["voting"] = {
        "f1": float(f1_score(labels, voting_preds, average="weighted")),
        "fp": int(fp),
        "fn": int(fn),
    }

    # Meta-learner
    meta_probs = meta_learner.predict_proba(probs_matrix)[:, 1]
    meta_preds = (meta_probs > threshold).astype(int)
    cm = confusion_matrix(labels, meta_preds)
    tn, fp, fn, tp = cm.ravel()
    strategies["meta_learner"] = {
        "f1": float(f1_score(labels, meta_preds, average="weighted")),
        "fp": int(fp),
        "fn": int(fn),
    }

    return strategies


def main():
    parser = argparse.ArgumentParser(description="Train meta-learner for PMF ensemble")
    parser.add_argument("--valid-file", type=str, default=None,
                        help="Validation CSV file path")
    parser.add_argument("--model-dir", type=str, default="models/pmf",
                        help="Directory containing trained models")
    parser.add_argument("--output", type=str, default="models/pmf/meta_learner.pkl",
                        help="Output path for meta-learner")
    parser.add_argument("--learner", choices=["xgboost", "logistic", "auto"],
                        default="auto", help="Meta-learner type")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Find validation file
    if args.valid_file:
        valid_path = Path(args.valid_file)
    else:
        valid_path = Path(__file__).parent.parent / "data" / "korean" / "korean_combined_v2_valid.csv"

    if not valid_path.exists():
        logger.error(f"Validation file not found: {valid_path}")
        sys.exit(1)

    logger.info(f"Loading validation data from {valid_path}...")
    valid_df = pd.read_csv(valid_path)
    texts = valid_df["text"].tolist()
    labels = valid_df["label"].values

    logger.info(f"Validation samples: {len(texts)}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Find model paths
    model_dir = Path(args.model_dir)
    model_configs = [
        ("kcelectra", model_dir / "kcelectra" / "best_model"),
        ("klue-bert", model_dir / "klue-bert" / "best_model"),
        ("koelectra-v3", model_dir / "koelectra-v3" / "best_model"),
    ]

    # Check if models exist
    available_models = []
    for name, path in model_configs:
        if path.exists():
            available_models.append((name, path))
        else:
            logger.warning(f"Model not found: {path}")

    if len(available_models) < 2:
        logger.error("Need at least 2 models for ensemble. Train models first:")
        logger.error("  python scripts/train_multi_model.py")
        sys.exit(1)

    logger.info(f"Found {len(available_models)} models: {[m[0] for m in available_models]}")

    # Get predictions from each model
    logger.info("\n" + "=" * 60)
    logger.info("GETTING BASE MODEL PREDICTIONS")
    logger.info("=" * 60)

    model_names = []
    probs_list = []

    for name, path in available_models:
        probs = get_model_predictions(str(path), texts, device, args.batch_size)
        model_names.append(name)
        probs_list.append(probs)

    # Stack predictions
    probs_matrix = np.stack(probs_list, axis=1)  # Shape: (n_samples, n_models)
    logger.info(f"Predictions shape: {probs_matrix.shape}")

    # Train meta-learner
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING META-LEARNER")
    logger.info("=" * 60)

    if args.learner == "xgboost" or (args.learner == "auto" and HAS_XGBOOST):
        meta_learner = train_xgboost_meta_learner(probs_matrix, labels, model_names)
    else:
        meta_learner = train_logistic_meta_learner(probs_matrix, labels, model_names)

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    metrics = evaluate_ensemble(meta_learner, probs_matrix, labels, args.threshold)
    logger.info(f"Meta-learner performance:")
    logger.info(f"  F1: {metrics['f1']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  FP: {metrics['fp']}, FN: {metrics['fn']}")

    # Compare strategies
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 60)

    strategies = compare_strategies(probs_matrix, labels, model_names, meta_learner, args.threshold)

    # Sort by F1 score
    sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]["f1"], reverse=True)

    logger.info(f"{'Strategy':<25} {'F1':<10} {'FP':<8} {'FN':<8}")
    logger.info("-" * 55)
    for strategy, m in sorted_strategies:
        logger.info(f"{strategy:<25} {m['f1']:<10.4f} {m['fp']:<8} {m['fn']:<8}")

    # Find best strategy
    best_strategy = sorted_strategies[0]
    logger.info(f"\nBest strategy: {best_strategy[0]} (F1: {best_strategy[1]['f1']:.4f})")

    # Save meta-learner
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(meta_learner, f)
    logger.info(f"\nMeta-learner saved to {output_path}")

    # Save results
    results = {
        "model_names": model_names,
        "metrics": metrics,
        "strategy_comparison": strategies,
        "best_strategy": best_strategy[0],
        "threshold": args.threshold,
        "learner_type": "xgboost" if HAS_XGBOOST and args.learner != "logistic" else "logistic",
        "valid_samples": len(texts),
    }

    results_path = output_path.parent / "meta_learner_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save predictions for analysis
    predictions_path = output_path.parent / "validation_predictions.csv"
    pred_df = pd.DataFrame({
        "text": texts,
        "label": labels,
        **{name: probs_list[i] for i, name in enumerate(model_names)},
        "meta_learner": meta_learner.predict_proba(probs_matrix)[:, 1],
    })
    pred_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
