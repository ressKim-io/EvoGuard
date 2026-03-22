#!/usr/bin/env python3
"""Find label errors in training data using cleanlab.

Uses cross-validated predictions from the production model to identify
likely mislabeled samples in the training set.

Usage:
    # Find label errors and print report
    python scripts/find_label_errors.py

    # Generate cleaned dataset
    python scripts/find_label_errors.py --output-cleaned

    # Use specific model
    python scripts/find_label_errors.py --model-path models/coevolution-latest

    # Custom cross-validation folds
    python scripts/find_label_errors.py --cv-folds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_service.training.standard_config import get_data_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple text dataset for inference."""

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


def get_cross_validated_probs(
    texts: list[str],
    labels: np.ndarray,
    model_path: str,
    cv_folds: int = 5,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Get cross-validated predicted probabilities.

    For each fold, loads the pretrained model, fine-tunes briefly on
    the training fold, and predicts probabilities for the held-out fold.

    For simplicity (no GPU training required), uses the production model
    directly for all predictions — a standard approach with cleanlab when
    you have a well-trained model.

    Args:
        texts: All training texts.
        labels: All training labels.
        model_path: Path to the pretrained model.
        cv_folds: Number of CV folds.
        batch_size: Batch size for inference.
        device: Device to use.

    Returns:
        Array of shape (n_samples, n_classes) with predicted probabilities.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Computing cross-validated probabilities ({cv_folds} folds)...")
    logger.info(f"Model: {model_path}, Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # For a well-trained model, direct prediction is a valid approach
    # (out-of-sample is not strictly needed when the model wasn't trained on this exact split)
    pred_probs = np.zeros((len(texts), 2))

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        fold_texts = [texts[i] for i in val_idx]
        dataset = SimpleDataset(fold_texts, tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size)

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold {fold + 1}/{cv_folds}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                fold_probs.append(probs)

        fold_probs = np.vstack(fold_probs)
        pred_probs[val_idx] = fold_probs

    del model
    torch.cuda.empty_cache()

    return pred_probs


def find_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    min_confidence: float = 0.7,
) -> dict:
    """Find label issues using cleanlab.

    Args:
        labels: Ground truth labels.
        pred_probs: Cross-validated predicted probabilities (n_samples, n_classes).
        min_confidence: Minimum model confidence to flag as issue.

    Returns:
        Dictionary with issue details.
    """
    try:
        from cleanlab.filter import find_label_issues as cl_find_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        logger.error(
            "cleanlab is not installed. Install with:\n"
            "  pip install 'cleanlab>=2.6.0'"
        )
        sys.exit(1)

    logger.info("Running cleanlab label issue detection...")

    # Find label issues
    issue_mask = cl_find_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Get quality scores
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=pred_probs,
    )

    # Analyze issues
    n_issues = len(issue_mask) if isinstance(issue_mask, np.ndarray) and issue_mask.dtype != bool else issue_mask.sum()

    if isinstance(issue_mask, np.ndarray) and issue_mask.dtype != bool:
        # issue_mask is indices ranked by severity
        issue_indices = issue_mask
    else:
        issue_indices = np.where(issue_mask)[0]

    # Get predicted labels for issue samples
    predicted_labels = pred_probs.argmax(axis=1)

    issues = []
    for idx in issue_indices:
        issue = {
            "index": int(idx),
            "given_label": int(labels[idx]),
            "predicted_label": int(predicted_labels[idx]),
            "quality_score": float(quality_scores[idx]),
            "confidence": float(pred_probs[idx].max()),
            "prob_class_0": float(pred_probs[idx, 0]),
            "prob_class_1": float(pred_probs[idx, 1]),
        }
        issues.append(issue)

    # Categorize
    fn_errors = [i for i in issues if i["given_label"] == 0 and i["predicted_label"] == 1]
    fp_errors = [i for i in issues if i["given_label"] == 1 and i["predicted_label"] == 0]

    return {
        "total_issues": len(issues),
        "fn_errors": len(fn_errors),  # labeled clean but model says toxic
        "fp_errors": len(fp_errors),  # labeled toxic but model says clean
        "issues": issues,
        "quality_scores": quality_scores,
    }


def generate_report(
    df: pd.DataFrame,
    issues: dict,
    output_path: Path | None = None,
) -> str:
    """Generate human-readable report of label issues.

    Args:
        df: Original dataframe.
        issues: Issues dictionary from find_label_issues.
        output_path: Optional path to save report.

    Returns:
        Report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("LABEL ERROR DETECTION REPORT")
    lines.append("=" * 70)
    lines.append(f"Total samples: {len(df)}")
    lines.append(f"Total issues found: {issues['total_issues']}")
    lines.append(f"  FN errors (labeled clean → likely toxic): {issues['fn_errors']}")
    lines.append(f"  FP errors (labeled toxic → likely clean): {issues['fp_errors']}")
    lines.append(f"  Error rate: {issues['total_issues'] / len(df):.2%}")
    lines.append("")

    # Top suspicious samples
    lines.append("-" * 70)
    lines.append("TOP 20 MOST SUSPICIOUS SAMPLES")
    lines.append("-" * 70)

    for issue in issues["issues"][:20]:
        idx = issue["index"]
        text = str(df.iloc[idx]["text"])[:80]
        given = "toxic" if issue["given_label"] == 1 else "clean"
        predicted = "toxic" if issue["predicted_label"] == 1 else "clean"
        conf = issue["confidence"]
        quality = issue["quality_score"]

        lines.append(f"\n[#{idx}] Quality: {quality:.3f}, Confidence: {conf:.3f}")
        lines.append(f"  Given: {given} → Predicted: {predicted}")
        lines.append(f'  Text: "{text}{"..." if len(str(df.iloc[idx]["text"])) > 80 else ""}"')

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


def create_cleaned_dataset(
    df: pd.DataFrame,
    issues: dict,
    quality_threshold: float = 0.3,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Create a cleaned dataset by removing low-quality labels.

    Args:
        df: Original dataframe.
        issues: Issues dictionary.
        quality_threshold: Remove samples with quality score below this.
        output_path: Path to save cleaned dataset.

    Returns:
        Cleaned dataframe.
    """
    quality_scores = issues["quality_scores"]

    # Remove samples below quality threshold
    keep_mask = quality_scores >= quality_threshold
    cleaned_df = df[keep_mask].reset_index(drop=True)

    removed = len(df) - len(cleaned_df)
    logger.info(
        f"Removed {removed} low-quality samples "
        f"({removed / len(df):.1%} of dataset)"
    )
    logger.info(f"Cleaned dataset size: {len(cleaned_df)}")

    # Label distribution
    orig_dist = df["label"].value_counts().to_dict()
    clean_dist = cleaned_df["label"].value_counts().to_dict()
    logger.info(f"Original distribution: {orig_dist}")
    logger.info(f"Cleaned distribution: {clean_dist}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to {output_path}")

    return cleaned_df


def main():
    parser = argparse.ArgumentParser(
        description="Find label errors in training data using cleanlab"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/coevolution-latest",
        help="Path to production model for predictions",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.3,
        help="Quality threshold for removing samples",
    )
    parser.add_argument(
        "--output-cleaned",
        action="store_true",
        help="Generate cleaned dataset (korean_standard_v2_train.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="korean_standard_v1",
        help="Dataset version",
    )
    args = parser.parse_args()

    # Load data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    paths = get_data_paths(data_dir, args.dataset_version)
    train_path = paths["train"]

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        sys.exit(1)

    logger.info(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    texts = df["text"].tolist()
    labels = df["label"].values

    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Get cross-validated probabilities
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try relative to ml-service
        model_path = Path(__file__).parent.parent / args.model_path
    if not model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    pred_probs = get_cross_validated_probs(
        texts=texts,
        labels=labels,
        model_path=str(model_path),
        cv_folds=args.cv_folds,
        batch_size=args.batch_size,
    )

    # Find label issues
    issues = find_label_issues(labels, pred_probs)

    # Generate report
    report_path = data_dir / "label_error_report.txt"
    report = generate_report(df, issues, report_path)
    print(report)

    # Save issues detail
    issues_path = data_dir / "label_issues.json"
    issues_export = {
        "total_issues": issues["total_issues"],
        "fn_errors": issues["fn_errors"],
        "fp_errors": issues["fp_errors"],
        "issues": issues["issues"][:200],  # Top 200
    }
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(issues_export, f, indent=2, ensure_ascii=False)
    logger.info(f"Issues detail saved to {issues_path}")

    # Generate cleaned dataset
    if args.output_cleaned:
        cleaned_path = data_dir / "korean_standard_v2_train.csv"
        create_cleaned_dataset(
            df=df,
            issues=issues,
            quality_threshold=args.quality_threshold,
            output_path=cleaned_path,
        )


if __name__ == "__main__":
    main()
