#!/usr/bin/env python3
"""Active Learning Loop for boundary case identification.

Identifies samples where model confidence is low (0.4~0.6) and exports them
for human review. Uses the existing BoundarySampleCollector infrastructure.

Usage:
    # Find boundary cases in training data
    python scripts/active_learning.py

    # Custom confidence range
    python scripts/active_learning.py --low 0.35 --high 0.65

    # Export specific number of samples
    python scripts/active_learning.py --max-samples 200

    # Use specific model
    python scripts/active_learning.py --model-path models/coevolution-latest
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


def find_boundary_cases(
    texts: list[str],
    labels: list[int],
    model_path: str,
    confidence_low: float = 0.4,
    confidence_high: float = 0.6,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    """Find samples where model confidence is in the boundary zone.

    Args:
        texts: List of texts.
        labels: List of ground truth labels.
        model_path: Path to the model.
        confidence_low: Lower bound of boundary zone.
        confidence_high: Upper bound of boundary zone.
        batch_size: Batch size for inference.
        device: Device to use.

    Returns:
        DataFrame with boundary cases and their predictions.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = SimpleDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)

    del model
    torch.cuda.empty_cache()

    # Find boundary cases
    boundary_indices = []
    for i, prob in enumerate(all_probs):
        confidence = max(prob, 1 - prob)
        if confidence_low <= confidence <= confidence_high:
            boundary_indices.append(i)

    logger.info(
        f"Found {len(boundary_indices)} boundary cases "
        f"(confidence {confidence_low:.0%}-{confidence_high:.0%})"
    )

    # Build result DataFrame
    results = []
    for idx in boundary_indices:
        toxic_prob = all_probs[idx]
        predicted_label = 1 if toxic_prob > 0.5 else 0
        confidence = max(toxic_prob, 1 - toxic_prob)

        results.append({
            "index": idx,
            "text": texts[idx],
            "given_label": labels[idx],
            "predicted_label": predicted_label,
            "toxic_prob": round(toxic_prob, 4),
            "confidence": round(confidence, 4),
            "agrees_with_label": int(predicted_label == labels[idx]),
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        # Sort by confidence (lowest first = most uncertain)
        result_df = result_df.sort_values("confidence").reset_index(drop=True)

    return result_df


def generate_review_csv(
    boundary_df: pd.DataFrame,
    output_path: Path,
    max_samples: int | None = None,
) -> None:
    """Export boundary cases as CSV for human review.

    Args:
        boundary_df: DataFrame with boundary cases.
        output_path: Path to save review CSV.
        max_samples: Maximum number of samples to export.
    """
    if boundary_df.empty:
        logger.warning("No boundary cases to export.")
        return

    export_df = boundary_df.copy()
    if max_samples and len(export_df) > max_samples:
        export_df = export_df.head(max_samples)

    # Add review columns
    export_df["corrected_label"] = ""
    export_df["reviewer_notes"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"Exported {len(export_df)} samples for review to {output_path}")


def analyze_boundary_cases(boundary_df: pd.DataFrame) -> dict:
    """Analyze the distribution of boundary cases.

    Args:
        boundary_df: DataFrame with boundary cases.

    Returns:
        Analysis dictionary.
    """
    if boundary_df.empty:
        return {"total": 0}

    total = len(boundary_df)
    agrees = boundary_df["agrees_with_label"].sum()
    disagrees = total - agrees

    labeled_toxic = (boundary_df["given_label"] == 1).sum()
    labeled_clean = (boundary_df["given_label"] == 0).sum()

    analysis = {
        "total_boundary_cases": total,
        "agrees_with_label": int(agrees),
        "disagrees_with_label": int(disagrees),
        "agreement_rate": round(agrees / total, 4) if total > 0 else 0,
        "labeled_toxic": int(labeled_toxic),
        "labeled_clean": int(labeled_clean),
        "avg_confidence": round(boundary_df["confidence"].mean(), 4),
        "min_confidence": round(boundary_df["confidence"].min(), 4),
        "max_confidence": round(boundary_df["confidence"].max(), 4),
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning: find and export boundary cases for review"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/coevolution-latest",
        help="Path to model for predictions",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.4,
        help="Lower confidence bound for boundary zone",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=0.6,
        help="Upper confidence bound for boundary zone",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to export",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
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
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="train",
        help="Data split to analyze",
    )
    args = parser.parse_args()

    # Load data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    paths = get_data_paths(data_dir, args.dataset_version)
    data_path = paths[args.split]

    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading {args.split} data from {data_path}...")
    df = pd.read_csv(data_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    logger.info(f"Loaded {len(df)} samples")

    # Resolve model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / args.model_path
    if not model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Find boundary cases
    boundary_df = find_boundary_cases(
        texts=texts,
        labels=labels,
        model_path=str(model_path),
        confidence_low=args.low,
        confidence_high=args.high,
        batch_size=args.batch_size,
    )

    # Analyze
    analysis = analyze_boundary_cases(boundary_df)
    print("\n" + "=" * 60)
    print("BOUNDARY CASE ANALYSIS")
    print("=" * 60)
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Export for review
    output_path = data_dir / f"active_learning_review_{args.split}.csv"
    generate_review_csv(boundary_df, output_path, max_samples=args.max_samples)

    # Save analysis
    analysis_path = data_dir / f"active_learning_analysis_{args.split}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {analysis_path}")

    print(f"\nReview file: {output_path}")
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
