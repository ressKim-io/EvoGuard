#!/usr/bin/env python3
"""Evaluate all models on the standard test set for fair comparison.

Evaluates:
- PMF individual models (kcelectra, klue-bert, koelectra-v3)
- PMF meta-learner ensemble
- Phase 2 Combined (if exists)
- Coevolution Latest (if exists)

All models are evaluated on the same korean_standard_v1_test.csv for fair comparison.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from ml_service.training.standard_config import evaluate_model, get_data_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def get_predictions(model_path: Path, texts: list, device: torch.device, batch_size: int = 32):
    """Get predictions from a transformer model."""
    if not model_path.exists():
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=model_path.parent.name):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    return np.array(all_probs), np.array(all_preds)


def evaluate_pmf_ensemble(model_probs: dict, labels: np.ndarray, meta_learner_path: Path = None):
    """Evaluate PMF ensemble strategies."""
    results = {}

    # Get available model predictions
    available = {k: v for k, v in model_probs.items() if v is not None}

    if len(available) < 2:
        logger.warning("Not enough models for ensemble")
        return results

    # Stack predictions
    model_names = list(available.keys())
    probs_stack = np.column_stack([available[k] for k in model_names])

    # Strategy 1: Simple Average
    avg_probs = probs_stack.mean(axis=1)
    avg_preds = (avg_probs >= 0.5).astype(int)
    results["pmf_average"] = evaluate_model(labels.tolist(), avg_preds.tolist(), avg_probs.tolist())

    # Strategy 2: Max (OR-like)
    max_probs = probs_stack.max(axis=1)
    max_preds = (max_probs >= 0.5).astype(int)
    results["pmf_max"] = evaluate_model(labels.tolist(), max_preds.tolist(), max_probs.tolist())

    # Strategy 3: Voting
    individual_preds = (probs_stack >= 0.5).astype(int)
    vote_preds = (individual_preds.sum(axis=1) >= 2).astype(int)  # Majority
    results["pmf_voting"] = evaluate_model(labels.tolist(), vote_preds.tolist())

    # Strategy 4: Meta-learner (if available)
    if meta_learner_path and meta_learner_path.exists():
        with open(meta_learner_path, "rb") as f:
            meta_learner = pickle.load(f)

        meta_probs = meta_learner.predict_proba(probs_stack)[:, 1]
        meta_preds = meta_learner.predict(probs_stack)
        results["pmf_meta_learner"] = evaluate_model(labels.tolist(), meta_preds.tolist(), meta_probs.tolist())

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on test set")
    parser.add_argument("--test-file", type=str, default=None, help="Test file path")
    parser.add_argument("--output", type=str, default="models/evaluation_results.json", help="Output file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test data
    if args.test_file:
        test_path = Path(args.test_file)
    else:
        paths = get_data_paths()
        test_path = paths["test"]

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        sys.exit(1)

    logger.info(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    texts = test_df["text"].tolist()
    labels = test_df["label"].values

    logger.info(f"Test samples: {len(texts)}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Model paths
    base_dir = Path(__file__).parent.parent / "models"
    models_to_evaluate = {
        # PMF models
        "kcelectra": base_dir / "pmf" / "kcelectra" / "best_model",
        "klue-bert": base_dir / "pmf" / "klue-bert" / "best_model",
        "koelectra-v3": base_dir / "pmf" / "koelectra-v3" / "best_model",
        # Other models (best_model subdirectory)
        "phase2-combined": base_dir / "phase2-combined" / "best_model",
        "coevolution-latest": base_dir / "coevolution-latest" / "best_model",
    }

    meta_learner_path = base_dir / "pmf" / "meta_learner.pkl"

    # Evaluate individual models
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATING INDIVIDUAL MODELS ON TEST SET")
    logger.info("=" * 60)

    all_results = {}
    model_probs = {}

    for name, path in models_to_evaluate.items():
        if not path.exists():
            logger.warning(f"Model not found: {name} ({path})")
            model_probs[name] = None
            continue

        logger.info(f"\nEvaluating {name}...")
        probs, preds = get_predictions(path, texts, device, args.batch_size)

        if probs is not None:
            metrics = evaluate_model(labels.tolist(), preds.tolist(), probs.tolist())
            all_results[name] = metrics
            model_probs[name] = probs

            logger.info(f"  F1: {metrics['f1_weighted']:.4f}, FP: {metrics['fp']}, FN: {metrics['fn']}")

    # Evaluate PMF ensemble strategies
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATING PMF ENSEMBLE STRATEGIES")
    logger.info("=" * 60)

    pmf_probs = {k: model_probs.get(k) for k in ["kcelectra", "klue-bert", "koelectra-v3"]}
    ensemble_results = evaluate_pmf_ensemble(pmf_probs, labels, meta_learner_path)
    all_results.update(ensemble_results)

    for name, metrics in ensemble_results.items():
        logger.info(f"  {name}: F1={metrics['f1_weighted']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS SUMMARY (TEST SET)")
    logger.info("=" * 60)

    # Sort by F1 score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["f1_weighted"], reverse=True)

    logger.info(f"{'Model':<25} {'F1':>8} {'FP':>6} {'FN':>6} {'Acc':>8}")
    logger.info("-" * 60)
    for name, metrics in sorted_results:
        logger.info(
            f"{name:<25} {metrics['f1_weighted']:>8.4f} {metrics['fp']:>6} "
            f"{metrics['fn']:>6} {metrics['accuracy']:>8.4f}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_file": str(test_path),
        "test_samples": len(texts),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
        "results": all_results,
        "ranking": [{"rank": i+1, "model": name, "f1": metrics["f1_weighted"]}
                    for i, (name, metrics) in enumerate(sorted_results)],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path}")

    # Best model
    best_name, best_metrics = sorted_results[0]
    logger.info(f"\n*** BEST MODEL: {best_name} (F1: {best_metrics['f1_weighted']:.4f}) ***")


if __name__ == "__main__":
    main()
