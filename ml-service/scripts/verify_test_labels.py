#!/usr/bin/env python3
"""Verify test set labels using multi-model consensus.

Uses 4 models (Production + PMF 3 models) to identify likely label errors
in the test set. Unlike find_label_errors.py (which uses CV on train set),
this directly runs inference with pre-trained models.

Usage:
    # Basic run
    python scripts/verify_test_labels.py

    # Custom paths
    python scripts/verify_test_labels.py \
      --production-model models/production \
      --pmf-dir models/pmf \
      --batch-size 32

    # Apply corrections from reviewed CSV
    python scripts/verify_test_labels.py --apply-corrections data/korean/label_verification/test_label_review.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_service.training.standard_config import get_data_paths, evaluate_model

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


def get_model_probs(
    texts: list[str],
    model_path: str,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Get predicted probabilities from a single model.

    Args:
        texts: List of texts.
        model_path: Path to the model.
        batch_size: Batch size for inference.
        device: Device to use.

    Returns:
        Array of shape (N, 2) with [prob_clean, prob_toxic].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = SimpleDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference ({Path(model_path).name})", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    del model
    torch.cuda.empty_cache()

    return np.vstack(all_probs)


def get_production_model_probs(
    texts: list[str],
    model_path: str,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Get probabilities from the production model.

    Returns:
        Array of shape (N, 2).
    """
    logger.info(f"Running production model inference: {model_path}")
    return get_model_probs(texts, model_path, batch_size, device)


def get_pmf_individual_probs(
    texts: list[str],
    pmf_dir: str,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Get probabilities from each PMF model individually.

    Args:
        texts: List of texts.
        pmf_dir: PMF models directory (contains kcelectra/, klue-bert/, koelectra-v3/).
        batch_size: Batch size.
        device: Device.

    Returns:
        Dict mapping model name to (N, 2) probability arrays.
    """
    pmf_dir = Path(pmf_dir)
    pmf_models = {
        "kcelectra": pmf_dir / "kcelectra" / "best_model",
        "klue-bert": pmf_dir / "klue-bert" / "best_model",
        "koelectra-v3": pmf_dir / "koelectra-v3" / "best_model",
    }

    results = {}
    for name, path in pmf_models.items():
        if not path.exists():
            logger.warning(f"PMF model not found, skipping: {path}")
            continue
        logger.info(f"Running PMF model inference: {name}")
        results[name] = get_model_probs(texts, str(path), batch_size, device)

    if not results:
        raise FileNotFoundError(f"No PMF models found in {pmf_dir}")

    return results


def compute_consensus_signals(
    labels: np.ndarray,
    prod_probs: np.ndarray,
    pmf_probs: dict[str, np.ndarray],
) -> list[dict]:
    """Analyze multi-model consensus for each sample.

    Priority levels:
        P1: 4/4 models disagree with label, avg confidence > 0.8
        P2: 3/3 PMF models disagree with label
        P3: 2/3 PMF + Production disagree
        P4: 2/3 PMF or Production high-confidence (>0.9) disagree

    Args:
        labels: Ground truth labels (N,).
        prod_probs: Production model probs (N, 2).
        pmf_probs: Dict of PMF model probs, each (N, 2).

    Returns:
        List of dicts with consensus signals per sample.
    """
    n = len(labels)
    pmf_names = list(pmf_probs.keys())
    n_pmf = len(pmf_names)

    signals = []
    for i in range(n):
        label = int(labels[i])
        prod_pred = int(prod_probs[i].argmax())
        prod_conf = float(prod_probs[i].max())
        prod_toxic_prob = float(prod_probs[i, 1])

        pmf_preds = {}
        pmf_confs = {}
        pmf_toxic_probs = {}
        for name in pmf_names:
            probs = pmf_probs[name][i]
            pmf_preds[name] = int(probs.argmax())
            pmf_confs[name] = float(probs.max())
            pmf_toxic_probs[name] = float(probs[1])

        # Count disagreements
        prod_disagrees = prod_pred != label
        pmf_disagree_count = sum(1 for name in pmf_names if pmf_preds[name] != label)
        all_pmf_disagree = pmf_disagree_count == n_pmf
        n_disagree = int(prod_disagrees) + pmf_disagree_count
        total_models = 1 + n_pmf

        # Average confidences of disagreeing models
        disagree_confs = []
        if prod_disagrees:
            disagree_confs.append(prod_conf)
        for name in pmf_names:
            if pmf_preds[name] != label:
                disagree_confs.append(pmf_confs[name])
        avg_disagree_conf = float(np.mean(disagree_confs)) if disagree_confs else 0.0

        # Average PMF confidence (among those disagreeing)
        pmf_disagree_confs = [pmf_confs[name] for name in pmf_names if pmf_preds[name] != label]
        avg_pmf_disagree_conf = float(np.mean(pmf_disagree_confs)) if pmf_disagree_confs else 0.0

        # Determine priority
        priority = None
        if n_disagree == total_models and avg_disagree_conf > 0.8:
            priority = 1
        elif all_pmf_disagree and n_pmf >= 3:
            priority = 2
        elif pmf_disagree_count >= 2 and prod_disagrees:
            priority = 3
        elif pmf_disagree_count >= 2 or (prod_disagrees and prod_conf > 0.9):
            priority = 4

        signal = {
            "index": i,
            "label": label,
            "prod_pred": prod_pred,
            "prod_conf": prod_conf,
            "prod_toxic_prob": prod_toxic_prob,
            "prod_disagrees": prod_disagrees,
            "n_disagree": n_disagree,
            "total_models": total_models,
            "all_pmf_disagree": all_pmf_disagree,
            "pmf_disagree_count": pmf_disagree_count,
            "avg_disagree_conf": avg_disagree_conf,
            "avg_pmf_disagree_conf": avg_pmf_disagree_conf,
            "priority": priority,
        }
        for name in pmf_names:
            signal[f"pmf_{name}_pred"] = pmf_preds[name]
            signal[f"pmf_{name}_conf"] = pmf_confs[name]
            signal[f"pmf_{name}_toxic_prob"] = pmf_toxic_probs[name]

        signals.append(signal)

    return signals


def run_cleanlab_audit(
    labels: np.ndarray,
    prod_probs: np.ndarray,
    pmf_probs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Run cleanlab using averaged probabilities from all models.

    Args:
        labels: Ground truth labels.
        prod_probs: Production model probs (N, 2).
        pmf_probs: Dict of PMF model probs.

    Returns:
        Tuple of (quality_scores, issue_mask) arrays.
    """
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        logger.warning(
            "cleanlab not installed. Skipping cleanlab audit. "
            "Install with: pip install 'cleanlab>=2.6.0'"
        )
        n = len(labels)
        return np.ones(n), np.zeros(n, dtype=bool)

    # Average probabilities from all models
    all_probs = [prod_probs]
    for name in pmf_probs:
        all_probs.append(pmf_probs[name])
    avg_probs = np.mean(all_probs, axis=0)

    # Ensure probabilities sum to 1
    avg_probs = avg_probs / avg_probs.sum(axis=1, keepdims=True)

    logger.info("Running cleanlab quality scoring on averaged probabilities...")
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=avg_probs,
    )

    return quality_scores, avg_probs


def combine_signals(
    signals: list[dict],
    quality_scores: np.ndarray,
) -> list[dict]:
    """Combine consensus signals with cleanlab quality scores.

    Suspicion score formula:
        0.35 * (1 - cleanlab_quality) +
        0.25 * all_pmf_disagree * avg_pmf_confidence +
        0.20 * prod_disagrees * prod_confidence +
        0.20 * (n_disagree / total_models)

    Args:
        signals: Consensus signals from compute_consensus_signals.
        quality_scores: Cleanlab quality scores.

    Returns:
        Updated signals with suspicion scores, sorted by suspicion descending.
    """
    for sig in signals:
        i = sig["index"]
        cl_quality = float(quality_scores[i])

        suspicion = (
            0.35 * (1.0 - cl_quality)
            + 0.25 * float(sig["all_pmf_disagree"]) * sig["avg_pmf_disagree_conf"]
            + 0.20 * float(sig["prod_disagrees"]) * sig["prod_conf"]
            + 0.20 * (sig["n_disagree"] / sig["total_models"])
        )

        sig["cleanlab_quality"] = cl_quality
        sig["suspicion"] = round(suspicion, 4)

    # Sort by suspicion descending
    signals.sort(key=lambda x: x["suspicion"], reverse=True)
    return signals


def generate_report(
    df: pd.DataFrame,
    signals: list[dict],
    suspicion_threshold: float,
    output_path: Path | None = None,
) -> str:
    """Generate human-readable verification report.

    Args:
        df: Test dataframe with text and label columns.
        signals: Combined signals sorted by suspicion.
        suspicion_threshold: Threshold to flag as suspicious.
        output_path: Optional path to save report.

    Returns:
        Report string.
    """
    suspicious = [s for s in signals if s["suspicion"] >= suspicion_threshold]
    by_priority = {}
    for s in suspicious:
        p = s["priority"]
        if p is not None:
            by_priority.setdefault(p, []).append(s)

    # Count label types in suspicious
    fn_suspects = [s for s in suspicious if s["label"] == 0 and s["n_disagree"] > 0]  # labeled clean, models say toxic
    fp_suspects = [s for s in suspicious if s["label"] == 1 and s["n_disagree"] > 0]  # labeled toxic, models say clean

    lines = []
    lines.append("=" * 70)
    lines.append("TEST SET LABEL VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total test samples: {len(df)}")
    lines.append(f"Suspicion threshold: {suspicion_threshold}")
    lines.append(f"Suspicious samples: {len(suspicious)} ({len(suspicious)/len(df):.2%})")
    lines.append(f"  Labeled clean → likely toxic (FN suspects): {len(fn_suspects)}")
    lines.append(f"  Labeled toxic → likely clean (FP suspects): {len(fp_suspects)}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("PRIORITY BREAKDOWN")
    lines.append("-" * 70)
    priority_desc = {
        1: "P1: All 4 models disagree (conf > 0.8) - VERY LIKELY MISLABELED",
        2: "P2: All 3 PMF models disagree",
        3: "P3: 2/3 PMF + Production disagree",
        4: "P4: 2/3 PMF or Production high-conf (>0.9) disagree",
    }
    for p in [1, 2, 3, 4]:
        count = len(by_priority.get(p, []))
        lines.append(f"  {priority_desc.get(p, f'P{p}')}: {count}")
    no_priority = len([s for s in suspicious if s["priority"] is None])
    if no_priority:
        lines.append(f"  No priority (cleanlab-only): {no_priority}")
    lines.append("")

    # Show top suspicious samples
    lines.append("-" * 70)
    lines.append("TOP 30 MOST SUSPICIOUS SAMPLES")
    lines.append("-" * 70)

    for sig in suspicious[:30]:
        idx = sig["index"]
        text = str(df.iloc[idx]["text"])[:80]
        label_text = "toxic" if sig["label"] == 1 else "clean"
        prod_text = "toxic" if sig["prod_pred"] == 1 else "clean"

        lines.append(f"\n[#{idx}] Suspicion: {sig['suspicion']:.3f}, Priority: P{sig['priority'] or '-'}")
        lines.append(f"  Label: {label_text} → Models say: {prod_text} ({sig['n_disagree']}/{sig['total_models']} disagree)")
        lines.append(f"  Production: conf={sig['prod_conf']:.3f}, CL quality={sig['cleanlab_quality']:.3f}")
        lines.append(f'  Text: "{text}{"..." if len(str(df.iloc[idx]["text"])) > 80 else ""}"')

    # Summary statistics
    lines.append("")
    lines.append("-" * 70)
    lines.append("OVERALL MODEL AGREEMENT")
    lines.append("-" * 70)
    agree_counts = [0, 0, 0, 0, 0]  # 0-4 models agree with label
    for s in signals:
        n_agree = s["total_models"] - s["n_disagree"]
        if n_agree < len(agree_counts):
            agree_counts[n_agree] += 1
    for n, count in enumerate(agree_counts):
        lines.append(f"  {n}/{signals[0]['total_models'] if signals else 4} models agree with label: {count}")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


def generate_review_csv(
    df: pd.DataFrame,
    signals: list[dict],
    suspicion_threshold: float,
    output_path: Path,
    max_samples: int | None = None,
) -> None:
    """Generate CSV file for manual review.

    Args:
        df: Test dataframe.
        signals: Combined signals sorted by suspicion.
        suspicion_threshold: Threshold to include.
        output_path: Output CSV path.
        max_samples: Max samples to export.
    """
    suspicious = [s for s in signals if s["suspicion"] >= suspicion_threshold]
    if max_samples:
        suspicious = suspicious[:max_samples]

    rows = []
    for sig in suspicious:
        idx = sig["index"]
        row = {
            "index": idx,
            "text": str(df.iloc[idx]["text"]),
            "current_label": sig["label"],
            "current_label_text": "toxic" if sig["label"] == 1 else "clean",
            "suspicion_score": sig["suspicion"],
            "priority": sig["priority"],
            "n_disagree": sig["n_disagree"],
            "total_models": sig["total_models"],
            "prod_pred": sig["prod_pred"],
            "prod_conf": sig["prod_conf"],
            "cleanlab_quality": sig["cleanlab_quality"],
            "corrected_label": "",
            "reviewer_notes": "",
        }
        rows.append(row)

    review_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Review CSV saved to {output_path} ({len(review_df)} samples)")


def calculate_corrected_f1(
    labels: np.ndarray,
    prod_probs: np.ndarray,
    signals: list[dict],
) -> dict:
    """Estimate corrected F1 scores if suspected labels are fixed.

    Simulates correction by flipping labels for P1, P2, P3 suspicious samples
    and recalculating F1 with the production model predictions.

    Args:
        labels: Original labels.
        prod_probs: Production model probabilities.
        signals: Combined signals.

    Returns:
        Dict with corrected F1 estimates per priority level.
    """
    prod_preds = prod_probs.argmax(axis=1).tolist()
    original_metrics = evaluate_model(labels.tolist(), prod_preds)

    results = {
        "original": {
            "f1_weighted": original_metrics["f1_weighted"],
            "fp": original_metrics["fp"],
            "fn": original_metrics["fn"],
        }
    }

    # Build priority index
    priority_indices = {}
    for sig in signals:
        p = sig["priority"]
        if p is not None:
            priority_indices.setdefault(p, []).append(sig["index"])

    # Cumulative corrections: P1 → P1+P2 → P1+P2+P3
    corrected_labels = labels.copy()
    for max_p, label in [(1, "p1"), (2, "p1_p2"), (3, "p1_p2_p3")]:
        for p in range(1, max_p + 1):
            for idx in priority_indices.get(p, []):
                # Flip the label
                corrected_labels[idx] = 1 - corrected_labels[idx]

        corrected_metrics = evaluate_model(corrected_labels.tolist(), prod_preds)
        n_corrected = sum(len(priority_indices.get(p, [])) for p in range(1, max_p + 1))

        results[label] = {
            "f1_weighted": corrected_metrics["f1_weighted"],
            "fp": corrected_metrics["fp"],
            "fn": corrected_metrics["fn"],
            "n_corrected": n_corrected,
        }

        # Reset for next iteration
        corrected_labels = labels.copy()

    return results


def apply_corrections(
    test_path: Path,
    review_csv_path: Path,
    output_path: Path,
) -> None:
    """Apply corrections from reviewed CSV to create a corrected test set.

    Args:
        test_path: Original test CSV.
        review_csv_path: Reviewed CSV with corrected_label filled in.
        output_path: Output path for corrected test set.
    """
    df = pd.read_csv(test_path)
    review_df = pd.read_csv(review_csv_path)

    n_corrections = 0
    for _, row in review_df.iterrows():
        corrected = row.get("corrected_label")
        if pd.isna(corrected) or corrected == "":
            continue
        corrected = int(corrected)
        idx = int(row["index"])
        if df.iloc[idx]["label"] != corrected:
            df.at[idx, "label"] = corrected
            n_corrections += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Applied {n_corrections} corrections → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify test set labels using multi-model consensus"
    )
    parser.add_argument(
        "--production-model",
        type=str,
        default="models/production",
        help="Path to production model",
    )
    parser.add_argument(
        "--pmf-dir",
        type=str,
        default="models/pmf",
        help="PMF models directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--suspicion-threshold",
        type=float,
        default=0.3,
        help="Suspicion score threshold to flag samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/korean/label_verification",
        help="Output directory for results",
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
        "--apply-corrections",
        type=str,
        default=None,
        help="Path to reviewed CSV to apply corrections",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    # Handle --apply-corrections mode
    if args.apply_corrections:
        paths = get_data_paths(data_dir, args.dataset_version)
        test_path = paths["test"]
        review_path = Path(args.apply_corrections)
        output_path = data_dir / f"{args.dataset_version}_test_corrected.csv"
        apply_corrections(test_path, review_path, output_path)
        return

    # Load test set
    paths = get_data_paths(data_dir, args.dataset_version)
    test_path = paths["test"]

    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        sys.exit(1)

    logger.info(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    texts = df["text"].tolist()
    labels = df["label"].values

    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution: {dict(pd.Series(labels).value_counts().sort_index())}")

    # Resolve model paths
    prod_path = Path(args.production_model)
    if not prod_path.exists():
        prod_path = Path(__file__).parent.parent / args.production_model
    if not prod_path.exists():
        logger.error(f"Production model not found: {args.production_model}")
        sys.exit(1)

    pmf_dir = Path(args.pmf_dir)
    if not pmf_dir.exists():
        pmf_dir = Path(__file__).parent.parent / args.pmf_dir
    if not pmf_dir.exists():
        logger.error(f"PMF directory not found: {args.pmf_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Step 1: Get probabilities from all models
    prod_probs = get_production_model_probs(texts, str(prod_path), args.batch_size, device)
    pmf_probs = get_pmf_individual_probs(texts, str(pmf_dir), args.batch_size, device)

    logger.info(f"Models used: production + {list(pmf_probs.keys())}")

    # Step 2: Compute consensus signals
    logger.info("Computing consensus signals...")
    signals = compute_consensus_signals(labels, prod_probs, pmf_probs)

    # Step 3: Run cleanlab audit
    quality_scores, avg_probs = run_cleanlab_audit(labels, prod_probs, pmf_probs)

    # Step 4: Combine signals
    logger.info("Combining signals into suspicion scores...")
    signals = combine_signals(signals, quality_scores)

    # Step 5: Generate outputs
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir

    # Report
    report = generate_report(df, signals, args.suspicion_threshold, output_dir / "test_label_verification_report.txt")
    print(report)

    # Review CSV
    generate_review_csv(df, signals, args.suspicion_threshold, output_dir / "test_label_review.csv")

    # Full issues JSON
    suspicious = [s for s in signals if s["suspicion"] >= args.suspicion_threshold]
    issues_export = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset_version,
        "total_samples": len(df),
        "suspicion_threshold": args.suspicion_threshold,
        "total_suspicious": len(suspicious),
        "by_priority": {
            str(p): len([s for s in suspicious if s["priority"] == p])
            for p in [1, 2, 3, 4, None]
        },
        "issues": [
            {k: v for k, v in s.items() if not isinstance(v, (np.integer, np.floating)) or True}
            for s in suspicious[:200]
        ],
    }
    issues_path = output_dir / "test_label_issues.json"
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(issues_export, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Issues JSON saved to {issues_path}")

    # Corrected F1 estimates
    corrected_f1 = calculate_corrected_f1(labels, prod_probs, signals)
    f1_path = output_dir / "corrected_f1_estimates.json"
    with open(f1_path, "w", encoding="utf-8") as f:
        json.dump(corrected_f1, f, indent=2)
    logger.info(f"Corrected F1 estimates saved to {f1_path}")

    # Print F1 summary
    print("\n" + "=" * 70)
    print("CORRECTED F1 ESTIMATES")
    print("=" * 70)
    for level, metrics in corrected_f1.items():
        n = metrics.get("n_corrected", 0)
        print(f"  {level}: F1={metrics['f1_weighted']:.4f}, FP={metrics['fp']}, FN={metrics['fn']}"
              + (f", corrections={n}" if n else ""))


if __name__ == "__main__":
    main()
