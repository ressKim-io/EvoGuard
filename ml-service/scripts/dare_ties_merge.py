"""DARE-TIES Model Merging Script.

References:
- TIES-Merging: Yadav et al., NeurIPS 2023
- DARE: Yu et al., 2024 (Drop And REscale)
- ACM Computing Surveys 2026: Model Merging Survey

Usage:
    python scripts/dare_ties_merge.py --checkpoints phase1_best phase3_best phase4_best
    python scripts/dare_ties_merge.py --top-k 3 --dare-rate 0.9
    python scripts/dare_ties_merge.py --evaluate
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load model state dict from checkpoint."""
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
    return model.state_dict()


def compute_task_vector(
    fine_tuned: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute delta (task vector) = fine_tuned - base."""
    return {k: fine_tuned[k] - base[k] for k in base if k in fine_tuned}


def dare_drop(
    task_vector: dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """DARE: randomly drop delta parameters and rescale remaining ones.

    Drop rate 0.9 means 90% of deltas are set to zero, remaining 10% are
    rescaled by 1/(1-drop_rate) = 10x to maintain expected magnitude.
    """
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for k, v in task_vector.items():
        mask = torch.bernoulli(torch.full_like(v, 1.0 - drop_rate), generator=generator)
        # Rescale remaining params to preserve expected value
        scale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        result[k] = v * mask * scale
    return result


def ties_trim(
    task_vectors: list[dict[str, torch.Tensor]],
    density: float = 0.2,
) -> list[dict[str, torch.Tensor]]:
    """TIES Step 1: Trim low-magnitude deltas.

    Keep only top `density` fraction of parameters by magnitude.
    """
    trimmed = []
    for tv in task_vectors:
        trimmed_tv = {}
        for k, v in tv.items():
            flat = v.abs().float().flatten()
            k_keep = max(1, int(len(flat) * density))
            if len(flat) > 0:
                threshold = flat.topk(k_keep).values[-1]
            else:
                threshold = 0.0
            mask = v.abs() >= threshold
            trimmed_tv[k] = v * mask
        trimmed.append(trimmed_tv)
    return trimmed


def ties_elect_sign(
    task_vectors: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """TIES Step 2: Elect sign by majority vote across models."""
    elected_signs = {}
    keys = task_vectors[0].keys()

    for k in keys:
        # Sum signs across all task vectors
        sign_sum = sum(tv[k].sign() for tv in task_vectors)
        # Majority vote: positive if sum > 0, negative if sum < 0
        elected_signs[k] = sign_sum.sign()

    return elected_signs


def ties_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    elected_signs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """TIES Step 3: Merge only parameters that agree with elected sign."""
    merged = {}

    for k in task_vectors[0]:
        # For each parameter, only include values that agree with elected sign
        agreeing_values = []
        for tv in task_vectors:
            agrees = (tv[k].sign() == elected_signs[k]) | (tv[k] == 0)
            agreeing_values.append(tv[k] * agrees)

        # Average the agreeing values
        stacked = torch.stack(agreeing_values)
        # Count non-zero contributors per element
        nonzero_count = (stacked != 0).float().sum(dim=0).clamp(min=1)
        merged[k] = stacked.sum(dim=0) / nonzero_count

    return merged


def dare_ties_merge(
    base_path: Path,
    checkpoint_paths: list[Path],
    dare_rate: float = 0.9,
    ties_density: float = 0.2,
    scaling_factor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Full DARE-TIES merge pipeline.

    1. Compute task vectors (deltas from base)
    2. DARE: random drop + rescale
    3. TIES: trim by magnitude, elect sign, merge agreeing params
    4. Add merged task vector back to base
    """
    print(f"Loading base model from {base_path}")
    base_sd = load_state_dict(base_path)

    # Step 1: Compute task vectors
    task_vectors = []
    for i, cp_path in enumerate(checkpoint_paths):
        print(f"Loading checkpoint {i + 1}/{len(checkpoint_paths)}: {cp_path.name}")
        ft_sd = load_state_dict(cp_path)
        tv = compute_task_vector(ft_sd, base_sd)
        task_vectors.append(tv)
        del ft_sd

    # Step 2: DARE - drop and rescale
    print(f"Applying DARE (drop_rate={dare_rate})")
    dared_vectors = [
        dare_drop(tv, drop_rate=dare_rate, seed=42 + i)
        for i, tv in enumerate(task_vectors)
    ]
    del task_vectors

    # Step 3: TIES - trim
    print(f"Applying TIES trim (density={ties_density})")
    trimmed = ties_trim(dared_vectors, density=ties_density)
    del dared_vectors

    # Step 4: TIES - elect sign
    print("Electing signs by majority vote")
    elected_signs = ties_elect_sign(trimmed)

    # Step 5: TIES - merge agreeing parameters
    print("Merging agreeing parameters")
    merged_tv = ties_merge(trimmed, elected_signs)
    del trimmed, elected_signs

    # Step 6: Add merged task vector to base
    print(f"Applying merged task vector (scaling={scaling_factor})")
    final_sd = {}
    for k in base_sd:
        if k in merged_tv:
            final_sd[k] = base_sd[k] + scaling_factor * merged_tv[k]
        else:
            final_sd[k] = base_sd[k]

    return final_sd


def evaluate_model(model, tokenizer, test_path: Path, device: str = "cuda"):
    """Evaluate model on test set."""
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score

    test_df = pd.read_csv(test_path)
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    model.eval()
    preds = []
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())

    f1 = f1_score(labels, preds, average="weighted")
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {"f1_weighted": f1, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def main():
    parser = argparse.ArgumentParser(description="DARE-TIES Model Merging")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["phase1_best", "phase3_best", "phase4_best", "phase5_best"],
        help="Checkpoint names to merge",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Use top-K checkpoints by F1")
    parser.add_argument("--dare-rate", type=float, default=0.9, help="DARE drop rate")
    parser.add_argument("--ties-density", type=float, default=0.2, help="TIES trim density")
    parser.add_argument("--scaling", type=float, default=1.0, help="Task vector scaling")
    parser.add_argument("--output", default="models/dare_ties_merged", help="Output path")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate merged model")
    parser.add_argument(
        "--base-model",
        default="beomi/KcELECTRA-base-v2022",
        help="Base model for delta computation",
    )
    args = parser.parse_args()

    checkpoints_dir = PROJECT_ROOT / "models" / "pipeline_12h" / "checkpoints"
    output_path = PROJECT_ROOT / args.output

    # Select checkpoints
    if args.top_k > 0:
        # Auto-select top-K by F1
        scored = []
        for cp_dir in checkpoints_dir.iterdir():
            metrics_file = cp_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                if "f1" in metrics:
                    scored.append((cp_dir, metrics["f1"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        checkpoint_paths = [s[0] for s in scored[: args.top_k]]
        print(f"Top-{args.top_k} checkpoints by F1:")
        for cp, f1 in scored[: args.top_k]:
            print(f"  {cp.name}: F1={f1:.4f}")
    else:
        checkpoint_paths = [checkpoints_dir / name for name in args.checkpoints]

    # Verify all exist
    for cp in checkpoint_paths:
        if not cp.exists():
            print(f"ERROR: Checkpoint not found: {cp}")
            sys.exit(1)

    # Load base model tokenizer (for saving)
    print(f"Base model: {args.base_model}")

    # Download/cache base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2
    )
    base_path_local = PROJECT_ROOT / "models" / "_base_cache"
    base_model.save_pretrained(str(base_path_local))
    del base_model

    # Run DARE-TIES merge
    merged_sd = dare_ties_merge(
        base_path=base_path_local,
        checkpoint_paths=checkpoint_paths,
        dare_rate=args.dare_rate,
        ties_density=args.ties_density,
        scaling_factor=args.scaling,
    )

    # Save merged model
    output_path.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_paths[0])  # Use first checkpoint for config
    )
    model.load_state_dict(merged_sd)
    model.save_pretrained(str(output_path))

    # Copy tokenizer from first checkpoint
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_paths[0]))
    tokenizer.save_pretrained(str(output_path))

    print(f"\nMerged model saved to: {output_path}")

    # Evaluate
    if args.evaluate:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        test_path = PROJECT_ROOT / "data" / "korean" / "korean_standard_v1_test.csv"

        print(f"\nEvaluating on {test_path}")
        metrics = evaluate_model(model, tokenizer, test_path, device)
        print(f"F1: {metrics['f1_weighted']:.4f}")
        print(f"FP: {metrics['fp']}, FN: {metrics['fn']}")
        print(f"TP: {metrics['tp']}, TN: {metrics['tn']}")

        # Save metrics
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Cleanup base cache
    import shutil
    shutil.rmtree(str(base_path_local), ignore_errors=True)


if __name__ == "__main__":
    main()
