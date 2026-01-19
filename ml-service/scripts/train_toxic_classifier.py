#!/usr/bin/env python3
"""
Long-running QLoRA training script for toxic comment classification.

Optimized for RTX 4060 Ti (8GB VRAM) with 4+ hours training time.

Usage:
    # Quick test (10 minutes)
    python scripts/train_toxic_classifier.py --mode quick

    # Standard training (2-3 hours)
    python scripts/train_toxic_classifier.py --mode standard

    # Full training (4-6 hours)
    python scripts/train_toxic_classifier.py --mode full

    # Extended training with cross-validation (8-12 hours)
    python scripts/train_toxic_classifier.py --mode extended

    # Resume from checkpoint
    python scripts/train_toxic_classifier.py --resume checkpoints/checkpoint-1000

    # Custom settings
    python scripts/train_toxic_classifier.py --epochs 15 --samples 100000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"training_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)


# Training presets for different durations
TRAINING_PRESETS = {
    "quick": {
        "description": "Quick test run (~10 minutes)",
        "max_samples": 1000,
        "num_epochs": 1,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "lora_r": 8,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
    },
    "standard": {
        "description": "Standard training (2-3 hours)",
        "max_samples": 50000,
        "num_epochs": 5,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "lora_r": 16,
        "eval_strategy": "epoch",
    },
    "full": {
        "description": "Full dataset training (4-6 hours)",
        "max_samples": None,  # All ~159K samples
        "num_epochs": 10,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "lora_r": 64,
        "lora_alpha": 16,
        "eval_strategy": "epoch",
    },
    "extended": {
        "description": "Extended training with more epochs (8-12 hours)",
        "max_samples": None,
        "num_epochs": 20,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "lora_r": 64,
        "lora_alpha": 16,
        "warmup_ratio": 0.05,
        "eval_strategy": "epoch",
    },
}


def check_gpu():
    """Check GPU availability and specs."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be slow on CPU.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    logger.info(f"GPU: {gpu_name}")
    logger.info(f"VRAM: {gpu_memory:.1f} GB")

    if gpu_memory < 6:
        logger.warning("Low VRAM! Consider reducing batch size.")

    return True


def get_training_config(args):
    """Build training config from arguments."""
    from ml_service.training.config import LoRAConfig, TrainingConfig

    # Start with preset if specified
    preset = TRAINING_PRESETS.get(args.mode, TRAINING_PRESETS["full"])
    logger.info(f"Using preset: {args.mode} - {preset['description']}")

    # Build LoRA config
    lora_config = LoRAConfig(
        r=args.lora_r or preset.get("lora_r", 64),
        lora_alpha=args.lora_alpha or preset.get("lora_alpha", 32),
        lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )

    # Build training config
    config = TrainingConfig(
        model_name=args.model,
        num_labels=2,
        max_length=args.max_length,
        learning_rate=args.lr,
        batch_size=args.batch_size or preset.get("batch_size", 8),
        eval_batch_size=16,
        num_epochs=args.epochs or preset.get("num_epochs", 10),
        warmup_ratio=preset.get("warmup_ratio", 0.1),
        weight_decay=0.01,
        gradient_accumulation_steps=args.grad_accum or preset.get("gradient_accumulation_steps", 4),
        use_gradient_checkpointing=True,
        use_4bit_quantization=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        lora=lora_config,
        output_dir=Path(args.output_dir),
        logging_dir=Path(args.output_dir) / "logs",
        eval_strategy=preset.get("eval_strategy", "epoch"),
        save_strategy=preset.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        mlflow_experiment_name=args.experiment_name,
        mlflow_tracking_uri=args.mlflow_uri,
        early_stopping_patience=args.early_stopping,
        seed=args.seed,
    )

    return config, preset


def load_dataset(args, preset):
    """Load and prepare dataset."""
    from ml_service.training.datasets import JigsawDatasetLoader

    loader = JigsawDatasetLoader()

    max_samples = args.samples or preset.get("max_samples")
    logger.info(f"Loading dataset: {args.dataset}")
    if max_samples:
        logger.info(f"Limiting to {max_samples} samples")

    datasets = loader.load_and_tokenize(
        dataset_name=args.dataset,
        tokenizer_name=args.model,
        max_length=args.max_length,
        max_samples=max_samples,
        seed=args.seed,
    )

    logger.info(f"Dataset loaded:")
    for split, ds in datasets.items():
        logger.info(f"  {split}: {len(ds)} samples")

    return datasets


def train(args):
    """Main training function."""
    import torch

    from ml_service.training.trainer import QLoRATrainer

    logger.info("=" * 60)
    logger.info("EvoGuard Toxic Comment Classifier - QLoRA Training")
    logger.info("=" * 60)

    # Check GPU
    has_gpu = check_gpu()

    # Get config
    config, preset = get_training_config(args)

    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  LoRA r: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    logger.info(f"  4-bit quantization: {config.use_4bit_quantization}")
    logger.info(f"  Output: {config.output_dir}")

    # Estimate training time
    samples = args.samples or preset.get("max_samples") or 159000
    steps_per_epoch = samples // (config.batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    # Rough estimate: ~1.5 seconds per step on RTX 4060 Ti
    estimated_hours = (total_steps * 1.5) / 3600

    logger.info(f"\nEstimated Training:")
    logger.info(f"  Steps per epoch: ~{steps_per_epoch}")
    logger.info(f"  Total steps: ~{total_steps}")
    logger.info(f"  Estimated time: ~{estimated_hours:.1f} hours")

    # Load dataset
    datasets = load_dataset(args, preset)

    # Create trainer
    trainer = QLoRATrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume)

    # Setup model
    trainer.setup_model()

    # Train
    run_name = f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    results = trainer.train(
        datasets=datasets,
        run_name=run_name,
        use_mlflow=args.use_mlflow,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Train loss: {results['train_loss']:.4f}")
    logger.info(f"Eval metrics:")
    for k, v in results["eval_metrics"].items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    logger.info(f"Model saved to: {results['model_path']}")

    # Save final metrics to file
    metrics_file = config.output_dir / "final_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Training completed: {datetime.now()}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Epochs: {config.num_epochs}\n")
        f.write(f"Train loss: {results['train_loss']:.4f}\n")
        for k, v in results["eval_metrics"].items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")

    logger.info(f"Metrics saved to: {metrics_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train toxic comment classifier with QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  quick     - Quick test (~10 min): 1K samples, 1 epoch
  standard  - Standard (~2-3 hours): 50K samples, 5 epochs
  full      - Full training (~4-6 hours): 159K samples, 10 epochs
  extended  - Extended (~8-12 hours): 159K samples, 20 epochs

Examples:
  python train_toxic_classifier.py --mode quick
  python train_toxic_classifier.py --mode full --epochs 15
  python train_toxic_classifier.py --resume checkpoints/checkpoint-5000
        """,
    )

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=list(TRAINING_PRESETS.keys()),
        help="Training preset (default: full)",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Base model from HuggingFace (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )

    # Training hyperparameters (override preset)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides preset)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides preset)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (overrides preset)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )

    # LoRA settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (overrides preset)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (overrides preset)",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="jigsaw",
        choices=["jigsaw", "jigsaw_balanced"],
        help="Dataset to use (default: jigsaw)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum samples to use (overrides preset)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/toxic-classifier",
        help="Output directory for model and logs",
    )

    # MLflow
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="evoguard-toxic-classifier",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI (default: local ./mlruns)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Disable MLflow tracking",
    )

    # Training control
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=3,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    try:
        train(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
