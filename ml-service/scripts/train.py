#!/usr/bin/env python3
"""QLoRA training script for toxic content classification.

Usage:
    python scripts/train.py --data data/train.csv
    python scripts/train.py --data data/train.csv --model distilbert-base-uncased
    python scripts/train.py --data data/train.csv --epochs 5 --batch-size 8
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_service.training import DataConfig, DataProcessor, QLoRATrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for toxic content classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (CSV, JSON, or Parquet)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in data",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of label column in data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (for debugging)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the training run",
    )

    # MLflow arguments
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="evoguard-toxic-classifier",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    # Memory optimization
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("QLoRA Training Script")
    logger.info("=" * 60)

    # Validate data file
    if not args.data.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    # Create configurations
    from ml_service.training.config import LoRAConfig

    training_config = TrainingConfig(
        model_name=args.model,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_4bit_quantization=not args.no_4bit,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment_name=args.experiment_name,
        seed=args.seed,
        lora=LoRAConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
        ),
    )

    data_config = DataConfig(
        text_column=args.text_column,
        label_column=args.label_column,
        max_samples=args.max_samples,
    )

    logger.info(f"Model: {training_config.model_name}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Epochs: {training_config.num_epochs}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"4-bit quantization: {training_config.use_4bit_quantization}")
    logger.info(f"LoRA rank: {training_config.lora.r}")

    # Prepare data
    logger.info("Preparing data...")
    data_processor = DataProcessor(training_config, data_config)
    datasets = data_processor.prepare_from_file(args.data)

    logger.info(f"Train samples: {len(datasets['train'])}")
    logger.info(f"Validation samples: {len(datasets['validation'])}")
    logger.info(f"Test samples: {len(datasets['test'])}")

    # Train
    logger.info("Starting training...")
    trainer = QLoRATrainer(training_config)
    results = trainer.train(
        datasets=datasets,
        run_name=args.run_name,
        use_mlflow=not args.no_mlflow,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(datasets["test"])

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Train loss: {results['train_loss']:.4f}")
    logger.info(f"Eval metrics: {results['eval_metrics']}")
    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()
