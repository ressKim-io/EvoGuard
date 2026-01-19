#!/usr/bin/env python3
"""QLoRA training script for toxic content classification.

Usage:
    # From local file
    python scripts/train.py --data data/train.csv

    # From HuggingFace dataset
    python scripts/train.py --dataset jigsaw --max-samples 10000

    # Quick test with sample data
    python scripts/train.py --sample --max-samples 100

    # Full training
    python scripts/train.py --dataset jigsaw --epochs 3 --batch-size 4
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_service.training import (
    DataConfig,
    DataProcessor,
    JigsawDatasetLoader,
    QLoRATrainer,
    TrainingConfig,
    get_sample_data,
)

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

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data",
        type=Path,
        help="Path to training data (CSV, JSON, or Parquet)",
    )
    data_group.add_argument(
        "--dataset",
        type=str,
        choices=["jigsaw", "jigsaw_balanced", "toxic_tweets"],
        help="HuggingFace dataset to use (jigsaw: Arsive/toxicity_classification_jigsaw)",
    )
    data_group.add_argument(
        "--sample",
        action="store_true",
        help="Use synthetic sample data for testing",
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

    logger.info(f"Model: {training_config.model_name}")
    logger.info(f"Epochs: {training_config.num_epochs}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"4-bit quantization: {training_config.use_4bit_quantization}")
    logger.info(f"LoRA rank: {training_config.lora.r}")

    # Prepare data based on source
    logger.info("Preparing data...")

    if args.sample:
        # Use synthetic sample data
        logger.info("Using synthetic sample data for testing")
        from transformers import AutoTokenizer

        datasets = get_sample_data(n_samples=args.max_samples or 100, seed=args.seed)

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=training_config.max_length,
            )

        datasets = datasets.map(tokenize_fn, batched=True, remove_columns=["text"])

    elif args.dataset:
        # Load from HuggingFace
        logger.info(f"Loading HuggingFace dataset: {args.dataset}")
        loader = JigsawDatasetLoader()
        datasets = loader.load_and_tokenize(
            dataset_name=args.dataset,
            tokenizer_name=training_config.model_name,
            max_length=training_config.max_length,
            max_samples=args.max_samples,
            seed=args.seed,
        )

    else:
        # Load from local file
        if not args.data.exists():
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)

        logger.info(f"Loading from file: {args.data}")
        data_config = DataConfig(
            text_column=args.text_column,
            label_column=args.label_column,
            max_samples=args.max_samples,
        )
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

    # Log experiment
    from ml_service.training.experiment_logger import log_training_run

    experiment_name = args.run_name or f"qlora-{training_config.model_name.split('/')[-1]}"
    dataset_name = args.dataset or (str(args.data) if args.data else "sample")

    experiment = log_training_run(
        name=experiment_name,
        config={
            "model_name": training_config.model_name,
            "max_length": training_config.max_length,
            "num_epochs": training_config.num_epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "use_4bit_quantization": training_config.use_4bit_quantization,
            "lora_r": training_config.lora.r,
            "lora_alpha": training_config.lora.lora_alpha,
            "max_samples": args.max_samples,
        },
        train_metrics={"train_loss": results["train_loss"]},
        eval_metrics=results["eval_metrics"],
        test_metrics=test_metrics,
        dataset=dataset_name,
        model_path=str(results["model_path"]),
        experiments_dir=Path(__file__).parent.parent / "experiments",
    )

    logger.info(f"Experiment logged: #{experiment['id']}")
    logger.info(f"Report saved to: experiments/reports/LATEST_REPORT.md")


if __name__ == "__main__":
    main()
