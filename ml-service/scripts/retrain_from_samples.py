#!/usr/bin/env python3
"""Retrain model from failed samples.

Usage:
    python scripts/retrain_from_samples.py --samples-file data/failed_samples/failed_samples_real_test_001.json
    python scripts/retrain_from_samples.py --cycle-id real_test_001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from ml_service.pipeline.config import PipelineConfig, RetrainConfig, StorageConfig
from ml_service.pipeline.data_augmentor import TrainingDataAugmentor
from ml_service.pipeline.sample_collector import FailedSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_failed_samples(filepath: Path) -> list[FailedSample]:
    """Load failed samples from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    samples = []
    for s in data["samples"]:
        samples.append(FailedSample(
            original_text=s["original_text"],
            variant_text=s["variant_text"],
            original_label=s["original_label"],
            predicted_label=s["predicted_label"],
            strategy_name=s["strategy_name"],
            confidence=s["confidence"],
        ))

    logger.info(f"Loaded {len(samples)} failed samples from {filepath}")
    return samples


def retrain_model(
    samples: list[FailedSample],
    output_dir: Path,
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
) -> dict:
    """Retrain model with failed samples."""
    from ml_service.training.trainer import QLoRATrainer
    from ml_service.training.config import TrainingConfig

    # Configure augmentor
    retrain_config = RetrainConfig(
        augmentation_multiplier=3,
        merge_with_original=False,  # Only use adversarial samples
        max_augmented_samples=5000,
    )
    storage_config = StorageConfig()

    augmentor = TrainingDataAugmentor(retrain_config, storage_config)

    # Augment data
    logger.info("Augmenting data...")
    augmented = augmentor.augment(samples)

    stats = augmentor.get_statistics(augmented)
    logger.info(f"Augmented dataset stats: {stats}")

    # Save augmented data
    augmentor.save(augmented, "retrain")

    # Prepare for training
    logger.info("Preparing data for training...")
    tokenized = augmentor.prepare_for_training(augmented, tokenizer_name=model_name)

    logger.info(f"Train samples: {len(tokenized['train'])}")
    logger.info(f"Eval samples: {len(tokenized['test'])}")

    # Configure training with MLflow
    training_config = TrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=8,
        eval_batch_size=16,
        learning_rate=2e-4,
        mlflow_experiment_name="adversarial-retraining",
    )

    # Train
    logger.info("Starting QLoRA training...")
    trainer = QLoRATrainer(training_config)
    trainer.setup_model()

    result = trainer.train(tokenized, run_name="adversarial_retrain")

    logger.info(f"Training complete!")
    logger.info(f"Eval metrics: {result['eval_metrics']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Retrain model from failed samples")
    parser.add_argument(
        "--samples-file",
        type=Path,
        help="Path to failed samples JSON file",
    )
    parser.add_argument(
        "--cycle-id",
        type=str,
        help="Cycle ID to load samples from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/adversarial-retrained"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Base model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    # Determine samples file
    if args.samples_file:
        samples_file = args.samples_file
    elif args.cycle_id:
        samples_file = Path(f"data/failed_samples/failed_samples_{args.cycle_id}.json")
    else:
        # Find most recent
        samples_dir = Path("data/failed_samples")
        files = sorted(samples_dir.glob("failed_samples_*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            logger.error("No failed samples found")
            sys.exit(1)
        samples_file = files[-1]
        logger.info(f"Using most recent samples file: {samples_file}")

    if not samples_file.exists():
        logger.error(f"Samples file not found: {samples_file}")
        sys.exit(1)

    # Load samples
    samples = load_failed_samples(samples_file)

    if len(samples) < 10:
        logger.warning(f"Only {len(samples)} samples - may not be enough for effective training")

    # Retrain
    try:
        result = retrain_model(
            samples,
            output_dir=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
        )

        print("\n" + "=" * 60)
        print("RETRAINING COMPLETE")
        print("=" * 60)
        print(f"Model saved to: {args.output_dir}")
        print(f"Eval Loss: {result['eval_metrics'].get('eval_loss', 'N/A'):.4f}")
        print(f"Eval Accuracy: {result['eval_metrics'].get('eval_accuracy', 'N/A'):.4f}")
        print(f"Eval F1: {result['eval_metrics'].get('eval_f1', 'N/A'):.4f}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
