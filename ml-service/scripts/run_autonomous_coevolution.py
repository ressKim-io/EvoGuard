#!/usr/bin/env python3
"""Autonomous Co-Evolution: Attacker and Defender learn from each other.

This script runs a GAN-style training loop where:
1. Attacker attacks the defender
2. If attacker wins (high evasion) → Defender retrains on failed samples
3. If defender wins (low evasion) → Attacker evolves new strategies via LLM
4. Repeat until time limit

Usage:
    # Run for 4 hours
    python scripts/run_autonomous_coevolution.py --hours 4

    # Run for 30 minutes with verbose output
    python scripts/run_autonomous_coevolution.py --minutes 30 --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CoevolutionConfig:
    """Configuration for co-evolution."""

    # Thresholds
    defender_retrain_threshold: float = 0.3  # Retrain if evasion > 30%
    attacker_evolve_threshold: float = 0.1   # Evolve if evasion < 10%

    # Training
    min_samples_for_retrain: int = 30
    retrain_epochs: int = 2
    merge_with_original: bool = True  # Keep balanced data

    # Attack
    attack_batch_size: int = 20
    attack_variants: int = 10

    # Timing
    cycle_interval_minutes: int = 5  # Wait between cycles

    # Paths
    model_path: Path = field(default_factory=lambda: Path("models/coevolution-model"))
    original_model_path: Path = field(default_factory=lambda: Path("models/toxic-classifier"))
    original_data_path: Path = field(default_factory=lambda: Path("data/balanced_corpus.csv"))


@dataclass
class CycleResult:
    """Result of one co-evolution cycle."""

    cycle_num: int
    timestamp: datetime
    evasion_rate: float
    action: str  # "retrain_defender", "evolve_attacker", "balanced"
    details: dict[str, Any] = field(default_factory=dict)


class AutonomousCoevolution:
    """Runs autonomous co-evolution between attacker and defender."""

    def __init__(self, config: CoevolutionConfig) -> None:
        self.config = config
        self.history: list[CycleResult] = []
        self._running = True
        self._classifier = None
        self._attacker = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("\nReceived stop signal. Finishing current cycle...")
        self._running = False

    def _load_classifier(self, model_path: Path | None = None):
        """Load or reload the classifier."""
        from ml_service.pipeline.classifier_adapter import TrainedClassifierAdapter

        path = model_path or self.config.model_path
        if not path.exists():
            path = self.config.original_model_path

        logger.info(f"Loading classifier from {path}")
        self._classifier = TrainedClassifierAdapter(path)
        self._classifier.load()
        return self._classifier

    def _get_attacker(self):
        """Get or create the evolving attacker."""
        if self._attacker is None:
            from ml_service.pipeline.evolving_attacker import create_evolving_attacker

            self._attacker = create_evolving_attacker(
                classifier=self._classifier,
                include_builtin=True,
            )
        return self._attacker

    async def run_attack_phase(self) -> tuple[float, list]:
        """Run attacks and return evasion rate and failed samples."""
        from ml_service.pipeline.attack_runner import AttackRunner
        from ml_service.pipeline.config import AttackConfig

        config = AttackConfig(
            batch_size=self.config.attack_batch_size,
            num_variants=self.config.attack_variants,
        )

        runner = AttackRunner(config, self._classifier)
        result = runner.run_batch()

        # Also run evolved attacks
        attacker = self._get_attacker()
        evolved_evasions = 0
        evolved_total = 0

        # Test evolved strategies on sample texts
        sample_texts = [r.original_text for r in result.attack_results[:5]]
        for text in sample_texts:
            evolved_results = attacker.apply_evolved_strategies(text, num_variants=3)
            if evolved_results:
                evolved_texts = [r["evasion"] for r in evolved_results]
                preds = self._classifier.predict(evolved_texts)
                evolved_evasions += sum(1 for p in preds if p["label"] == 0)
                evolved_total += len(preds)

        # Combined evasion rate
        total = result.total_variants + evolved_total
        evasions = result.total_evasions + evolved_evasions
        combined_rate = evasions / total if total > 0 else 0

        # Get failed samples
        failed_samples = result.get_failed_samples()

        logger.info(
            f"Attack phase: {result.total_evasions}/{result.total_variants} standard, "
            f"{evolved_evasions}/{evolved_total} evolved, "
            f"combined: {combined_rate:.1%}"
        )

        return combined_rate, failed_samples

    async def retrain_defender(self, failed_samples: list, cycle_num: int) -> dict:
        """Retrain the defender with failed samples."""
        from ml_service.pipeline.sample_collector import FailedSample
        from ml_service.pipeline.data_augmentor import TrainingDataAugmentor
        from ml_service.pipeline.config import RetrainConfig, StorageConfig
        from ml_service.training.trainer import QLoRATrainer
        from ml_service.training.config import TrainingConfig
        import pandas as pd

        logger.info(f"Retraining defender with {len(failed_samples)} failed samples...")

        # Convert to FailedSample objects
        samples = [
            FailedSample(
                original_text=s.original_text,
                variant_text=s.variant_text,
                original_label=s.original_label,
                predicted_label=s.model_prediction,
                strategy_name=s.strategy_name,
                confidence=s.model_confidence,
            )
            for s in failed_samples
        ]

        # Augment data
        retrain_config = RetrainConfig(
            augmentation_multiplier=3,
            merge_with_original=self.config.merge_with_original,
            original_data_path=self.config.original_data_path if self.config.merge_with_original else None,
        )
        storage_config = StorageConfig()
        augmentor = TrainingDataAugmentor(retrain_config, storage_config)

        # Load original data for balance (MUST include non-toxic samples)
        original_data = None
        if self.config.merge_with_original and self.config.original_data_path.exists():
            original_data = pd.read_csv(self.config.original_data_path)
            # Ensure we have both classes
            non_toxic = original_data[original_data['label'] == 0]
            toxic = original_data[original_data['label'] == 1]
            logger.info(f"Original data: {len(toxic)} toxic, {len(non_toxic)} non-toxic")
            # Need at least some non-toxic for balanced training
            if len(non_toxic) < 5:
                logger.warning("Not enough non-toxic samples in original data!")
            logger.info(f"Merging with {len(original_data)} original samples for balance")

        augmented = augmentor.augment(samples, original_data)
        augmentor.save(augmented, f"cycle_{cycle_num}")

        logger.info(f"Augmented dataset: {augmented.total_count} samples")

        # Prepare for training
        tokenized = augmentor.prepare_for_training(augmented)

        # Train
        output_dir = self.config.model_path
        training_config = TrainingConfig(
            model_name="bert-base-uncased",
            output_dir=output_dir,
            num_epochs=self.config.retrain_epochs,
            batch_size=8,
            eval_batch_size=16,
        )

        trainer = QLoRATrainer(training_config)
        trainer.setup_model()

        # Train without MLflow to avoid connection issues
        result = trainer.train(tokenized, use_mlflow=False)

        logger.info(f"Retraining complete. Metrics: {result['eval_metrics']}")

        # Reload the classifier with new model
        self._load_classifier(output_dir)

        return {
            "samples_used": len(samples),
            "augmented_total": augmented.total_count,
            "eval_metrics": result["eval_metrics"],
        }

    async def evolve_attacker(self, blocked_strategies: list) -> dict:
        """Evolve new attack strategies using LLM."""
        logger.info(f"Evolving attacker. Blocked strategies: {blocked_strategies[:5]}...")

        attacker = self._get_attacker()

        # Sample texts for testing new strategies
        sample_texts = [
            "You are an idiot and I hate you",
            "Go kill yourself you worthless person",
            "I hope you die in a fire",
            "Everyone hates you, disappear",
            "You're disgusting and pathetic",
        ]

        result = await attacker.evolve(
            blocked_strategies=blocked_strategies,
            sample_texts=sample_texts,
            min_success_rate=0.2,
        )

        logger.info(
            f"Evolution complete. "
            f"New strategies: {len(result.new_strategies)}, "
            f"Successful: {result.successful_count}"
        )

        return {
            "strategies_tested": result.tested_count,
            "strategies_successful": result.successful_count,
            "new_strategy_names": [s.name for s in result.new_strategies],
        }

    async def run_cycle(self, cycle_num: int) -> CycleResult:
        """Run one co-evolution cycle."""
        logger.info(f"\n{'='*60}")
        logger.info(f"CO-EVOLUTION CYCLE {cycle_num}")
        logger.info(f"{'='*60}")

        timestamp = datetime.now(UTC)

        # Phase 1: Attack
        logger.info("\n[Phase 1] Running attacks...")
        evasion_rate, failed_samples = await self.run_attack_phase()

        # Phase 2: Decide action
        if evasion_rate > self.config.defender_retrain_threshold:
            # Attacker winning - retrain defender
            action = "retrain_defender"
            logger.info(f"\n[Phase 2] Evasion {evasion_rate:.1%} > {self.config.defender_retrain_threshold:.1%}")
            logger.info("→ DEFENDER needs retraining!")

            if len(failed_samples) >= self.config.min_samples_for_retrain:
                details = await self.retrain_defender(failed_samples, cycle_num)
            else:
                logger.info(f"Not enough samples ({len(failed_samples)} < {self.config.min_samples_for_retrain})")
                details = {"skipped": True, "reason": "not_enough_samples"}

        elif evasion_rate < self.config.attacker_evolve_threshold:
            # Defender winning - evolve attacker
            action = "evolve_attacker"
            logger.info(f"\n[Phase 2] Evasion {evasion_rate:.1%} < {self.config.attacker_evolve_threshold:.1%}")
            logger.info("→ ATTACKER needs evolution!")

            # Get blocked strategies
            blocked = list(set(s.strategy_name for s in failed_samples))
            if not blocked:
                blocked = ["all_strategies_blocked"]

            details = await self.evolve_attacker(blocked)

        else:
            # Balanced
            action = "balanced"
            logger.info(f"\n[Phase 2] Evasion {evasion_rate:.1%} is balanced. No action needed.")
            details = {"status": "equilibrium"}

        result = CycleResult(
            cycle_num=cycle_num,
            timestamp=timestamp,
            evasion_rate=evasion_rate,
            action=action,
            details=details,
        )

        self.history.append(result)
        self._save_history()

        return result

    def _save_history(self):
        """Save history to file."""
        history_path = Path("data/coevolution_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "cycle_num": r.cycle_num,
                "timestamp": r.timestamp.isoformat(),
                "evasion_rate": r.evasion_rate,
                "action": r.action,
                "details": r.details,
            }
            for r in self.history
        ]

        with open(history_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def run(self, hours: float = 0, minutes: float = 0):
        """Run co-evolution for specified duration."""
        total_minutes = hours * 60 + minutes
        end_time = datetime.now(UTC) + timedelta(minutes=total_minutes)

        logger.info(f"Starting autonomous co-evolution")
        logger.info(f"Duration: {total_minutes:.0f} minutes (until {end_time.strftime('%H:%M:%S')})")
        logger.info(f"Defender retrain threshold: >{self.config.defender_retrain_threshold:.0%} evasion")
        logger.info(f"Attacker evolve threshold: <{self.config.attacker_evolve_threshold:.0%} evasion")

        # Load initial model
        self._load_classifier()

        cycle_num = 0
        while self._running and datetime.now(UTC) < end_time:
            cycle_num += 1

            try:
                result = await self.run_cycle(cycle_num)

                # Print summary
                remaining = (end_time - datetime.now(UTC)).total_seconds() / 60
                logger.info(f"\nCycle {cycle_num} complete: {result.action}")
                logger.info(f"Time remaining: {remaining:.0f} minutes")

            except Exception as e:
                logger.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)

            # Wait before next cycle
            if self._running and datetime.now(UTC) < end_time:
                wait_time = min(
                    self.config.cycle_interval_minutes * 60,
                    (end_time - datetime.now(UTC)).total_seconds(),
                )
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time/60:.1f} minutes before next cycle...")
                    await asyncio.sleep(wait_time)

        # Final summary
        self._print_summary()

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("CO-EVOLUTION COMPLETE")
        print("=" * 60)

        if not self.history:
            print("No cycles completed")
            return

        print(f"\nTotal cycles: {len(self.history)}")

        # Count actions
        actions = {}
        evasion_rates = []
        for r in self.history:
            actions[r.action] = actions.get(r.action, 0) + 1
            evasion_rates.append(r.evasion_rate)

        print(f"\nActions taken:")
        for action, count in actions.items():
            print(f"  {action}: {count}")

        print(f"\nEvasion rate:")
        print(f"  Start: {evasion_rates[0]:.1%}")
        print(f"  End: {evasion_rates[-1]:.1%}")
        print(f"  Min: {min(evasion_rates):.1%}")
        print(f"  Max: {max(evasion_rates):.1%}")

        # Show progression
        print(f"\nProgression:")
        print(f"{'Cycle':<8} {'Evasion':<12} {'Action':<20}")
        print("-" * 40)
        for r in self.history[-10:]:  # Last 10 cycles
            print(f"{r.cycle_num:<8} {r.evasion_rate:.1%:<12} {r.action:<20}")

        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Autonomous Co-Evolution Training")
    parser.add_argument("--hours", type=float, default=0, help="Hours to run")
    parser.add_argument("--minutes", type=float, default=30, help="Minutes to run")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between cycles")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = CoevolutionConfig(
        cycle_interval_minutes=args.interval,
    )

    coevolution = AutonomousCoevolution(config)
    await coevolution.run(hours=args.hours, minutes=args.minutes)


if __name__ == "__main__":
    asyncio.run(main())
