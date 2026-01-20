#!/usr/bin/env python3
"""Korean Autonomous Co-Evolution Training.

한국어 Adversarial Co-Evolution:
1. Attacker가 한국어 공격 전략으로 Defender 공격
2. Evasion rate > 30% → Defender 재학습
3. Evasion rate < 10% → Attacker 진화 (LLM)
4. 지정된 시간 동안 반복

Usage:
    # 4시간 동안 실행
    python scripts/run_korean_coevolution.py --hours 4

    # 30분 동안 실행
    python scripts/run_korean_coevolution.py --minutes 30 --verbose
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
class KoreanCoevolutionConfig:
    """Configuration for Korean co-evolution."""

    # 임계값
    defender_retrain_threshold: float = 0.3  # Evasion > 30% → 재학습
    attacker_evolve_threshold: float = 0.1   # Evasion < 10% → 진화

    # 학습
    min_samples_for_retrain: int = 30
    retrain_epochs: int = 2
    merge_with_original: bool = True

    # 공격 - 강화된 설정
    attack_batch_size: int = 100  # 30 → 100 (더 많은 샘플 공격)
    attack_variants: int = 15     # 10 → 15 (더 많은 변형 생성)

    # 타이밍 - 연속 모드 (대기 없음)
    cycle_interval_minutes: int = 0

    # 모델
    model_name: str = "beomi/KcELECTRA-base-v2022"

    # 경로
    model_path: Path = field(default_factory=lambda: Path("models/korean-coevolution-model"))
    original_model_path: Path = field(default_factory=lambda: Path("models/korean-toxic-classifier"))
    original_data_path: Path = field(default_factory=lambda: Path("data/korean/korean_hate_speech_balanced.csv"))


@dataclass
class CycleResult:
    """Result of one co-evolution cycle."""

    cycle_num: int
    timestamp: datetime
    evasion_rate: float
    action: str  # "retrain_defender", "evolve_attacker", "balanced"
    details: dict[str, Any] = field(default_factory=dict)


class KoreanCoevolution:
    """Korean autonomous co-evolution trainer."""

    def __init__(self, config: KoreanCoevolutionConfig) -> None:
        self.config = config
        self.history: list[CycleResult] = []
        self._running = True
        self._classifier = None
        self._attacker = None

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("\nReceived stop signal. Finishing current cycle...")
        self._running = False

    def _load_classifier(self, model_path: Path | None = None):
        """Load or reload the Korean classifier."""
        from ml_service.pipeline.classifier_adapter import TrainedClassifierAdapter

        path = model_path or self.config.model_path
        if not path.exists():
            # 최초 실행시 원본 모델 또는 새로 학습
            path = self.config.original_model_path
            if not path.exists():
                logger.info("No pre-trained model found. Training initial model...")
                self._train_initial_model()
                path = self.config.model_path

        logger.info(f"Loading classifier from {path}")
        self._classifier = TrainedClassifierAdapter(path)
        self._classifier.load()
        return self._classifier

    def _train_initial_model(self):
        """Train initial Korean model."""
        from ml_service.training.trainer import QLoRATrainer
        from ml_service.training.config import TrainingConfig
        from ml_service.training.data import DataProcessor

        logger.info("Training initial Korean toxic classifier...")

        # Load Korean data
        data_path = self.config.original_data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Korean data not found: {data_path}")

        # Training config - disable quantization for ELECTRA models
        training_config = TrainingConfig(
            model_name=self.config.model_name,
            output_dir=self.config.model_path,
            num_epochs=3,
            batch_size=16,
            eval_batch_size=32,
            max_length=128,
            use_4bit_quantization=False,  # ELECTRA doesn't support 4-bit quantization well
        )

        # Prepare dataset using DataProcessor
        processor = DataProcessor(training_config)
        tokenized = processor.prepare_from_file(data_path)
        logger.info(f"Prepared dataset from {data_path}")

        # Train
        trainer = QLoRATrainer(training_config)
        trainer.setup_model()
        result = trainer.train(tokenized, use_mlflow=False)

        logger.info(f"Initial training complete. F1: {result['eval_metrics'].get('eval_f1', 'N/A')}")

    async def run_attack_phase(self) -> tuple[float, list]:
        """Run Korean attacks and return evasion rate and failed samples."""
        from ml_service.pipeline.korean_attack_runner import KoreanAttackRunner
        from ml_service.pipeline.korean_config import KoreanAttackConfig

        config = KoreanAttackConfig(
            batch_size=self.config.attack_batch_size,
            num_variants=self.config.attack_variants,
        )

        runner = KoreanAttackRunner(config, self._classifier)
        result = runner.run_batch()

        logger.info(
            f"Attack phase: {result.total_evasions}/{result.total_attacks} "
            f"evasions ({result.evasion_rate:.1%})"
        )

        return result.evasion_rate, result.get_failed_samples()

    async def retrain_defender(self, failed_samples: list, cycle_num: int) -> dict:
        """Retrain the Korean defender with failed samples."""
        from ml_service.pipeline.sample_collector import FailedSample
        from ml_service.pipeline.data_augmentor import TrainingDataAugmentor
        from ml_service.pipeline.config import RetrainConfig, StorageConfig
        from ml_service.training.trainer import QLoRATrainer
        from ml_service.training.config import TrainingConfig
        import pandas as pd

        logger.info(f"Retraining Korean defender with {len(failed_samples)} failed samples...")

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
        storage_config = StorageConfig(
            augmented_data_dir=Path("data/korean/augmented"),
        )
        augmentor = TrainingDataAugmentor(retrain_config, storage_config)

        # Load original data for balance
        original_data = None
        if self.config.merge_with_original and self.config.original_data_path.exists():
            original_data = pd.read_csv(self.config.original_data_path)
            # Sample for balance
            non_toxic = original_data[original_data['label'] == 0].sample(
                n=min(1000, len(original_data[original_data['label'] == 0])),
                random_state=42
            )
            toxic = original_data[original_data['label'] == 1].sample(
                n=min(1000, len(original_data[original_data['label'] == 1])),
                random_state=42
            )
            original_data = pd.concat([toxic, non_toxic])
            logger.info(f"Merging with {len(original_data)} original samples")

        augmented = augmentor.augment(samples, original_data)
        augmentor.save(augmented, f"korean_cycle_{cycle_num}")

        logger.info(f"Augmented dataset: {augmented.total_count} samples")

        # Prepare for training
        tokenized = augmentor.prepare_for_training(augmented, tokenizer_name=self.config.model_name)

        # Train - disable quantization for ELECTRA models
        training_config = TrainingConfig(
            model_name=self.config.model_name,
            output_dir=self.config.model_path,
            num_epochs=self.config.retrain_epochs,
            batch_size=16,
            eval_batch_size=32,
            use_4bit_quantization=False,
        )

        trainer = QLoRATrainer(training_config)
        trainer.setup_model()
        result = trainer.train(tokenized, use_mlflow=False)

        logger.info(f"Retraining complete. Metrics: {result['eval_metrics']}")

        # Reload classifier
        self._load_classifier(self.config.model_path)

        return {
            "samples_used": len(samples),
            "augmented_total": augmented.total_count,
            "eval_metrics": result["eval_metrics"],
        }

    async def evolve_attacker(self, blocked_strategies: list) -> dict:
        """Evolve new Korean attack strategies using LLM."""
        logger.info(f"Evolving Korean attacker. Blocked: {blocked_strategies[:5]}...")

        # For now, just log - LLM evolution can be added later
        # The Korean attack strategies are already comprehensive

        return {
            "status": "skipped",
            "reason": "Korean strategies are comprehensive, LLM evolution not implemented yet",
            "blocked_strategies": blocked_strategies,
        }

    async def run_cycle(self, cycle_num: int) -> CycleResult:
        """Run one co-evolution cycle."""
        logger.info(f"\n{'='*60}")
        logger.info(f"KOREAN CO-EVOLUTION CYCLE {cycle_num}")
        logger.info(f"{'='*60}")

        timestamp = datetime.now(UTC)

        # Phase 1: Attack
        logger.info("\n[Phase 1] Running Korean attacks...")
        evasion_rate, failed_samples = await self.run_attack_phase()

        # Phase 2: Decide action
        if evasion_rate > self.config.defender_retrain_threshold:
            action = "retrain_defender"
            logger.info(f"\n[Phase 2] Evasion {evasion_rate:.1%} > {self.config.defender_retrain_threshold:.1%}")
            logger.info("→ DEFENDER needs retraining!")

            if len(failed_samples) >= self.config.min_samples_for_retrain:
                details = await self.retrain_defender(failed_samples, cycle_num)
            else:
                logger.info(f"Not enough samples ({len(failed_samples)} < {self.config.min_samples_for_retrain})")
                details = {"skipped": True, "reason": "not_enough_samples"}

        elif evasion_rate < self.config.attacker_evolve_threshold:
            action = "evolve_attacker"
            logger.info(f"\n[Phase 2] Evasion {evasion_rate:.1%} < {self.config.attacker_evolve_threshold:.1%}")
            logger.info("→ ATTACKER needs evolution!")

            blocked = list(set(s.strategy_name for s in failed_samples)) or ["all_blocked"]
            details = await self.evolve_attacker(blocked)

        else:
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
        history_path = Path("data/korean/coevolution_history.json")
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
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    async def run(self, hours: float = 0, minutes: float = 0):
        """Run co-evolution for specified duration."""
        total_minutes = hours * 60 + minutes
        end_time = datetime.now(UTC) + timedelta(minutes=total_minutes)

        logger.info("Starting Korean autonomous co-evolution")
        logger.info(f"Duration: {total_minutes:.0f} minutes (until {end_time.strftime('%H:%M:%S')})")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Defender retrain threshold: >{self.config.defender_retrain_threshold:.0%} evasion")
        logger.info(f"Attacker evolve threshold: <{self.config.attacker_evolve_threshold:.0%} evasion")

        # Load initial model
        self._load_classifier()

        cycle_num = 0
        while self._running and datetime.now(UTC) < end_time:
            cycle_num += 1

            try:
                result = await self.run_cycle(cycle_num)

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

        self._print_summary()

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("KOREAN CO-EVOLUTION COMPLETE")
        print("=" * 60)

        if not self.history:
            print("No cycles completed")
            return

        print(f"\nTotal cycles: {len(self.history)}")

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

        print(f"\nProgression:")
        print(f"{'Cycle':<8} {'Evasion':<12} {'Action':<20}")
        print("-" * 40)
        for r in self.history[-10:]:
            evasion_str = f"{r.evasion_rate:.1%}"
            print(f"{r.cycle_num:<8} {evasion_str:<12} {r.action:<20}")

        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Korean Autonomous Co-Evolution Training")
    parser.add_argument("--hours", type=float, default=0, help="Hours to run")
    parser.add_argument("--minutes", type=float, default=30, help="Minutes to run")
    parser.add_argument("--interval", type=int, default=10, help="Minutes between cycles")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--model", type=str, default="beomi/KcELECTRA-base-v2022",
                        help="Korean model to use")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = KoreanCoevolutionConfig(
        cycle_interval_minutes=args.interval,
        model_name=args.model,
    )

    coevolution = KoreanCoevolution(config)
    await coevolution.run(hours=args.hours, minutes=args.minutes)


if __name__ == "__main__":
    asyncio.run(main())
