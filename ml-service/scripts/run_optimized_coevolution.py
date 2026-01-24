#!/usr/bin/env python3
"""Optimized Korean Co-Evolution Training.

GPU 활용 최적화 버전:
1. Phase 2 모델을 베이스로 시작 (최고 성능)
2. 임계값 조정으로 더 자주 재학습
3. 배치 사이즈 증가로 GPU 활용 극대화
4. 더 강력한 공격으로 evasion rate 상승

Usage:
    # 4시간 실행 (권장)
    python scripts/run_optimized_coevolution.py --hours 4

    # 테스트 실행
    python scripts/run_optimized_coevolution.py --minutes 30 --verbose
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

import torch

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
class OptimizedCoevolutionConfig:
    """Optimized configuration for GPU utilization."""

    # 임계값 - 더 자주 재학습하도록 조정
    defender_retrain_threshold: float = 0.08  # 8% 초과시 재학습 (더 공격적)
    attacker_evolve_threshold: float = 0.03   # 3% 미만시 진화

    # 연속 학습 모드 - balanced 상태에서도 샘플 축적 후 학습
    accumulate_samples: bool = True
    accumulate_threshold: int = 100  # 100개 샘플 모이면 재학습
    force_retrain_cycles: int = 5    # N 사이클마다 강제 재학습

    # 학습 - RTX 4060 Ti 8GB 최적화
    min_samples_for_retrain: int = 20
    retrain_epochs: int = 3
    batch_size: int = 16  # 8GB GPU 안정적
    eval_batch_size: int = 32
    merge_with_original: bool = True
    original_sample_size: int = 2000  # 기존 1000 → 2000

    # 공격 - 더 강력하게
    attack_batch_size: int = 200  # 기존 100 → 200
    attack_variants: int = 20     # 기존 15 → 20

    # 타이밍
    cycle_interval_seconds: int = 0  # 연속 실행

    # 모델 - Phase 2 베이스
    model_name: str = "beomi/KcELECTRA-base-v2022"

    # 경로 - Phase 2 모델 베이스
    model_path: Path = field(default_factory=lambda: Path("models/coevolution-optimized"))
    base_model_path: Path = field(default_factory=lambda: Path("models/phase2-combined"))
    original_data_path: Path = field(default_factory=lambda: Path("data/korean/korean_hate_speech_full.csv"))


@dataclass
class CycleResult:
    """Result of one co-evolution cycle."""

    cycle_num: int
    timestamp: datetime
    evasion_rate: float
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    gpu_util: float = 0.0


class OptimizedCoevolution:
    """Optimized co-evolution trainer with GPU utilization."""

    def __init__(self, config: OptimizedCoevolutionConfig) -> None:
        self.config = config
        self.history: list[CycleResult] = []
        self._running = True
        self._classifier = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._start_time = None
        self._retrain_count = 0
        self._total_gpu_time = 0.0
        self._accumulated_samples = []  # 축적된 실패 샘플
        self._cycles_since_retrain = 0  # 마지막 재학습 이후 사이클 수

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(f"Device: {self._device}")
        if self._device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _handle_signal(self, signum, frame):
        logger.info("\nReceived stop signal. Finishing current cycle...")
        self._running = False

    def _load_classifier(self):
        """Load the Phase 2 classifier directly."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F

        model_path = self.config.base_model_path / "best_model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading classifier from {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

        # Create classifier wrapper
        class ClassifierWrapper:
            def __init__(wrapper_self, model, tokenizer, device):
                wrapper_self.model = model
                wrapper_self.tokenizer = tokenizer
                wrapper_self.device = device

            def predict(wrapper_self, texts):
                if isinstance(texts, str):
                    texts = [texts]

                with torch.no_grad():
                    enc = wrapper_self.tokenizer(
                        texts,
                        truncation=True,
                        padding=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(wrapper_self.device) for k, v in enc.items()}
                    out = wrapper_self.model(**enc)
                    probs = F.softmax(out.logits, dim=-1)

                    results = []
                    for i in range(len(texts)):
                        toxic_prob = probs[i, 1].item()
                        results.append({
                            "label": 1 if toxic_prob > 0.5 else 0,
                            "confidence": max(toxic_prob, 1 - toxic_prob),
                            "toxic_prob": toxic_prob,
                        })
                    return results

        self._classifier = ClassifierWrapper(self._model, self._tokenizer, self._device)
        logger.info(f"Classifier loaded on {self._device}")
        return self._classifier

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        if self._device != "cuda":
            return 0.0
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    async def run_attack_phase(self) -> tuple[float, list]:
        """Run attacks with larger batch."""
        from ml_service.pipeline.korean_attack_runner import KoreanAttackRunner
        from ml_service.pipeline.korean_config import KoreanAttackConfig

        config = KoreanAttackConfig(
            batch_size=self.config.attack_batch_size,
            num_variants=self.config.attack_variants,
        )

        runner = KoreanAttackRunner(config, self._classifier)
        result = runner.run_batch()

        logger.info(
            f"Attack: {result.total_evasions}/{result.total_attacks} "
            f"evasions ({result.evasion_rate:.1%})"
        )

        return result.evasion_rate, result.get_failed_samples()

    async def retrain_defender(self, failed_samples: list, cycle_num: int) -> dict:
        """Retrain with optimized GPU settings (direct fine-tuning)."""
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        import torch.nn.functional as F
        import pandas as pd
        import time
        import random

        gpu_start = time.time()
        num_samples = len(failed_samples)
        logger.info(f"Retraining with {num_samples} failed samples (GPU optimized)...")

        # Prepare training data from failed samples + original data
        train_texts = []
        train_labels = []

        # Add failed samples (these are toxic samples that evaded detection)
        for s in failed_samples:
            train_texts.append(s.variant_text)
            train_labels.append(s.original_label)
            # Also add original
            train_texts.append(s.original_text)
            train_labels.append(s.original_label)

        # Add original data for balance
        if self.config.original_data_path.exists():
            original_data = pd.read_csv(self.config.original_data_path)
            sample_size = min(self.config.original_sample_size, len(original_data) // 2)

            # Sample balanced data
            toxic_df = original_data[original_data['label'] == 1].sample(n=sample_size, random_state=cycle_num)
            clean_df = original_data[original_data['label'] == 0].sample(n=sample_size, random_state=cycle_num)

            for _, row in pd.concat([toxic_df, clean_df]).iterrows():
                train_texts.append(str(row['text']))
                train_labels.append(int(row['label']))

        # Shuffle
        combined = list(zip(train_texts, train_labels))
        random.shuffle(combined)
        train_texts, train_labels = zip(*combined)
        train_texts, train_labels = list(train_texts), list(train_labels)

        logger.info(f"Training dataset: {len(train_texts)} samples")

        # Create dataset
        class SimpleDataset(Dataset):
            def __init__(ds_self, texts, labels, tokenizer, max_length=256):
                ds_self.encodings = tokenizer(
                    texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
                )
                ds_self.labels = torch.tensor(labels)

            def __len__(ds_self):
                return len(ds_self.labels)

            def __getitem__(ds_self, idx):
                return {
                    "input_ids": ds_self.encodings["input_ids"][idx],
                    "attention_mask": ds_self.encodings["attention_mask"][idx],
                    "labels": ds_self.labels[idx],
                }

        dataset = SimpleDataset(train_texts, train_labels, self._tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Training setup
        self._model.train()
        optimizer = AdamW(self._model.parameters(), lr=2e-5)
        total_steps = len(dataloader) * self.config.retrain_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Training loop
        for epoch in range(self.config.retrain_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self._device) for k, v in batch.items()}

                outputs = self._model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{self.config.retrain_epochs}, Loss: {avg_loss:.4f}")

        self._model.eval()

        gpu_time = time.time() - gpu_start
        self._total_gpu_time += gpu_time
        self._retrain_count += 1

        logger.info(f"Retraining complete in {gpu_time:.1f}s")

        return {
            "samples_used": num_samples,
            "total_train_samples": len(train_texts),
            "gpu_time": gpu_time,
            "epochs": self.config.retrain_epochs,
        }

    async def run_cycle(self, cycle_num: int) -> CycleResult:
        """Run one optimized cycle."""
        timestamp = datetime.now(UTC)
        gpu_util = self._get_gpu_utilization()

        # Phase 1: Attack
        evasion_rate, failed_samples = await self.run_attack_phase()
        self._cycles_since_retrain += 1

        # 실패 샘플 축적
        if self.config.accumulate_samples and failed_samples:
            self._accumulated_samples.extend(failed_samples)
            logger.info(f"Accumulated samples: {len(self._accumulated_samples)}")

        # Phase 2: Decide action - GPU를 최대한 활용하도록 조건 완화
        should_retrain = False
        retrain_reason = ""

        if evasion_rate > self.config.defender_retrain_threshold:
            should_retrain = True
            retrain_reason = f"evasion {evasion_rate:.1%} > {self.config.defender_retrain_threshold:.1%}"
        elif len(self._accumulated_samples) >= self.config.accumulate_threshold:
            should_retrain = True
            retrain_reason = f"accumulated {len(self._accumulated_samples)} samples"
        elif self._cycles_since_retrain >= self.config.force_retrain_cycles and len(self._accumulated_samples) > 0:
            should_retrain = True
            retrain_reason = f"force retrain after {self._cycles_since_retrain} cycles"

        if should_retrain:
            action = "retrain_defender"
            logger.info(f"RETRAIN: {retrain_reason}")

            # 축적된 샘플 + 현재 샘플 사용
            all_samples = self._accumulated_samples if self._accumulated_samples else failed_samples
            if len(all_samples) >= self.config.min_samples_for_retrain:
                details = await self.retrain_defender(all_samples, cycle_num)
                self._accumulated_samples = []  # 축적 샘플 초기화
                self._cycles_since_retrain = 0
            else:
                details = {"skipped": True, "reason": f"samples={len(all_samples)}"}

        elif evasion_rate < self.config.attacker_evolve_threshold:
            action = "evolve_attacker"
            logger.info(f"Evasion {evasion_rate:.1%} < {self.config.attacker_evolve_threshold:.1%} → stronger attacks")
            # 더 강력한 공격 시도 - 공격 variants 증가
            self.config.attack_variants = min(30, self.config.attack_variants + 2)
            details = {"status": "evolve_needed", "new_variants": self.config.attack_variants}

        else:
            action = "balanced"
            details = {"status": "accumulating", "accumulated": len(self._accumulated_samples)}

        result = CycleResult(
            cycle_num=cycle_num,
            timestamp=timestamp,
            evasion_rate=evasion_rate,
            action=action,
            details=details,
            gpu_util=gpu_util,
        )

        self.history.append(result)
        self._save_history()

        return result

    def _save_history(self):
        """Save history to file."""
        history_path = Path("data/korean/coevolution_optimized_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "cycle_num": r.cycle_num,
                "timestamp": r.timestamp.isoformat(),
                "evasion_rate": r.evasion_rate,
                "action": r.action,
                "details": r.details,
                "gpu_util": r.gpu_util,
            }
            for r in self.history
        ]

        with open(history_path, "w") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    async def run(self, hours: float = 0, minutes: float = 0):
        """Run optimized co-evolution."""
        total_minutes = hours * 60 + minutes
        end_time = datetime.now(UTC) + timedelta(minutes=total_minutes)
        self._start_time = datetime.now(UTC)

        logger.info("=" * 60)
        logger.info("OPTIMIZED CO-EVOLUTION (GPU Maximized)")
        logger.info("=" * 60)
        logger.info(f"Duration: {total_minutes:.0f} minutes")
        logger.info(f"Base model: {self.config.base_model_path}")
        logger.info(f"Retrain threshold: >{self.config.defender_retrain_threshold:.0%}")
        logger.info(f"Evolve threshold: <{self.config.attacker_evolve_threshold:.0%}")
        logger.info(f"Batch size: {self.config.batch_size} (train), {self.config.eval_batch_size} (eval)")
        logger.info(f"Attack: {self.config.attack_batch_size} samples x {self.config.attack_variants} variants")
        logger.info("=" * 60)

        # Load initial model
        self._load_classifier()

        cycle_num = 0
        while self._running and datetime.now(UTC) < end_time:
            cycle_num += 1

            try:
                result = await self.run_cycle(cycle_num)

                remaining = (end_time - datetime.now(UTC)).total_seconds() / 60
                logger.info(
                    f"[Cycle {cycle_num}] {result.action} | "
                    f"Evasion: {result.evasion_rate:.1%} | "
                    f"Retrains: {self._retrain_count} | "
                    f"GPU Time: {self._total_gpu_time:.0f}s | "
                    f"Remaining: {remaining:.0f}min"
                )

            except Exception as e:
                logger.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)

            # Small delay between cycles
            if self._running and datetime.now(UTC) < end_time:
                await asyncio.sleep(self.config.cycle_interval_seconds)

        self._print_summary()

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("OPTIMIZED CO-EVOLUTION COMPLETE")
        print("=" * 60)

        if not self.history:
            print("No cycles completed")
            return

        total_time = (datetime.now(UTC) - self._start_time).total_seconds()

        print(f"\nTotal cycles: {len(self.history)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Retrains: {self._retrain_count}")
        print(f"GPU time: {self._total_gpu_time:.1f}s ({self._total_gpu_time/total_time*100:.1f}%)")

        actions = {}
        evasion_rates = []
        for r in self.history:
            actions[r.action] = actions.get(r.action, 0) + 1
            evasion_rates.append(r.evasion_rate)

        print(f"\nActions:")
        for action, count in actions.items():
            print(f"  {action}: {count}")

        print(f"\nEvasion rate:")
        print(f"  Start: {evasion_rates[0]:.1%}")
        print(f"  End: {evasion_rates[-1]:.1%}")
        print(f"  Min: {min(evasion_rates):.1%}")
        print(f"  Max: {max(evasion_rates):.1%}")

        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Optimized Korean Co-Evolution")
    parser.add_argument("--hours", type=float, default=0, help="Hours to run")
    parser.add_argument("--minutes", type=float, default=30, help="Minutes to run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--retrain-threshold", type=float, default=0.12,
                        help="Evasion threshold for retraining (default: 0.12)")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = OptimizedCoevolutionConfig(
        defender_retrain_threshold=args.retrain_threshold,
    )

    coevolution = OptimizedCoevolution(config)
    await coevolution.run(hours=args.hours, minutes=args.minutes)


if __name__ == "__main__":
    asyncio.run(main())
