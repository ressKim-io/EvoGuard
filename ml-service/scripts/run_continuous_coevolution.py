#!/usr/bin/env python3
"""Continuous Co-Evolution Training (Trigger-Based).

작업 완료 기반 공진화 학습:
- 시간 기반 X → 작업 완료 즉시 다음 사이클
- GPU 유휴 시간 최소화
- AMP(Mixed Precision) 적용
- Early Stopping (수렴 감지)

Usage:
    # 무한 실행 (Ctrl+C로 중단)
    python scripts/run_continuous_coevolution.py

    # 사이클 수 제한
    python scripts/run_continuous_coevolution.py --max-cycles 100

    # 목표 evasion rate 도달 시 중단
    python scripts/run_continuous_coevolution.py --target-evasion 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from ml_service.attacker.slang_dictionary import (
    evolve_attack_corpus,
    get_all_slang,
    generate_variants,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ContinuousConfig:
    """Configuration for continuous co-evolution."""

    # 트리거 조건 (하나라도 만족하면 재학습)
    retrain_evasion_threshold: float = 0.08  # 8% 초과시 재학습
    retrain_sample_threshold: int = 50       # 50개 샘플 축적시 재학습
    retrain_cycle_interval: int = 3          # 3 사이클마다 강제 재학습

    # 수렴 감지 (Early Stopping)
    convergence_threshold: float = 0.03      # 3% 미만이면 수렴
    convergence_patience: int = 10           # 10 사이클 연속 수렴시 공격 강화

    # 학습 설정 (RTX 4060 Ti 8GB 최적화)
    batch_size: int = 16
    retrain_epochs: int = 2
    learning_rate: float = 2e-5
    use_amp: bool = True  # Mixed Precision

    # 공격 설정
    attack_batch_size: int = 150
    attack_variants: int = 15
    max_attack_variants: int = 40  # 공격 강화 시 최대값

    # 공격-방어 균형 (GAN 스타일)
    attack_defense_ratio: int = 1  # N번 공격 후 1번 방어

    # 데이터
    original_data_path: Path = field(default_factory=lambda: Path("data/korean/korean_hate_speech_full.csv"))
    original_sample_size: int = 2000

    # 모델
    base_model_path: Path = field(default_factory=lambda: Path("models/phase2-combined/best_model"))


@dataclass
class CycleStats:
    """Statistics for a cycle."""
    cycle_num: int
    evasion_rate: float
    action: str
    gpu_time: float = 0.0
    train_loss: float = 0.0
    samples_used: int = 0


class ContinuousCoevolution:
    """Continuous co-evolution with trigger-based retraining."""

    def __init__(self, config: ContinuousConfig) -> None:
        self.config = config
        self._running = True
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # 상태
        self._model = None
        self._tokenizer = None
        self._classifier = None
        self._scaler = GradScaler() if config.use_amp else None

        # 통계
        self._cycle_num = 0
        self._retrain_count = 0
        self._total_gpu_time = 0.0
        self._accumulated_samples = []
        self._cycles_since_retrain = 0
        self._convergence_count = 0  # 연속 수렴 횟수
        self._history: list[CycleStats] = []
        self._start_time = None

        # 슬랭 기반 공격 큐
        self._slang_attack_queue: list[dict] = []
        self._slang_evasion_count = 0  # 슬랭 탐지 우회 누적

        # 시그널 핸들러
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._log_config()

    def _log_config(self):
        """Log configuration."""
        logger.info("=" * 60)
        logger.info("CONTINUOUS CO-EVOLUTION (Trigger-Based)")
        logger.info("=" * 60)
        logger.info(f"Device: {self._device}")
        if self._device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"AMP: {self.config.use_amp}")
        logger.info(f"Retrain triggers:")
        logger.info(f"  - Evasion > {self.config.retrain_evasion_threshold:.0%}")
        logger.info(f"  - Samples >= {self.config.retrain_sample_threshold}")
        logger.info(f"  - Every {self.config.retrain_cycle_interval} cycles")
        logger.info(f"Convergence: < {self.config.convergence_threshold:.0%} for {self.config.convergence_patience} cycles")
        logger.info("=" * 60)

    def _handle_signal(self, signum, frame):
        logger.info("\n[STOP] Graceful shutdown requested...")
        self._running = False

    def _load_model(self):
        """Load model with GPU optimization."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading model from {self.config.base_model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.config.base_model_path)
        self._model.to(self._device)
        self._model.eval()

        # Classifier wrapper
        class Classifier:
            def __init__(cls, model, tokenizer, device, use_amp):
                cls.model = model
                cls.tokenizer = tokenizer
                cls.device = device
                cls.use_amp = use_amp

            def predict(cls, texts):
                if isinstance(texts, str):
                    texts = [texts]

                cls.model.eval()
                with torch.no_grad():
                    enc = cls.tokenizer(
                        texts, truncation=True, padding=True,
                        max_length=256, return_tensors="pt"
                    )
                    enc = {k: v.to(cls.device) for k, v in enc.items()}

                    if cls.use_amp:
                        with autocast():
                            out = cls.model(**enc)
                    else:
                        out = cls.model(**enc)

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

        self._classifier = Classifier(self._model, self._tokenizer, self._device, self.config.use_amp)
        logger.info(f"Model loaded on {self._device}")

        # 초기 슬랭 공격 큐 채우기
        self._initialize_slang_attacks()

    def _initialize_slang_attacks(self):
        """슬랭 사전에서 초기 공격 표현 로드."""
        all_slang = get_all_slang()
        for word in all_slang:
            variants = generate_variants(word, num_variants=3)
            for variant in variants:
                self._slang_attack_queue.append({
                    "text": variant,
                    "label": 1,
                    "source": "slang_init",
                    "original": word,
                })
        logger.info(f"[INIT] 슬랭 사전에서 {len(self._slang_attack_queue)}개 공격 표현 로드")

    def _run_attack(self) -> tuple[float, list]:
        """Run attack phase."""
        from ml_service.pipeline.korean_attack_runner import KoreanAttackRunner
        from ml_service.pipeline.korean_config import KoreanAttackConfig

        config = KoreanAttackConfig(
            batch_size=self.config.attack_batch_size,
            num_variants=self.config.attack_variants,
        )

        runner = KoreanAttackRunner(config, self._classifier)
        result = runner.run_batch()

        return result.evasion_rate, result.get_failed_samples()

    def _retrain(self, samples: list) -> dict:
        """Retrain with AMP optimization."""
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        import pandas as pd
        import random

        gpu_start = time.time()

        # Prepare data
        train_texts, train_labels = [], []

        for s in samples:
            train_texts.extend([s.variant_text, s.original_text])
            train_labels.extend([s.original_label, s.original_label])

        # Add original data
        if self.config.original_data_path.exists():
            df = pd.read_csv(self.config.original_data_path)
            sample_size = min(self.config.original_sample_size, len(df) // 2)

            toxic = df[df['label'] == 1].sample(n=sample_size, random_state=self._cycle_num)
            clean = df[df['label'] == 0].sample(n=sample_size, random_state=self._cycle_num)

            for _, row in pd.concat([toxic, clean]).iterrows():
                train_texts.append(str(row['text']))
                train_labels.append(int(row['label']))

        # Shuffle
        combined = list(zip(train_texts, train_labels))
        random.shuffle(combined)
        train_texts, train_labels = zip(*combined)

        logger.info(f"Training on {len(train_texts)} samples")

        # Dataset
        class SimpleDataset(Dataset):
            def __init__(ds, texts, labels, tokenizer):
                ds.encodings = tokenizer(
                    list(texts), truncation=True, padding=True,
                    max_length=256, return_tensors="pt"
                )
                ds.labels = torch.tensor(list(labels))

            def __len__(ds):
                return len(ds.labels)

            def __getitem__(ds, idx):
                return {
                    "input_ids": ds.encodings["input_ids"][idx],
                    "attention_mask": ds.encodings["attention_mask"][idx],
                    "labels": ds.labels[idx],
                }

        dataset = SimpleDataset(train_texts, train_labels, self._tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,  # GPU 전송 최적화
        )

        # Training
        self._model.train()
        optimizer = AdamW(self._model.parameters(), lr=self.config.learning_rate)
        total_steps = len(dataloader) * self.config.retrain_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

        total_loss = 0
        for epoch in range(self.config.retrain_epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self._device) for k, v in batch.items()}

                optimizer.zero_grad()

                if self.config.use_amp:
                    with autocast():
                        outputs = self._model(**batch)
                        loss = outputs.loss

                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self._model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss
            logger.info(f"  Epoch {epoch+1}/{self.config.retrain_epochs}: loss={avg_loss:.4f}")

        self._model.eval()
        gpu_time = time.time() - gpu_start

        return {
            "gpu_time": gpu_time,
            "train_loss": total_loss / self.config.retrain_epochs,
            "samples": len(train_texts),
        }

    def _should_retrain(self, evasion_rate: float, num_samples: int) -> tuple[bool, str]:
        """Check if retraining should be triggered."""
        self._cycles_since_retrain += 1

        # Trigger 1: Evasion rate 초과
        if evasion_rate > self.config.retrain_evasion_threshold:
            return True, f"evasion {evasion_rate:.1%} > {self.config.retrain_evasion_threshold:.0%}"

        # Trigger 2: 샘플 축적
        if len(self._accumulated_samples) >= self.config.retrain_sample_threshold:
            return True, f"accumulated {len(self._accumulated_samples)} samples"

        # Trigger 3: 주기적 재학습
        if self._cycles_since_retrain >= self.config.retrain_cycle_interval and self._accumulated_samples:
            return True, f"periodic (every {self.config.retrain_cycle_interval} cycles)"

        return False, ""

    def _check_convergence(self, evasion_rate: float) -> bool:
        """Check if model has converged (too good)."""
        if evasion_rate < self.config.convergence_threshold:
            self._convergence_count += 1
            if self._convergence_count >= self.config.convergence_patience:
                return True
        else:
            self._convergence_count = 0
        return False

    def _strengthen_attack(self):
        """Strengthen attack when model converges - 슬랭 사전으로 진화."""
        old_variants = self.config.attack_variants
        self.config.attack_variants = min(
            self.config.attack_variants + 5,
            self.config.max_attack_variants
        )
        self._convergence_count = 0
        logger.info(f"[EVOLVE] Attack variants: {old_variants} → {self.config.attack_variants}")

        # 슬랭 사전에서 새로운 공격 표현 생성
        last_evasion = self._history[-1].evasion_rate if self._history else 0.1
        new_attack_samples = evolve_attack_corpus(
            current_evasion_rate=last_evasion,
            blocked_strategies=None,
        )

        if new_attack_samples:
            logger.info(f"[EVOLVE] 슬랭 사전에서 {len(new_attack_samples)}개 새 공격 표현 추가")
            self._slang_attack_queue.extend(new_attack_samples)

    def _run_slang_attack(self) -> tuple[float, list]:
        """슬랭 사전 기반 공격 실행."""
        if not self._slang_attack_queue:
            return 0.0, []

        # 큐에서 최대 50개 가져오기
        samples = self._slang_attack_queue[:50]
        self._slang_attack_queue = self._slang_attack_queue[50:]

        texts = [s["text"] for s in samples]
        predictions = self._classifier.predict(texts)

        evasions = []
        for sample, pred in zip(samples, predictions):
            expected_label = sample["label"]
            if pred["label"] != expected_label:
                # 슬랭이 탐지 우회함 - 방어 강화 필요
                evasions.append({
                    "original_text": sample.get("original", sample["text"]),
                    "variant_text": sample["text"],
                    "strategy_name": sample.get("source", "slang_evolution"),
                    "original_label": expected_label,
                    "model_prediction": pred["label"],
                    "model_confidence": pred["confidence"],
                    "is_evasion": True,
                })

        evasion_rate = len(evasions) / len(samples) if samples else 0
        logger.info(f"[SLANG] {len(evasions)}/{len(samples)} 슬랭 표현 탐지 우회 ({evasion_rate:.1%})")

        return evasion_rate, evasions

    async def run_cycle(self) -> CycleStats:
        """Run one cycle (attack → decide → maybe retrain)."""
        self._cycle_num += 1

        # Attack phase
        evasion_rate, failed_samples = self._run_attack()

        # 슬랭 공격도 실행 (큐에 있으면)
        if self._slang_attack_queue:
            slang_evasion, slang_fails = self._run_slang_attack()
            if slang_fails:
                # 슬랭 탐지 실패를 샘플에 변환하여 추가
                for fail in slang_fails:
                    class SlangSample:
                        def __init__(self, data):
                            self.variant_text = data["variant_text"]
                            self.original_text = data["original_text"]
                            self.original_label = data["original_label"]
                    failed_samples.append(SlangSample(fail))
                self._slang_evasion_count += len(slang_fails)

        # Accumulate samples
        if failed_samples:
            self._accumulated_samples.extend(failed_samples)

        # Check convergence (model too good)
        if self._check_convergence(evasion_rate):
            self._strengthen_attack()

        # Decide action
        should_retrain, reason = self._should_retrain(evasion_rate, len(failed_samples))

        stats = CycleStats(
            cycle_num=self._cycle_num,
            evasion_rate=evasion_rate,
            action="",
        )

        if should_retrain and self._accumulated_samples:
            stats.action = "retrain"
            logger.info(f"[RETRAIN] {reason}")

            result = self._retrain(self._accumulated_samples)

            stats.gpu_time = result["gpu_time"]
            stats.train_loss = result["train_loss"]
            stats.samples_used = result["samples"]

            self._total_gpu_time += result["gpu_time"]
            self._retrain_count += 1
            self._accumulated_samples = []
            self._cycles_since_retrain = 0

        else:
            stats.action = "attack_only"

        self._history.append(stats)
        self._save_history()

        return stats

    def _save_history(self):
        """Save history to file."""
        path = Path("data/korean/coevolution_continuous_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "cycle": s.cycle_num,
                "evasion_rate": s.evasion_rate,
                "action": s.action,
                "gpu_time": s.gpu_time,
                "train_loss": s.train_loss,
            }
            for s in self._history
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    async def run(
        self,
        max_cycles: int | None = None,
        target_evasion: float | None = None,
    ):
        """Run continuous co-evolution."""
        self._start_time = time.time()
        self._load_model()

        logger.info("\n[START] Continuous co-evolution running...")
        logger.info("Press Ctrl+C to stop gracefully\n")

        while self._running:
            # Check termination conditions
            if max_cycles and self._cycle_num >= max_cycles:
                logger.info(f"[DONE] Reached max cycles: {max_cycles}")
                break

            if target_evasion and self._history:
                if self._history[-1].evasion_rate <= target_evasion:
                    logger.info(f"[DONE] Reached target evasion: {target_evasion:.1%}")
                    break

            # Run cycle
            try:
                stats = await self.run_cycle()

                # Log progress
                elapsed = time.time() - self._start_time
                logger.info(
                    f"[Cycle {stats.cycle_num}] "
                    f"evasion={stats.evasion_rate:.1%} | "
                    f"action={stats.action} | "
                    f"retrains={self._retrain_count} | "
                    f"gpu_time={self._total_gpu_time:.0f}s | "
                    f"elapsed={elapsed/60:.1f}min"
                )

            except Exception as e:
                logger.error(f"Cycle failed: {e}", exc_info=True)
                await asyncio.sleep(1)

        self._print_summary()

    def _print_summary(self):
        """Print final summary."""
        elapsed = time.time() - self._start_time

        print("\n" + "=" * 60)
        print("CO-EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Total cycles: {self._cycle_num}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Retrains: {self._retrain_count}")
        print(f"GPU time: {self._total_gpu_time:.1f}s ({self._total_gpu_time/elapsed*100:.1f}%)")

        if self._history:
            evasions = [s.evasion_rate for s in self._history]
            print(f"\nEvasion rate:")
            print(f"  Start: {evasions[0]:.1%}")
            print(f"  End: {evasions[-1]:.1%}")
            print(f"  Min: {min(evasions):.1%}")
            print(f"  Max: {max(evasions):.1%}")

            # Trend
            if len(evasions) > 5:
                recent = sum(evasions[-5:]) / 5
                early = sum(evasions[:5]) / 5
                trend = "↓ improving" if recent < early else "↑ worsening"
                print(f"  Trend: {trend}")

        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Continuous Co-Evolution (Trigger-Based)")
    parser.add_argument("--max-cycles", type=int, help="Max cycles (default: unlimited)")
    parser.add_argument("--target-evasion", type=float, help="Stop when evasion <= target")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = ContinuousConfig(
        use_amp=not args.no_amp,
    )

    coevolution = ContinuousCoevolution(config)
    await coevolution.run(
        max_cycles=args.max_cycles,
        target_evasion=args.target_evasion,
    )


if __name__ == "__main__":
    asyncio.run(main())
