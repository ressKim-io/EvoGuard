#!/usr/bin/env python3
"""Balanced Co-Evolution Training.

공격자-방어자 양쪽이 자동으로 진화하는 균형 공진화 시스템.

Features:
- AttackerEvolver: evasion < 5%일 때 공격자 자동 진화
- HardNegativeMiner: 어려운 샘플 수집 및 집중 학습
- EvolutionTracker: 진화 이벤트 및 균형 구간 추적

Usage:
    # 기본 실행
    python scripts/run_balanced_coevolution.py

    # 100 사이클
    python scripts/run_balanced_coevolution.py --max-cycles 100

    # 목표 evasion rate
    python scripts/run_balanced_coevolution.py --target-evasion 0.03
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
from ml_service.attacker.learning_attacker import (
    LearningAttacker,
    LearningAttackerConfig,
)
from ml_service.attacker.attacker_evolver import (
    AttackerEvolver,
    AttackerEvolverConfig,
)
from ml_service.pipeline.hard_negative_miner import (
    HardNegativeMiner,
    HardNegativeMinerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BalancedConfig:
    """Configuration for balanced co-evolution."""

    # 균형 구간 정의
    balance_low: float = 0.05   # 5% - 균형 하한
    balance_high: float = 0.08  # 8% - 균형 상한

    # 공격자 진화 트리거
    evolve_attacker_threshold: float = 0.05  # 5% 미만이면 공격자 진화

    # 방어자 재학습 트리거
    retrain_defender_threshold: float = 0.08  # 8% 초과시 재학습

    # 균형 구간 행동
    balance_evolve_interval: int = 5   # 균형 구간에서 N 사이클마다 약한 진화
    balance_retrain_interval: int = 5  # 균형 구간에서 N 사이클마다 약한 재학습

    # HNM 트리거
    hnm_retrain_threshold: int = 100  # HNM 샘플 100개 이상이면 재학습

    # 학습 설정 (RTX 4060 Ti 8GB 최적화)
    batch_size: int = 16
    retrain_epochs: int = 2
    weak_retrain_epochs: int = 1  # 균형 구간 약한 재학습
    learning_rate: float = 2e-5
    use_amp: bool = True

    # 공격 설정
    attack_batch_size: int = 150
    attack_variants: int = 15

    # 데이터
    original_data_path: Path = field(default_factory=lambda: Path("data/korean/korean_hate_speech_full.csv"))
    original_sample_size: int = 2000

    # 모델
    base_model_path: Path = field(default_factory=lambda: Path("models/phase2-slang-enhanced"))


@dataclass
class CycleStats:
    """Statistics for a cycle."""

    cycle_num: int
    evasion_rate: float
    action: str
    in_balance_zone: bool = False
    gpu_time: float = 0.0
    train_loss: float = 0.0
    samples_used: int = 0
    hnm_samples: int = 0
    new_strategies: int = 0
    new_slang: int = 0


class EvolutionTracker:
    """Track evolution events and balance zone metrics."""

    def __init__(
        self,
        balance_low: float = 0.05,
        balance_high: float = 0.08,
    ) -> None:
        self.balance_low = balance_low
        self.balance_high = balance_high

        # 상태
        self._in_balance_zone = False
        self._balance_zone_cycles = 0
        self._total_balance_cycles = 0

        # 이벤트 기록
        self._events: list[dict] = []
        self._balance_entries: list[dict] = []
        self._balance_exits: list[dict] = []

    def record_event(
        self,
        cycle: int,
        evasion_rate: float,
        action: str,
        details: dict | None = None,
    ) -> None:
        """Record an evolution event."""
        event = {
            "cycle": cycle,
            "evasion_rate": evasion_rate,
            "action": action,
            "in_balance_zone": self._in_balance_zone,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": details or {},
        }
        self._events.append(event)

        # 균형 구간 체크
        in_balance = self.balance_low <= evasion_rate <= self.balance_high

        if in_balance and not self._in_balance_zone:
            self.enter_balance_zone(cycle, evasion_rate)
        elif not in_balance and self._in_balance_zone:
            self.exit_balance_zone(cycle, evasion_rate)

        if in_balance:
            self._balance_zone_cycles += 1
            self._total_balance_cycles += 1

    def enter_balance_zone(self, cycle: int, evasion_rate: float) -> None:
        """Record entering balance zone."""
        self._in_balance_zone = True
        self._balance_zone_cycles = 0

        entry = {
            "cycle": cycle,
            "evasion_rate": evasion_rate,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._balance_entries.append(entry)
        logger.info(f"[BALANCE] Entered balance zone at cycle {cycle} (evasion={evasion_rate:.1%})")

    def exit_balance_zone(self, cycle: int, evasion_rate: float) -> None:
        """Record exiting balance zone."""
        duration = self._balance_zone_cycles
        self._in_balance_zone = False

        exit_record = {
            "cycle": cycle,
            "evasion_rate": evasion_rate,
            "duration": duration,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._balance_exits.append(exit_record)
        logger.info(
            f"[BALANCE] Exited balance zone at cycle {cycle} "
            f"(evasion={evasion_rate:.1%}, duration={duration} cycles)"
        )

    def is_in_balance_zone(self) -> bool:
        """Check if currently in balance zone."""
        return self._in_balance_zone

    def get_balance_zone_cycles(self) -> int:
        """Get consecutive cycles in balance zone."""
        return self._balance_zone_cycles

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "in_balance_zone": self._in_balance_zone,
            "balance_zone_cycles": self._balance_zone_cycles,
            "total_balance_cycles": self._total_balance_cycles,
            "balance_entries": len(self._balance_entries),
            "balance_exits": len(self._balance_exits),
            "total_events": len(self._events),
        }


class BalancedCoevolution:
    """Balanced co-evolution with attacker evolution and HNM."""

    def __init__(self, config: BalancedConfig) -> None:
        self.config = config
        self._running = True
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # 상태
        self._model = None
        self._tokenizer = None
        self._classifier = None
        self._scaler = GradScaler() if config.use_amp else None

        # 컴포넌트
        self._learning_attacker: LearningAttacker | None = None
        self._attacker_evolver: AttackerEvolver | None = None
        self._hard_negative_miner: HardNegativeMiner | None = None
        self._evolution_tracker: EvolutionTracker | None = None

        # 통계
        self._cycle_num = 0
        self._retrain_count = 0
        self._evolve_count = 0
        self._total_gpu_time = 0.0
        self._accumulated_samples = []
        self._history: list[CycleStats] = []
        self._start_time = None

        # 시그널 핸들러
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._log_config()

    def _log_config(self):
        """Log configuration."""
        logger.info("=" * 60)
        logger.info("BALANCED CO-EVOLUTION")
        logger.info("=" * 60)
        logger.info(f"Device: {self._device}")
        if self._device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"AMP: {self.config.use_amp}")
        logger.info(f"Balance zone: {self.config.balance_low:.0%} - {self.config.balance_high:.0%}")
        logger.info(f"Attacker evolution: evasion < {self.config.evolve_attacker_threshold:.0%}")
        logger.info(f"Defender retraining: evasion > {self.config.retrain_defender_threshold:.0%}")
        logger.info(f"HNM threshold: {self.config.hnm_retrain_threshold} samples")
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

        # 컴포넌트 초기화
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all components."""
        # Evolution Tracker
        self._evolution_tracker = EvolutionTracker(
            balance_low=self.config.balance_low,
            balance_high=self.config.balance_high,
        )

        # Hard Negative Miner
        hnm_config = HardNegativeMinerConfig(
            retrain_threshold=self.config.hnm_retrain_threshold,
        )
        self._hard_negative_miner = HardNegativeMiner(config=hnm_config)
        logger.info(f"[INIT] HardNegativeMiner: {self._hard_negative_miner}")

        # Learning Attacker
        la_config = LearningAttackerConfig(
            batch_size=self.config.attack_batch_size,
            num_variants=self.config.attack_variants,
            exploration_weight=2.0,
            auto_generate_slang=True,
            auto_generate_strategies=True,
        )
        self._learning_attacker = LearningAttacker(
            config=la_config,
            classifier=self._classifier,
        )
        logger.info(f"[INIT] LearningAttacker: {self._learning_attacker}")

        # Attacker Evolver
        evolver_config = AttackerEvolverConfig()
        self._attacker_evolver = AttackerEvolver(
            config=evolver_config,
            classifier=self._classifier,
        )
        self._attacker_evolver.set_learning_attacker(self._learning_attacker)
        logger.info(f"[INIT] AttackerEvolver: {self._attacker_evolver}")

    def _determine_action(self, evasion_rate: float) -> str:
        """Determine what action to take.

        Args:
            evasion_rate: 현재 evasion rate

        Returns:
            Action: "evolve_attacker", "retrain_defender", "retrain_defender_hnm",
                   "balance_evolve", "balance_retrain", "none"
        """
        in_balance = self._evolution_tracker.is_in_balance_zone()
        balance_cycles = self._evolution_tracker.get_balance_zone_cycles()

        # HNM 샘플이 많으면 우선 재학습
        if self._hard_negative_miner.should_trigger_retrain():
            return "retrain_defender_hnm"

        # 방어자 재학습 (evasion > 8%)
        if evasion_rate > self.config.retrain_defender_threshold:
            return "retrain_defender"

        # 공격자 진화 (evasion < 5%)
        if evasion_rate < self.config.evolve_attacker_threshold:
            return "evolve_attacker"

        # 균형 구간 (5% <= evasion <= 8%)
        if in_balance:
            # 주기적으로 약한 진화/재학습
            if balance_cycles > 0 and balance_cycles % self.config.balance_evolve_interval == 0:
                return "balance_evolve"
            if balance_cycles > 0 and balance_cycles % self.config.balance_retrain_interval == 0:
                if self._accumulated_samples:
                    return "balance_retrain"

        return "none"

    def _evolve_attacker(self, evasion_rate: float, attack_results: list) -> dict:
        """Evolve attacker when it's too weak.

        Args:
            evasion_rate: 현재 evasion rate
            attack_results: 공격 결과

        Returns:
            Evolution summary
        """
        # 성공한 우회 추출
        successful_evasions = [
            {
                "variant_text": r.variant_text,
                "original_text": r.original_text,
            }
            for r in attack_results
            if r.is_evasion
        ]

        result = self._attacker_evolver.evolve(
            evasion_rate=evasion_rate,
            successful_evasions=successful_evasions,
        )

        self._evolve_count += 1
        return result

    def _retrain_with_hnm(self) -> dict:
        """Retrain with Hard Negative Mining samples.

        Returns:
            Training result
        """
        texts, labels, weights = self._hard_negative_miner.get_training_data(
            max_samples=500,
            clear_after=True,
        )

        if not texts:
            return {"samples": 0}

        logger.info(f"[HNM] Training with {len(texts)} hard negative samples")

        # 일반 재학습과 합침
        result = self._retrain(
            samples=self._accumulated_samples,
            extra_texts=texts,
            extra_labels=labels,
            extra_weights=weights,
        )

        self._accumulated_samples = []
        return result

    def _retrain(
        self,
        samples: list,
        extra_texts: list | None = None,
        extra_labels: list | None = None,
        extra_weights: list | None = None,
        epochs: int | None = None,
    ) -> dict:
        """Retrain with AMP optimization."""
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        import pandas as pd
        import random

        gpu_start = time.time()
        epochs = epochs or self.config.retrain_epochs

        # Prepare data
        train_texts, train_labels, train_weights = [], [], []

        for s in samples:
            train_texts.extend([s.variant_text, s.original_text])
            train_labels.extend([s.original_label, s.original_label])
            train_weights.extend([1.0, 1.0])

        # HNM 데이터 추가
        if extra_texts:
            train_texts.extend(extra_texts)
            train_labels.extend(extra_labels)
            train_weights.extend(extra_weights or [1.0] * len(extra_texts))

        # Add original data
        if self.config.original_data_path.exists():
            df = pd.read_csv(self.config.original_data_path)
            sample_size = min(self.config.original_sample_size, len(df) // 2)

            toxic = df[df['label'] == 1].sample(n=sample_size, random_state=self._cycle_num)
            clean = df[df['label'] == 0].sample(n=sample_size, random_state=self._cycle_num)

            for _, row in pd.concat([toxic, clean]).iterrows():
                train_texts.append(str(row['text']))
                train_labels.append(int(row['label']))
                train_weights.append(1.0)

        # Shuffle
        combined = list(zip(train_texts, train_labels, train_weights))
        random.shuffle(combined)
        train_texts, train_labels, train_weights = zip(*combined)

        logger.info(f"Training on {len(train_texts)} samples")

        # Dataset with weights
        class WeightedDataset(Dataset):
            def __init__(ds, texts, labels, weights, tokenizer):
                ds.encodings = tokenizer(
                    list(texts), truncation=True, padding=True,
                    max_length=256, return_tensors="pt"
                )
                ds.labels = torch.tensor(list(labels))
                ds.weights = torch.tensor(list(weights))

            def __len__(ds):
                return len(ds.labels)

            def __getitem__(ds, idx):
                return {
                    "input_ids": ds.encodings["input_ids"][idx],
                    "attention_mask": ds.encodings["attention_mask"][idx],
                    "labels": ds.labels[idx],
                    "weight": ds.weights[idx],
                }

        dataset = WeightedDataset(train_texts, train_labels, train_weights, self._tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # Training
        self._model.train()
        optimizer = AdamW(self._model.parameters(), lr=self.config.learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)
                weights = batch["weight"].to(self._device)

                optimizer.zero_grad()

                if self.config.use_amp:
                    with autocast():
                        outputs = self._model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        # 가중치 적용된 손실
                        loss = (outputs.loss * weights).mean()

                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = (outputs.loss * weights).mean()
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        self._model.eval()
        gpu_time = time.time() - gpu_start

        return {
            "gpu_time": gpu_time,
            "train_loss": total_loss / epochs,
            "samples": len(train_texts),
        }

    def _run_attack(self) -> tuple[float, list]:
        """Run attack phase."""
        result = self._learning_attacker.run_batch()

        # HNM에 결과 전달
        attack_dicts = [
            {
                "variant_text": r.variant_text,
                "original_text": r.original_text,
                "original_label": r.original_label,
                "model_prediction": r.model_prediction,
                "model_confidence": r.model_confidence,
                "is_evasion": r.is_evasion,
                "strategy_name": r.strategy_name,
            }
            for r in result.attack_results
        ]
        self._hard_negative_miner.mine_from_attack(attack_dicts)

        return result.evasion_rate, result.get_failed_samples()

    async def run_cycle(self) -> CycleStats:
        """Run one cycle."""
        self._cycle_num += 1

        # Attack phase
        evasion_rate, failed_samples = self._run_attack()

        # Accumulate samples
        if failed_samples:
            self._accumulated_samples.extend(failed_samples)

        # Determine action
        action = self._determine_action(evasion_rate)

        # Record event
        self._evolution_tracker.record_event(
            cycle=self._cycle_num,
            evasion_rate=evasion_rate,
            action=action,
        )

        stats = CycleStats(
            cycle_num=self._cycle_num,
            evasion_rate=evasion_rate,
            action=action,
            in_balance_zone=self._evolution_tracker.is_in_balance_zone(),
            hnm_samples=len(self._hard_negative_miner),
        )

        # Execute action
        if action == "evolve_attacker":
            result = self._evolve_attacker(evasion_rate, failed_samples)
            stats.new_strategies = result.get("new_strategies", 0)
            stats.new_slang = result.get("new_slang", 0)

        elif action == "retrain_defender":
            if self._accumulated_samples:
                result = self._retrain(self._accumulated_samples)
                stats.gpu_time = result["gpu_time"]
                stats.train_loss = result["train_loss"]
                stats.samples_used = result["samples"]
                self._total_gpu_time += result["gpu_time"]
                self._retrain_count += 1
                self._accumulated_samples = []

        elif action == "retrain_defender_hnm":
            result = self._retrain_with_hnm()
            stats.gpu_time = result.get("gpu_time", 0)
            stats.train_loss = result.get("train_loss", 0)
            stats.samples_used = result.get("samples", 0)
            self._total_gpu_time += result.get("gpu_time", 0)
            self._retrain_count += 1

        elif action == "balance_evolve":
            result = self._attacker_evolver.evolve(
                evasion_rate=evasion_rate,
                force_mode="maintenance",
            )
            stats.new_strategies = result.get("new_strategies", 0)

        elif action == "balance_retrain":
            if self._accumulated_samples:
                result = self._retrain(
                    self._accumulated_samples,
                    epochs=self.config.weak_retrain_epochs,
                )
                stats.gpu_time = result["gpu_time"]
                stats.train_loss = result["train_loss"]
                stats.samples_used = result["samples"]
                self._total_gpu_time += result["gpu_time"]
                self._retrain_count += 1
                self._accumulated_samples = []

        self._history.append(stats)
        self._save_history()

        return stats

    def _save_history(self):
        """Save history to file."""
        path = Path("data/korean/coevolution_balanced_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "cycle": s.cycle_num,
                "evasion_rate": s.evasion_rate,
                "action": s.action,
                "in_balance_zone": s.in_balance_zone,
                "gpu_time": s.gpu_time,
                "train_loss": s.train_loss,
                "hnm_samples": s.hnm_samples,
                "new_strategies": s.new_strategies,
                "new_slang": s.new_slang,
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
        """Run balanced co-evolution."""
        self._start_time = time.time()
        self._load_model()

        logger.info("\n[START] Balanced co-evolution running...")
        logger.info("Press Ctrl+C to stop gracefully\n")

        while self._running:
            # Check termination conditions
            if max_cycles and self._cycle_num >= max_cycles:
                logger.info(f"[DONE] Reached max cycles: {max_cycles}")
                break

            if target_evasion and self._history:
                recent = [s.evasion_rate for s in self._history[-5:]]
                if recent and sum(recent) / len(recent) <= target_evasion:
                    logger.info(f"[DONE] Reached target evasion: {target_evasion:.1%}")
                    break

            # Run cycle
            try:
                stats = await self.run_cycle()

                # Log progress
                elapsed = time.time() - self._start_time
                balance_marker = "[B]" if stats.in_balance_zone else "   "

                logger.info(
                    f"[Cycle {stats.cycle_num}] {balance_marker} "
                    f"evasion={stats.evasion_rate:.1%} | "
                    f"action={stats.action} | "
                    f"retrains={self._retrain_count} | "
                    f"evolves={self._evolve_count} | "
                    f"hnm={stats.hnm_samples} | "
                    f"elapsed={elapsed/60:.1f}min"
                )

            except Exception as e:
                logger.error(f"Cycle failed: {e}", exc_info=True)
                await asyncio.sleep(1)

        self._save_model()
        self._print_summary()

    def _save_model(self):
        """Save trained model."""
        save_path = Path("models/coevolution-latest")
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        logger.info(f"[SAVE] Model saved to {save_path}")

        # Learning Attacker 상태 저장
        if self._learning_attacker:
            self._learning_attacker.save_state()

        # 버전 스냅샷
        try:
            import subprocess
            subprocess.run(
                ["python", "scripts/model_version_manager.py", "save", "--tag", f"balanced_cycle{self._cycle_num}"],
                capture_output=True,
                cwd=PROJECT_ROOT,
            )
            subprocess.run(
                ["python", "scripts/model_version_manager.py", "prune", "--keep", "3"],
                capture_output=True,
                cwd=PROJECT_ROOT,
            )
        except Exception as e:
            logger.warning(f"[VERSION] Version management failed: {e}")

    def _print_summary(self):
        """Print final summary."""
        elapsed = time.time() - self._start_time

        print("\n" + "=" * 60)
        print("BALANCED CO-EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Total cycles: {self._cycle_num}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Retrains: {self._retrain_count}")
        print(f"Evolutions: {self._evolve_count}")
        print(f"GPU time: {self._total_gpu_time:.1f}s ({self._total_gpu_time/elapsed*100:.1f}%)")

        if self._history:
            evasions = [s.evasion_rate for s in self._history]
            print(f"\nEvasion rate:")
            print(f"  Start: {evasions[0]:.1%}")
            print(f"  End: {evasions[-1]:.1%}")
            print(f"  Min: {min(evasions):.1%}")
            print(f"  Max: {max(evasions):.1%}")

            # Balance zone 통계
            tracker_stats = self._evolution_tracker.get_statistics()
            print(f"\nBalance zone:")
            print(f"  Total cycles in zone: {tracker_stats['total_balance_cycles']}")
            print(f"  Entries: {tracker_stats['balance_entries']}")
            print(f"  Exits: {tracker_stats['balance_exits']}")

            # 액션 분포
            actions = [s.action for s in self._history]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            print(f"\nAction distribution:")
            for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
                print(f"  {action}: {count} ({count/len(actions)*100:.1f}%)")

        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Balanced Co-Evolution Training")
    parser.add_argument("--max-cycles", type=int, help="Max cycles (default: unlimited)")
    parser.add_argument("--target-evasion", type=float, help="Stop when avg evasion <= target")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = BalancedConfig(
        use_amp=not args.no_amp,
    )

    coevolution = BalancedCoevolution(config)
    await coevolution.run(
        max_cycles=args.max_cycles,
        target_evasion=args.target_evasion,
    )


if __name__ == "__main__":
    asyncio.run(main())
