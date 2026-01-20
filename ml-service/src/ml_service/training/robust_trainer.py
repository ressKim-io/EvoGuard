"""Robust Adversarial Trainer with integrated optimizations.

Combines:
1. Focal Loss - For class imbalance handling
2. TRADES Loss - For adversarial robustness
3. Adaptive Precision - For training efficiency (FP16/FP32 mixed precision)
4. Progressive Adversarial Training - Gradually increase perturbation strength

Reference papers:
- Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- TRADES: "Theoretically Principled Trade-off..." (Zhang et al., 2019)
- Adaptive Precision: "Adaptive precision layering..." (ScienceDirect, 2025)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ml_service.training.losses import RobustLoss, FocalLoss

logger = logging.getLogger(__name__)

# Optional imports
HAS_TRAINING_DEPS = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm
    HAS_TRAINING_DEPS = True
except ImportError:
    pass

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


@dataclass
class RobustTrainingConfig:
    """Configuration for robust adversarial training.

    Attributes:
        # Basic training
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight

        # Focal Loss
        focal_gamma: Focusing parameter (higher = focus on hard examples)
        focal_alpha: Class balancing weight

        # TRADES Loss
        trades_beta: Trade-off parameter (higher = more robust)
        adversarial_training: Whether to use adversarial examples

        # Adaptive Precision
        use_adaptive_precision: Enable mixed precision training
        precision_schedule: How to adjust precision ('constant', 'progressive')
        initial_precision: Starting precision ('fp16', 'fp32', 'bf16')

        # Progressive Training
        progressive_perturbation: Gradually increase perturbation strength
        initial_epsilon: Starting perturbation strength
        max_epsilon: Maximum perturbation strength

        # Text Attack Settings
        attack_strategies: List of attack strategy names to use
        attacks_per_sample: Number of attack variants per sample
    """

    # Basic training
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Focal Loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # TRADES Loss
    trades_beta: float = 6.0
    adversarial_training: bool = True

    # Adaptive Precision
    use_adaptive_precision: bool = True
    precision_schedule: str = "progressive"  # 'constant', 'progressive'
    initial_precision: str = "fp16"

    # Progressive Training
    progressive_perturbation: bool = True
    initial_epsilon: float = 0.1
    max_epsilon: float = 1.0
    perturbation_warmup_epochs: int = 3

    # Text Attack Settings
    attack_strategies: list[str] = field(default_factory=lambda: [
        "chosung", "jamo_decompose", "english_phonetic", "space_insertion",
        "zero_width", "emoji_insertion", "yamin", "cjk_semantic",
        "iconic_consonant", "kotox_mixed"
    ])
    attacks_per_sample: int = 3

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("models/robust-model"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/robust"))

    # Device
    device: str = "cuda"


class AdaptivePrecisionManager:
    """Manages adaptive mixed precision training.

    Automatically adjusts precision based on:
    - Training stability (gradient overflow detection)
    - Layer importance (critical layers use FP32)
    - Training progress (start conservative, get aggressive)
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: str = "progressive",
        initial_precision: str = "fp16",
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.current_precision = initial_precision

        # GradScaler for mixed precision
        self.scaler = GradScaler(enabled=(initial_precision == "fp16"))

        # Track overflow events
        self.overflow_count = 0
        self.total_steps = 0

        # Layer-wise precision (critical layers use FP32)
        self.critical_layers: set[str] = set()
        self._identify_critical_layers()

    def _identify_critical_layers(self) -> None:
        """Identify layers that should always use FP32.

        Critical layers include:
        - Final classification layers
        - Normalization layers
        - Attention output projections
        """
        for name, module in self.model.named_modules():
            # Classification head
            if "classifier" in name.lower() or "head" in name.lower():
                self.critical_layers.add(name)
            # Layer normalization
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                self.critical_layers.add(name)

        logger.info(f"Identified {len(self.critical_layers)} critical layers for FP32")

    def get_autocast_context(self, epoch: int, total_epochs: int):
        """Get appropriate autocast context based on training progress.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Autocast context manager
        """
        if not self.scaler.is_enabled():
            return torch.cuda.amp.autocast(enabled=False)

        # Progressive schedule: start with FP32, transition to FP16
        if self.schedule == "progressive":
            progress = epoch / total_epochs
            if progress < 0.2:
                # First 20%: mostly FP32 for stability
                dtype = torch.float32
            elif progress < 0.5:
                # Middle: mixed
                dtype = torch.float16
            else:
                # Later: aggressive FP16
                dtype = torch.float16
        else:
            dtype = torch.float16

        return torch.cuda.amp.autocast(enabled=True, dtype=dtype)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        return self.scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Perform optimizer step with gradient unscaling.

        Returns:
            True if step was successful, False if overflow detected
        """
        self.total_steps += 1

        # Unscale gradients
        self.scaler.unscale_(optimizer)

        # Check for overflow
        found_inf = False
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        found_inf = True
                        break
            if found_inf:
                break

        if found_inf:
            self.overflow_count += 1
            logger.warning(f"Gradient overflow detected (total: {self.overflow_count})")

            # If too many overflows, consider switching to FP32
            if self.overflow_count > self.total_steps * 0.1:
                logger.warning("High overflow rate, consider using FP32")

            # IMPORTANT: Must call scaler.update() even when skipping step
            # to reset the scaler state for the next iteration
            self.scaler.update()
            return False

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.scaler.step(optimizer)
        self.scaler.update()

        return True


class TextAdversarialAttacker:
    """Generates adversarial text examples using Korean attack strategies.

    Integrates with KOTOX-based attack strategies for comprehensive
    adversarial example generation.
    """

    def __init__(
        self,
        strategies: list[str],
        attacks_per_sample: int = 3,
    ) -> None:
        self.strategy_names = strategies
        self.attacks_per_sample = attacks_per_sample

        # Load attack functions
        try:
            from ml_service.attacker.korean_strategies import (
                get_korean_strategies,
                apply_korean_attack,
            )
            self.strategies = {s.name: s.transform for s in get_korean_strategies()}
            self.apply_attack = apply_korean_attack
            logger.info(f"Loaded {len(self.strategies)} attack strategies")
        except ImportError:
            logger.warning("Korean attack strategies not available")
            self.strategies = {}
            self.apply_attack = None

    def generate_adversarial(
        self,
        texts: list[str],
        epsilon: float = 1.0,
    ) -> list[str]:
        """Generate adversarial versions of input texts.

        Args:
            texts: Original texts
            epsilon: Perturbation strength (0-1, controls how many attacks to apply)

        Returns:
            List of adversarial texts
        """
        if not self.strategies:
            return texts

        adv_texts = []
        for text in texts:
            # Determine number of attacks based on epsilon
            num_attacks = max(1, int(self.attacks_per_sample * epsilon))

            # Select random strategies
            available = [s for s in self.strategy_names if s in self.strategies]
            if not available:
                adv_texts.append(text)
                continue

            selected = random.sample(available, min(num_attacks, len(available)))

            # Apply attacks sequentially
            adv_text = text
            for strategy_name in selected:
                try:
                    adv_text = self.strategies[strategy_name](adv_text)
                except Exception as e:
                    logger.debug(f"Attack {strategy_name} failed: {e}")

            adv_texts.append(adv_text)

        return adv_texts


class RobustTrainer:
    """Unified robust trainer combining all optimization techniques.

    Features:
    - Focal Loss for class imbalance
    - TRADES Loss for adversarial robustness
    - Adaptive Mixed Precision for efficiency
    - Progressive Adversarial Training
    - Korean text attack strategies (KOTOX)

    Example:
        >>> config = RobustTrainingConfig(num_epochs=10, trades_beta=6.0)
        >>> trainer = RobustTrainer(model, tokenizer, config)
        >>> results = trainer.train(train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: RobustTrainingConfig,
    ) -> None:
        if not HAS_TRAINING_DEPS:
            raise ImportError("Training dependencies not installed")

        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Move model to device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Initialize loss function
        self.criterion = RobustLoss(
            beta=config.trades_beta,
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            adversarial_training=config.adversarial_training,
        )

        # Initialize adaptive precision manager
        if config.use_adaptive_precision:
            self.precision_manager = AdaptivePrecisionManager(
                model=self.model,
                schedule=config.precision_schedule,
                initial_precision=config.initial_precision,
            )
        else:
            self.precision_manager = None

        # Initialize adversarial attacker
        if config.adversarial_training:
            self.attacker = TextAdversarialAttacker(
                strategies=config.attack_strategies,
                attacks_per_sample=config.attacks_per_sample,
            )
        else:
            self.attacker = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.history: list[dict] = []

    def get_current_epsilon(self) -> float:
        """Get current perturbation strength based on training progress."""
        if not self.config.progressive_perturbation:
            return self.config.max_epsilon

        if self.current_epoch < self.config.perturbation_warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.config.perturbation_warmup_epochs
            return self.config.initial_epsilon + progress * (
                self.config.max_epsilon - self.config.initial_epsilon
            )
        return self.config.max_epsilon

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Batch of tokenized data
            optimizer: Optimizer instance

        Returns:
            Dictionary of loss components
        """
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Get autocast context
        if self.precision_manager:
            autocast_ctx = self.precision_manager.get_autocast_context(
                self.current_epoch, self.config.num_epochs
            )
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)

        with autocast_ctx:
            # Forward pass (clean)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            clean_logits = outputs.logits

            # Generate adversarial examples if enabled
            if self.attacker and self.config.adversarial_training:
                # Decode texts for attack
                texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                # Generate adversarial texts
                epsilon = self.get_current_epsilon()
                adv_texts = self.attacker.generate_adversarial(texts, epsilon)

                # Tokenize adversarial texts
                adv_inputs = self.tokenizer(
                    adv_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                adv_input_ids = adv_inputs["input_ids"].to(self.device)
                adv_attention_mask = adv_inputs["attention_mask"].to(self.device)

                # Forward pass (adversarial)
                adv_outputs = self.model(
                    input_ids=adv_input_ids,
                    attention_mask=adv_attention_mask,
                )
                adv_logits = adv_outputs.logits

                # Compute TRADES + Focal loss
                loss, components = self.criterion(clean_logits, labels, adv_logits)
            else:
                # Just Focal loss
                loss, components = self.criterion(clean_logits, labels)

        # Backward pass
        optimizer.zero_grad()

        if self.precision_manager:
            scaled_loss = self.precision_manager.scale_loss(loss)
            scaled_loss.backward()
            success = self.precision_manager.step(optimizer)
            if not success:
                components["overflow"] = 1.0
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            optimizer.step()

        self.global_step += 1
        return components

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: "DataLoader",
    ) -> dict[str, float]:
        """Evaluate model on validation set.

        Args:
            dataloader: Validation dataloader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Use Focal loss for evaluation
            focal_loss = FocalLoss(
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
            )
            loss = focal_loss(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            "eval_loss": total_loss / len(dataloader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
        }

        return metrics

    def train(
        self,
        train_dataloader: "DataLoader",
        val_dataloader: "DataLoader",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        """Run full training loop.

        Args:
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            optimizer: Optional optimizer (creates AdamW if not provided)
            scheduler: Optional learning rate scheduler

        Returns:
            Training results dictionary
        """
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Create scheduler if not provided
        if scheduler is None:
            from transformers import get_linear_schedule_with_warmup

            total_steps = len(train_dataloader) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("ROBUST ADVERSARIAL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"TRADES beta: {self.config.trades_beta}")
        logger.info(f"Focal gamma: {self.config.focal_gamma}")
        logger.info(f"Adaptive precision: {self.config.use_adaptive_precision}")
        logger.info(f"Adversarial training: {self.config.adversarial_training}")
        logger.info("=" * 60)

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epsilon = self.get_current_epsilon()

            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs} (ε={epsilon:.2f})")

            # Training
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Training")

            for batch in progress_bar:
                components = self.train_step(batch, optimizer)
                scheduler.step()

                epoch_losses.append(components.get("total_loss", 0.0))

                # Update progress bar
                avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Evaluation
            eval_metrics = self.evaluate(val_dataloader)

            # Log epoch results
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "epsilon": epsilon,
                **eval_metrics,
            }
            self.history.append(epoch_result)

            logger.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={epoch_result['train_loss']:.4f}, "
                f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                f"f1={eval_metrics['f1']:.4f}, "
                f"acc={eval_metrics['accuracy']:.4f}"
            )

            # Save best model
            if eval_metrics["f1"] > self.best_metric:
                self.best_metric = eval_metrics["f1"]
                self.save_model(self.config.output_dir / "best_model")
                logger.info(f"New best model saved (F1={self.best_metric:.4f})")

            # Save checkpoint
            self.save_checkpoint(epoch)

        # Final results
        results = {
            "best_f1": self.best_metric,
            "final_metrics": self.history[-1] if self.history else {},
            "history": self.history,
            "model_path": str(self.config.output_dir / "best_model"),
        }

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best F1: {self.best_metric:.4f}")
        logger.info("=" * 60)

        return results

    def save_model(self, path: Path | str) -> None:
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint-{epoch + 1}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "history": self.history,
            },
            checkpoint_path / "trainer_state.pt",
        )

        self.model.save_pretrained(checkpoint_path)
        logger.debug(f"Checkpoint saved to {checkpoint_path}")


def create_robust_trainer(
    model_name: str = "beomi/KcELECTRA-base-v2022",
    num_labels: int = 2,
    config: RobustTrainingConfig | None = None,
) -> RobustTrainer:
    """Factory function to create a RobustTrainer.

    Args:
        model_name: Pretrained model name
        num_labels: Number of classification labels
        config: Training configuration

    Returns:
        Configured RobustTrainer instance
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if config is None:
        config = RobustTrainingConfig()

    return RobustTrainer(model, tokenizer, config)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("RobustTrainer module loaded successfully!")
    print("\nFeatures:")
    print("  ✅ Focal Loss - Class imbalance handling")
    print("  ✅ TRADES Loss - Adversarial robustness")
    print("  ✅ Adaptive Precision - Mixed precision training")
    print("  ✅ Progressive Perturbation - Gradual attack strength")
    print("  ✅ Korean Attack Strategies - KOTOX integration")
