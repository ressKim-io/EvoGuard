"""Advanced loss functions for robust adversarial training.

Implements:
1. Focal Loss - For handling class imbalance
2. TRADES Loss - For adversarial robustness with clean accuracy balance
3. Combined Focal-TRADES Loss - Best of both worlds
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Optional imports
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    # Create dummy classes for type checking
    class nn:  # type: ignore
        class Module:
            pass

if TYPE_CHECKING:
    import torch


def _check_torch():
    """Check if PyTorch is available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for loss functions. "
            "Install with: pip install torch"
        )


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal Loss down-weights well-classified examples and focuses on hard,
    misclassified examples. Particularly useful for imbalanced datasets.

    Loss = -α * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for class imbalance. Can be:
            - float: Applied to positive class
            - Tensor: Per-class weights
            - None: No weighting
        gamma: Focusing parameter. Higher values focus more on hard examples.
            - gamma=0: Equivalent to Cross Entropy
            - gamma=2: Recommended default
        reduction: 'mean', 'sum', or 'none'

    Reference:
        "Focal Loss for Dense Object Detection" (Lin et al., 2017)
        https://arxiv.org/abs/1708.02002

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        _check_torch()
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits of shape (N, C) where C is num_classes
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probability of correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Binary case: alpha for positive, (1-alpha) for negative
                alpha_t = torch.where(
                    targets == 1,
                    torch.tensor(self.alpha, device=inputs.device),
                    torch.tensor(1 - self.alpha, device=inputs.device),
                )
            else:
                # Multi-class case: per-class weights
                alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TRADESLoss(nn.Module):
    """TRADES Loss for adversarial robustness.

    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    balances clean accuracy and adversarial robustness through a regularization term.

    Loss = CE(f(x), y) + β * KL(f(x) || f(x + δ))

    Where:
        - CE: Cross entropy on clean examples
        - KL: KL divergence between clean and adversarial predictions
        - β: Trade-off parameter (higher = more robust, lower = better clean accuracy)
        - δ: Adversarial perturbation

    Args:
        beta: Trade-off parameter between clean and robust loss.
            - β=1.0: Equal weight
            - β=6.0: Recommended for robustness
        use_focal: Use Focal Loss instead of CE for clean loss
        focal_gamma: Gamma parameter for Focal Loss
        focal_alpha: Alpha parameter for Focal Loss

    Reference:
        "Theoretically Principled Trade-off between Robustness and Accuracy"
        (Zhang et al., 2019) https://arxiv.org/abs/1901.08573

    Example:
        >>> criterion = TRADESLoss(beta=6.0, use_focal=True)
        >>> loss = criterion(clean_logits, adv_logits, targets)
    """

    def __init__(
        self,
        beta: float = 6.0,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.25,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.use_focal = use_focal

        if use_focal:
            self.clean_loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction="mean",
            )
        else:
            self.clean_loss_fn = nn.CrossEntropyLoss()

        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        clean_logits: torch.Tensor,
        adv_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute TRADES loss.

        Args:
            clean_logits: Predictions on clean examples (N, C)
            adv_logits: Predictions on adversarial examples (N, C)
            targets: Ground truth labels (N,)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Clean loss (CE or Focal)
        clean_loss = self.clean_loss_fn(clean_logits, targets)

        # KL divergence loss for robustness
        clean_probs = F.softmax(clean_logits, dim=-1)
        adv_log_probs = F.log_softmax(adv_logits, dim=-1)
        kl_loss = self.kl_loss_fn(adv_log_probs, clean_probs)

        # Combined TRADES loss
        total_loss = clean_loss + self.beta * kl_loss

        # Return components for logging
        components = {
            "clean_loss": clean_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, components


class RobustLoss(nn.Module):
    """Unified robust loss combining Focal Loss and TRADES.

    This is the main loss function for robust adversarial training,
    combining the benefits of:
    - Focal Loss: Better handling of class imbalance
    - TRADES: Better adversarial robustness

    Args:
        beta: TRADES trade-off parameter
        gamma: Focal loss focusing parameter
        alpha: Focal loss class weighting
        adversarial_training: Whether to use adversarial training

    Example:
        >>> criterion = RobustLoss(beta=6.0, gamma=2.0)
        >>> # For clean training
        >>> loss = criterion(logits, targets=labels)
        >>> # For adversarial training
        >>> loss = criterion(clean_logits, targets=labels, adv_logits=adv_logits)
    """

    def __init__(
        self,
        beta: float = 6.0,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        adversarial_training: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.adversarial_training = adversarial_training

        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

        if adversarial_training:
            self.trades_loss = TRADESLoss(
                beta=beta,
                use_focal=True,
                focal_gamma=gamma,
                focal_alpha=alpha,
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        adv_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute robust loss.

        Args:
            logits: Clean predictions (N, C)
            targets: Ground truth labels (N,)
            adv_logits: Optional adversarial predictions (N, C)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if adv_logits is not None and self.adversarial_training:
            # Full TRADES + Focal loss
            return self.trades_loss(logits, adv_logits, targets)
        else:
            # Just Focal loss
            loss = self.focal_loss(logits, targets)
            return loss, {"focal_loss": loss.item(), "total_loss": loss.item()}


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization.

    Instead of hard labels (0, 1), uses soft labels:
    - Target class: 1 - smoothing
    - Other classes: smoothing / (num_classes - 1)

    Args:
        smoothing: Label smoothing factor (0.0 to 1.0)
        num_classes: Number of classes
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 2) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label smoothing loss."""
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


def get_loss_function(
    loss_type: str = "robust",
    **kwargs,
) -> nn.Module:
    """Factory function to get loss function by name.

    Args:
        loss_type: One of 'focal', 'trades', 'robust', 'label_smoothing', 'ce'
        **kwargs: Arguments passed to the loss function

    Returns:
        Loss function module

    Example:
        >>> loss_fn = get_loss_function('robust', beta=6.0, gamma=2.0)
    """
    loss_functions = {
        "focal": FocalLoss,
        "trades": TRADESLoss,
        "robust": RobustLoss,
        "label_smoothing": LabelSmoothingLoss,
        "ce": nn.CrossEntropyLoss,
    }

    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}")

    return loss_functions[loss_type](**kwargs)


if __name__ == "__main__":
    # Quick test
    if HAS_TORCH:
        print("Testing loss functions...")

        # Create dummy data
        batch_size = 8
        num_classes = 2
        logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Test Focal Loss
        focal = FocalLoss(gamma=2.0, alpha=0.25)
        focal_loss = focal(logits, targets)
        print(f"Focal Loss: {focal_loss.item():.4f}")

        # Test TRADES Loss
        trades = TRADESLoss(beta=6.0, use_focal=True)
        trades_loss, components = trades(logits, adv_logits, targets)
        print(f"TRADES Loss: {trades_loss.item():.4f}")
        print(f"  Components: {components}")

        # Test Robust Loss
        robust = RobustLoss(beta=6.0, gamma=2.0)
        robust_loss, components = robust(logits, targets, adv_logits)
        print(f"Robust Loss: {robust_loss.item():.4f}")
        print(f"  Components: {components}")

        print("\n✅ All loss functions working!")
    else:
        print("PyTorch not available")
