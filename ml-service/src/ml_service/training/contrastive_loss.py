"""Supervised Contrastive Learning Loss.

Pulls same-class embeddings closer and pushes different-class
embeddings apart, improving decision boundaries.

Reference:
    "Supervised Contrastive Learning" (Khosla et al., 2020)
    https://arxiv.org/abs/2004.11362
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Args:
        temperature: Scaling factor for similarity scores.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: (N, D) L2-normalized feature vectors.
            labels: (N,) class labels.
        """
        features = F.normalize(features, dim=1)
        n = features.size(0)
        if n < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask: same-class pairs (excluding self)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)

        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Exclude self-similarity
        self_mask = torch.eye(n, device=features.device, dtype=torch.bool)
        exp_logits = torch.exp(logits) * (~self_mask).float()

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        pos_per_sample = mask.sum(dim=1).clamp(min=1)
        loss = -(mask.float() * log_prob).sum(dim=1) / pos_per_sample

        return loss.mean()
