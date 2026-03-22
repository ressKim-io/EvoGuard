"""Label-Aware Hard Negative (LAHN) Contrastive Loss.

Momentum contrastive learning with label-aware hard negative sampling
for improved hate speech detection, especially implicit/subtle cases.

Key components:
    1. Momentum Encoder (EMA) - stable representation target
    2. Negative Embedding Queue - large pool of cached embeddings
    3. Label-Aware Hard Negative Selection - hardest opposite-label samples
    4. InfoNCE Loss with hard negatives

Reference:
    "LAHN: Label-aware Hard Negative Sampling for Hate Speech Detection"
    ACL 2024 Findings (Hanyang University)
    https://aclanthology.org/2024.findings-acl.957/

Usage:
    from ml_service.training.lahn_loss import LAHNLoss

    lahn = LAHNLoss(embedding_dim=768)
    loss = lahn(embeddings, labels, momentum_embeddings)
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LAHNLoss(nn.Module):
    """Label-Aware Hard Negative contrastive loss.

    Uses a momentum-updated embedding queue and selects hardest negatives
    from opposite-label samples for each anchor.

    Args:
        embedding_dim: Dimension of input embeddings.
        queue_size: Size of the negative embedding queue.
        momentum: EMA momentum for momentum encoder update.
        temperature: Softmax temperature for contrastive loss.
        hard_negative_k: Number of hard negatives to select per anchor.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.07,
        hard_negative_k: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.hard_negative_k = hard_negative_k

        # Projection head: project embeddings to contrastive space
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 128),
        )

        # Momentum projection head (no gradient)
        self.momentum_projector = copy.deepcopy(self.projector)
        for param in self.momentum_projector.parameters():
            param.requires_grad = False

        # Queue: stores (queue_size, 128) embeddings and labels
        self.register_buffer("queue_embeddings", torch.randn(queue_size, 128))
        self.register_buffer("queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize initial queue
        self.queue_embeddings = F.normalize(self.queue_embeddings, dim=1)

    @torch.no_grad()
    def update_momentum_encoder(self, encoder: nn.Module, momentum_encoder: nn.Module) -> None:
        """Update momentum encoder with EMA of main encoder.

        Args:
            encoder: Main encoder (trained with gradients).
            momentum_encoder: Momentum encoder (updated with EMA).
        """
        for param, mom_param in zip(encoder.parameters(), momentum_encoder.parameters()):
            mom_param.data = self.momentum * mom_param.data + (1.0 - self.momentum) * param.data

    @torch.no_grad()
    def update_momentum_projector(self) -> None:
        """Update momentum projector with EMA of main projector."""
        for param, mom_param in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            mom_param.data = self.momentum * mom_param.data + (1.0 - self.momentum) * param.data

    @torch.no_grad()
    def _enqueue_dequeue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Add new embeddings to queue, remove oldest.

        Args:
            embeddings: (N, 128) L2-normalized projected embeddings.
            labels: (N,) labels.
        """
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue_embeddings[ptr:ptr + batch_size] = embeddings
            self.queue_labels[ptr:ptr + batch_size] = labels
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue_embeddings[ptr:] = embeddings[:remaining]
            self.queue_labels[ptr:] = labels[:remaining]
            overflow = batch_size - remaining
            if overflow > 0:
                self.queue_embeddings[:overflow] = embeddings[remaining:]
                self.queue_labels[:overflow] = labels[remaining:]

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def _select_hard_negatives(
        self,
        query: torch.Tensor,
        query_label: int,
        queue_embeddings: torch.Tensor,
        queue_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Select hardest negatives from queue for a query.

        Hard negatives = opposite-label samples closest in embedding space.

        Args:
            query: (128,) query embedding.
            query_label: Label of the query.
            queue_embeddings: (Q, 128) queue embeddings.
            queue_labels: (Q,) queue labels.

        Returns:
            (K, 128) hard negative embeddings.
        """
        # Find opposite-label samples in queue
        opposite_mask = (queue_labels != query_label) & (queue_labels >= 0)

        if opposite_mask.sum() == 0:
            # No opposite-label samples available
            return queue_embeddings[:min(self.hard_negative_k, len(queue_embeddings))]

        opposite_embeddings = queue_embeddings[opposite_mask]

        # Calculate cosine similarity (higher = harder negative)
        similarities = torch.matmul(opposite_embeddings, query)

        # Select top-k hardest (most similar to query despite different label)
        k = min(self.hard_negative_k, len(opposite_embeddings))
        _, top_indices = similarities.topk(k)

        return opposite_embeddings[top_indices]

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        momentum_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute LAHN contrastive loss.

        Args:
            embeddings: (N, D) embeddings from main encoder.
            labels: (N,) class labels.
            momentum_embeddings: (N, D) embeddings from momentum encoder.
                If None, uses main embeddings (less stable but works).

        Returns:
            Scalar loss value.
        """
        n = embeddings.size(0)
        if n < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Project embeddings
        z = F.normalize(self.projector(embeddings), dim=1)  # (N, 128)

        # Momentum projections (no gradient)
        with torch.no_grad():
            if momentum_embeddings is not None:
                z_mom = F.normalize(self.momentum_projector(momentum_embeddings), dim=1)
            else:
                z_mom = z.detach()

        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        valid_count = 0

        for i in range(n):
            query = z[i]  # (128,)
            query_label = labels[i].item()

            # Positive: same-label samples in batch
            pos_mask = (labels == query_label)
            pos_mask[i] = False  # exclude self

            if pos_mask.sum() == 0:
                continue  # No positives available

            # Select hard negatives from queue
            hard_negatives = self._select_hard_negatives(
                query, query_label,
                self.queue_embeddings.detach(),
                self.queue_labels.detach(),
            )

            # Also include in-batch negatives
            neg_mask = labels != query_label
            batch_negatives = z[neg_mask] if neg_mask.sum() > 0 else torch.empty(0, 128, device=z.device)

            # Combine all negatives
            if batch_negatives.numel() > 0:
                all_negatives = torch.cat([batch_negatives, hard_negatives], dim=0)
            else:
                all_negatives = hard_negatives

            # Positive similarities
            pos_embeddings = z[pos_mask]
            pos_sim = torch.matmul(pos_embeddings, query) / self.temperature  # (P,)

            # Negative similarities
            neg_sim = torch.matmul(all_negatives, query) / self.temperature  # (K,)

            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # Average over all positive pairs
            for j in range(len(pos_sim)):
                numerator = pos_sim[j]
                denominator = torch.cat([pos_sim[j:j+1], neg_sim])
                log_prob = numerator - torch.logsumexp(denominator, dim=0)
                total_loss = total_loss + (-log_prob)
                valid_count += 1

        # Update queue with momentum embeddings
        with torch.no_grad():
            self._enqueue_dequeue(z_mom.detach(), labels.detach())
            self.update_momentum_projector()

        if valid_count > 0:
            return total_loss / valid_count
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class CombinedLAHNLoss(nn.Module):
    """Combined classification + LAHN contrastive loss.

    Loss = alpha * classification_loss + (1 - alpha) * lahn_loss

    Args:
        embedding_dim: Dimension of model hidden states.
        alpha: Weight for classification loss (1-alpha for contrastive).
        classification_loss: Classification loss function.
        queue_size: LAHN queue size.
        temperature: Contrastive temperature.
        hard_negative_k: Number of hard negatives.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        alpha: float = 0.7,
        classification_loss: nn.Module | None = None,
        queue_size: int = 4096,
        temperature: float = 0.07,
        hard_negative_k: int = 256,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        if classification_loss is None:
            from ml_service.training.losses import FocalLoss
            self.cls_loss = FocalLoss(gamma=2.0, alpha=0.25)
        else:
            self.cls_loss = classification_loss

        self.lahn_loss = LAHNLoss(
            embedding_dim=embedding_dim,
            queue_size=queue_size,
            temperature=temperature,
            hard_negative_k=hard_negative_k,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        momentum_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            logits: Classification logits (N, C).
            labels: Ground truth labels (N,).
            hidden_states: Encoder hidden states for contrastive loss (N, D).
            momentum_hidden: Momentum encoder hidden states (optional).

        Returns:
            Tuple of (total_loss, components_dict).
        """
        cls_loss = self.cls_loss(logits, labels)
        contrastive_loss = self.lahn_loss(hidden_states, labels, momentum_hidden)

        total_loss = self.alpha * cls_loss + (1.0 - self.alpha) * contrastive_loss

        components = {
            "cls_loss": cls_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, components
