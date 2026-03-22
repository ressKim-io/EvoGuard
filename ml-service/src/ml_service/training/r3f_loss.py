"""R3F (Regularized Finetuning with Random perturbations) Loss.

Adds Gaussian noise to token embeddings and minimizes KL divergence
between clean and noisy predictions, improving model robustness.

Reference:
    "Better Fine-Tuning by Reducing Representational Collapse"
    (Aghajanyan et al., 2020) https://arxiv.org/abs/2008.03156

Example:
    >>> r3f = R3FLoss(noise_std=1e-5, r3f_lambda=1.0)
    >>> total_loss = r3f(model, input_ids, attention_mask, labels, focal_loss_fn)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    pass

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class R3FLoss:
    """R3F regularization loss.

    Computes: task_loss + lambda * KL(clean_logits || noisy_logits)

    Where noise is added to the token embedding layer of the model.

    Args:
        noise_std: Standard deviation of Gaussian noise added to embeddings.
        r3f_lambda: Weight of the KL divergence regularization term.
    """

    def __init__(
        self,
        noise_std: float = 1e-5,
        r3f_lambda: float = 1.0,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for R3FLoss.")
        self.noise_std = noise_std
        self.r3f_lambda = r3f_lambda

    def _get_embedding_layer(self, model: nn.Module) -> nn.Module:
        """Extract the token embedding layer from a transformer model.

        Supports HuggingFace AutoModelForSequenceClassification wrappers
        for ELECTRA, BERT, DeBERTa, etc.
        """
        # Try common model attribute names
        for attr in ("electra", "bert", "deberta", "roberta", "distilbert", "base_model"):
            base = getattr(model, attr, None)
            if base is not None:
                embeddings = getattr(base, "embeddings", None)
                if embeddings is not None:
                    word_emb = getattr(embeddings, "word_embeddings", None)
                    if word_emb is not None:
                        return word_emb

        # Fallback: search for Embedding layers
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                return module

        raise AttributeError(
            "Could not find token embedding layer. "
            "Ensure the model has a standard HuggingFace architecture."
        )

    def __call__(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        task_loss_fn,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute R3F-regularized loss.

        Args:
            model: The transformer model.
            input_ids: Token IDs (N, L).
            attention_mask: Attention mask (N, L).
            labels: Ground truth labels (N,).
            task_loss_fn: Primary loss function (e.g., FocalLoss) that accepts
                (logits, labels) and returns a scalar loss.

        Returns:
            Tuple of (total_loss, components_dict).
        """
        embedding_layer = self._get_embedding_layer(model)

        # --- Clean forward pass ---
        clean_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        clean_logits = clean_outputs.logits
        task_loss = task_loss_fn(clean_logits, labels)

        # --- Noisy forward pass ---
        # Hook to add noise to embeddings
        noise_holder = {}

        def _add_noise_hook(module, input, output):
            noise = torch.randn_like(output) * self.noise_std
            noise_holder["noise"] = noise
            return output + noise

        handle = embedding_layer.register_forward_hook(_add_noise_hook)
        try:
            noisy_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        finally:
            handle.remove()

        noisy_logits = noisy_outputs.logits

        # --- KL divergence between clean and noisy ---
        clean_log_probs = F.log_softmax(clean_logits, dim=-1)
        noisy_probs = F.softmax(noisy_logits, dim=-1)

        # KL(noisy || clean) — measures how noisy differs from clean
        kl_loss = F.kl_div(
            clean_log_probs,
            noisy_probs,
            reduction="batchmean",
        )

        total_loss = task_loss + self.r3f_lambda * kl_loss

        components = {
            "task_loss": task_loss.item(),
            "kl_loss": kl_loss.item(),
            "r3f_total_loss": total_loss.item(),
        }

        return total_loss, components
