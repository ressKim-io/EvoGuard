"""Classifier adapter for pipeline integration.

Provides a wrapper to use trained models with the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainedClassifierAdapter:
    """Adapter to use trained QLoRA model with the pipeline.

    Wraps the QLoRATrainer to provide a simple predict interface
    compatible with the pipeline's ClassifierProtocol.

    Example:
        >>> adapter = TrainedClassifierAdapter("models/toxic-classifier")
        >>> results = adapter.predict(["toxic text", "normal text"])
    """

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the classifier adapter.

        Args:
            model_path: Path to the trained model directory.
        """
        self.model_path = Path(model_path)
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "Training dependencies required. Install with: uv pip install --group training"
            ) from e

        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load adapter config to get base model name
        import json
        config_path = self.model_path / "adapter_config.json"
        with open(config_path) as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path", "bert-base-uncased")

        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
        )

        # Load PEFT model
        self._model = PeftModel.from_pretrained(base_model, self.model_path)
        self._model.eval()

        # Move to GPU if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        logger.info(f"Model loaded on {self._device}")

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict labels for texts.

        Args:
            texts: List of texts to classify.

        Returns:
            List of dicts with keys: label, label_name, confidence
        """
        import torch

        if self._model is None:
            self.load()

        # Tokenize
        inputs = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for prob in probs:
            label = prob.argmax().item()
            confidence = prob[label].item()
            results.append({
                "label": label,
                "label_name": "toxic" if label == 1 else "non-toxic",
                "confidence": round(confidence, 4),
            })

        return results


def create_trained_classifier(model_path: str | Path | None = None) -> TrainedClassifierAdapter:
    """Create a trained classifier adapter.

    Args:
        model_path: Path to model. Defaults to models/toxic-classifier.

    Returns:
        Loaded classifier adapter.
    """
    if model_path is None:
        model_path = Path("models/toxic-classifier")

    adapter = TrainedClassifierAdapter(model_path)
    adapter.load()
    return adapter
