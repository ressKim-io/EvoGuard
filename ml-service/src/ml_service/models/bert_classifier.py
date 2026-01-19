"""BERT-based classifier for production inference."""

import logging
from pathlib import Path
from typing import Any

from ml_service.models.classifier import BaseClassifier, ClassifierResult

logger = logging.getLogger(__name__)

# Optional imports for BERT inference
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]


class BertClassifier(BaseClassifier):
    """BERT-based text classifier for toxic content detection.

    Can load models from:
    - HuggingFace Hub (e.g., 'unitary/toxic-bert')
    - Local path (fine-tuned model)
    - MLflow model registry

    Example:
        >>> classifier = BertClassifier.from_pretrained("unitary/toxic-bert")
        >>> result = classifier.predict("This is a test")
        >>> print(result.is_toxic, result.confidence)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        version: str = "bert-v1.0.0",
        threshold: float = 0.5,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        """Initialize the BERT classifier.

        Args:
            model: HuggingFace model for sequence classification.
            tokenizer: HuggingFace tokenizer.
            version: Model version string.
            threshold: Confidence threshold for classification.
            max_length: Maximum sequence length.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch and transformers not installed. "
                "Install with: uv pip install --group training"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._version = version
        self._threshold = threshold
        self._max_length = max_length

        # Set device
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model.to(self._device)
        self._model.eval()

        logger.info(f"BertClassifier initialized on {self._device}")

    @property
    def version(self) -> str:
        """Return the model version."""
        return self._version

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        version: str | None = None,
        threshold: float = 0.5,
        max_length: int = 256,
        device: str | None = None,
    ) -> "BertClassifier":
        """Load a pre-trained model.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            version: Optional version string.
            threshold: Confidence threshold.
            max_length: Maximum sequence length.
            device: Device to use.

        Returns:
            Initialized BertClassifier.
        """
        logger.info(f"Loading model: {model_name_or_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            device_map="auto" if device is None else None,
        )

        # Determine version
        if version is None:
            if isinstance(model_name_or_path, Path):
                version = f"local-{model_name_or_path.name}"
            else:
                version = f"hf-{model_name_or_path.replace('/', '-')}"

        return cls(
            model=model,
            tokenizer=tokenizer,
            version=version,
            threshold=threshold,
            max_length=max_length,
            device=device,
        )

    @classmethod
    def from_mlflow(
        cls,
        model_uri: str,
        threshold: float = 0.5,
        max_length: int = 256,
        device: str | None = None,
    ) -> "BertClassifier":
        """Load a model from MLflow.

        Args:
            model_uri: MLflow model URI (e.g., 'models:/toxic-classifier@champion').
            threshold: Confidence threshold.
            max_length: Maximum sequence length.
            device: Device to use.

        Returns:
            Initialized BertClassifier.
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError("MLflow not installed") from e

        logger.info(f"Loading model from MLflow: {model_uri}")

        # Load model from MLflow
        model = mlflow.transformers.load_model(model_uri)

        # Extract version from URI
        version = model_uri.split("/")[-1].replace("@", "-")

        return cls(
            model=model.model,
            tokenizer=model.tokenizer,
            version=f"mlflow-{version}",
            threshold=threshold,
            max_length=max_length,
            device=device,
        )

    def predict(self, text: str) -> ClassifierResult:
        """Classify a single text.

        Args:
            text: The text to classify.

        Returns:
            ClassifierResult with is_toxic, confidence, and label.
        """
        # Tokenize
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Get prediction (assuming label 1 = toxic)
        prob_toxic = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()

        is_toxic = prob_toxic >= self._threshold

        return ClassifierResult(
            is_toxic=is_toxic,
            confidence=round(prob_toxic if is_toxic else 1 - prob_toxic, 4),
            label="toxic" if is_toxic else "non-toxic",
        )

    def predict_batch(self, texts: list[str]) -> list[ClassifierResult]:
        """Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify.

        Returns:
            List of ClassifierResult objects.
        """
        if not texts:
            return []

        # Tokenize all texts
        inputs = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for prob in probs:
            prob_toxic = prob[1].item() if len(prob) > 1 else prob[0].item()
            is_toxic = prob_toxic >= self._threshold

            results.append(
                ClassifierResult(
                    is_toxic=is_toxic,
                    confidence=round(prob_toxic if is_toxic else 1 - prob_toxic, 4),
                    label="toxic" if is_toxic else "non-toxic",
                )
            )

        return results

    def get_probabilities(self, text: str) -> dict[str, float]:
        """Get class probabilities for a text.

        Args:
            text: The text to classify.

        Returns:
            Dictionary mapping class names to probabilities.
        """
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        return {
            "non-toxic": round(probs[0].item(), 4),
            "toxic": round(probs[1].item(), 4) if len(probs) > 1 else round(1 - probs[0].item(), 4),
        }
