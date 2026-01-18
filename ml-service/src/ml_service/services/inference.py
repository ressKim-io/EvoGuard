"""Inference service for text classification."""

from ml_service.core.config import get_settings
from ml_service.core.logging import get_logger
from ml_service.models.classifier import BaseClassifier, ClassifierResult, MockClassifier

logger = get_logger(__name__)


class InferenceService:
    """Service for managing model inference."""

    def __init__(self) -> None:
        """Initialize the inference service."""
        self._classifier: BaseClassifier | None = None
        self._settings = get_settings()

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._classifier is not None

    @property
    def model_version(self) -> str:
        """Get the current model version."""
        if self._classifier is None:
            return "not-loaded"
        return self._classifier.version

    def load_model(self) -> None:
        """Load the classification model."""
        model_name = self._settings.model_name

        logger.info("loading_model", model_name=model_name)

        if model_name == "mock":
            self._classifier = MockClassifier(threshold=self._settings.confidence_threshold)
        else:
            # Future: Load real model from MLflow or local path
            logger.warning("unknown_model_falling_back_to_mock", model_name=model_name)
            self._classifier = MockClassifier(threshold=self._settings.confidence_threshold)

        logger.info("model_loaded", version=self.model_version)

    def unload_model(self) -> None:
        """Unload the current model."""
        self._classifier = None
        logger.info("model_unloaded")

    def classify(self, text: str) -> ClassifierResult:
        """Classify a single text.

        Args:
            text: The text to classify.

        Returns:
            ClassifierResult with classification details.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._classifier is None:
            raise RuntimeError("Model not loaded")

        result = self._classifier.predict(text)

        logger.debug(
            "classification_complete",
            text_length=len(text),
            is_toxic=result.is_toxic,
            confidence=result.confidence,
        )

        return result

    def classify_batch(self, texts: list[str]) -> list[ClassifierResult]:
        """Classify multiple texts.

        Args:
            texts: List of texts to classify.

        Returns:
            List of ClassifierResult objects.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._classifier is None:
            raise RuntimeError("Model not loaded")

        results = self._classifier.predict_batch(texts)

        logger.debug(
            "batch_classification_complete",
            batch_size=len(texts),
            toxic_count=sum(1 for r in results if r.is_toxic),
        )

        return results


# Singleton instance
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Get the singleton inference service instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
