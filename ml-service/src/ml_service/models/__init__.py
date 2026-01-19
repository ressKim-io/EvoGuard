"""Models module for ml-service."""

from ml_service.models.classifier import BaseClassifier, ClassifierResult, MockClassifier

__all__ = ["BaseClassifier", "BertClassifier", "ClassifierResult", "MockClassifier"]

# BertClassifier requires PyTorch (optional dependency)
try:
    from ml_service.models.bert_classifier import BertClassifier
except ImportError:
    BertClassifier = None  # type: ignore[assignment, misc]
