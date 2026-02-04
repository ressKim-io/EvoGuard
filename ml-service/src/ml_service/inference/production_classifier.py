"""Production classifier using the deployed coevolution model.

This is the recommended classifier for production use.
Model: Coevolution-Latest (2026-02-04)
F1: 0.9621, FP: 182, FN: 51

Usage:
    from ml_service.inference.production_classifier import ProductionClassifier

    classifier = ProductionClassifier()
    result = classifier.predict("텍스트")
    # result: {'label': 1, 'confidence': 0.95, 'toxic': True}
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Union, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ProductionClassifier:
    """Production toxic text classifier using coevolution model."""

    DEFAULT_MODEL_PATH = "models/production"

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        threshold: float = 0.5,
    ):
        """Initialize production classifier.

        Args:
            model_path: Path to model directory (default: models/production)
            device: Device to use ('cuda', 'cpu', or None for auto)
            threshold: Classification threshold (default: 0.5)
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.threshold = threshold

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        self._load_model()

        # Load deployment info
        self._load_deployment_info()

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading production model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def _load_deployment_info(self):
        """Load deployment metadata."""
        info_path = Path(self.model_path) / "DEPLOYMENT_INFO.json"
        if info_path.exists():
            with open(info_path) as f:
                self.deployment_info = json.load(f)
            print(f"Model version: {self.deployment_info.get('version', 'unknown')}")
        else:
            self.deployment_info = {}

    def predict(
        self,
        text: Union[str, List[str]],
        return_probs: bool = False,
    ) -> Union[Dict, List[Dict]]:
        """Predict toxicity of text.

        Args:
            text: Single text or list of texts
            return_probs: Whether to return class probabilities

        Returns:
            Dict or list of dicts with keys:
                - label: 0 (normal) or 1 (toxic)
                - confidence: Confidence score (0-1)
                - toxic: Boolean indicating toxicity
                - probs: (optional) Class probabilities [normal, toxic]
        """
        single_input = isinstance(text, str)
        if single_input:
            text = [text]

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        # Process results
        results = []
        for i in range(len(text)):
            prob = probs[i].cpu().numpy()
            toxic_prob = prob[1]
            label = 1 if toxic_prob >= self.threshold else 0

            result = {
                "label": label,
                "confidence": float(max(prob)),
                "toxic": bool(label == 1),
            }

            if return_probs:
                result["probs"] = prob.tolist()

            results.append(result)

        return results[0] if single_input else results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> List[Dict]:
        """Predict toxicity for a batch of texts.

        Args:
            texts: List of texts
            batch_size: Batch size for inference
            return_probs: Whether to return class probabilities

        Returns:
            List of prediction dicts
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_probs=return_probs)
            results.extend(batch_results)
        return results

    def get_model_info(self) -> Dict:
        """Get model deployment information."""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "threshold": self.threshold,
            **self.deployment_info,
        }


# Convenience function
def create_classifier(device: str = None) -> ProductionClassifier:
    """Create production classifier with default settings."""
    return ProductionClassifier(device=device)


if __name__ == "__main__":
    # Quick test
    classifier = ProductionClassifier()

    test_texts = [
        "안녕하세요, 좋은 하루 되세요!",
        "이 XX야 꺼져",
        "오늘 날씨가 좋네요",
    ]

    print("\n=== Production Classifier Test ===")
    for text in test_texts:
        result = classifier.predict(text, return_probs=True)
        status = "🚫 TOXIC" if result["toxic"] else "✅ NORMAL"
        print(f"{status} ({result['confidence']:.2%}): {text[:30]}...")

    print("\n=== Model Info ===")
    info = classifier.get_model_info()
    print(f"Version: {info.get('version', 'N/A')}")
    print(f"F1: {info.get('metrics', {}).get('f1_weighted', 'N/A')}")
