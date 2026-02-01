"""Ensemble classifier combining Phase 2 and Coevolution models.

Best configuration: Phase2 + Coevolution with AND strategy.
F1: 0.9696, FP: 60, FN: 168

AND strategy: Both models must predict toxic for final toxic prediction.
This reduces false positives while maintaining high detection accuracy.

Supports multiple ensemble strategies:
- 'and': Both models must predict toxic (low FP) - DEFAULT
- 'or': Either model predicts toxic (low FN)
- 'weighted': Weighted average of probabilities
- 'pmf': PMF (Parallel Model Fusion) with 3 models and meta-learner
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Union, Tuple, Literal, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EnsembleClassifier:
    """Ensemble toxic text classifier using Phase2 + Coevolution models."""

    def __init__(
        self,
        model1_path: str = "models/phase2-combined/best_model",
        model2_path: str = "models/coevolution-latest",
        strategy: Literal["and", "or", "weighted"] = "and",
        weight1: float = 0.6,
        weight2: float = 0.4,
        threshold: float = 0.5,
        device: str = None,
    ):
        """Initialize ensemble classifier.

        Args:
            model1_path: Path to Phase 2 model (balanced)
            model2_path: Path to Coevolution model (attack-resistant)
            strategy: Ensemble strategy ('and', 'or', 'weighted')
                - 'and': Both models must predict toxic (low FP)
                - 'or': Either model predicts toxic (low FN)
                - 'weighted': Weighted average of probabilities
            weight1: Weight for model 1 (only used with 'weighted' strategy)
            weight2: Weight for model 2 (only used with 'weighted' strategy)
            threshold: Classification threshold (default: 0.5)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.strategy = strategy
        self.weight1 = weight1
        self.weight2 = weight2
        self.threshold = threshold

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load models
        print(f"Loading Phase 2 model from {model1_path}...")
        self.tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
        self.model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)
        self.model1.to(self.device)
        self.model1.eval()

        print(f"Loading Coevolution model from {model2_path}...")
        self.tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
        self.model2 = AutoModelForSequenceClassification.from_pretrained(model2_path)
        self.model2.to(self.device)
        self.model2.eval()

        print(f"Ensemble ready on {self.device}")
        print(f"Strategy: {strategy.upper()}, Threshold={threshold}")
        if strategy == "weighted":
            print(f"Weights: Phase2={weight1}, Coevo={weight2}")

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False,
        max_length: int = 256,
    ) -> Union[Dict, List[Dict]]:
        """Predict toxicity for given texts.

        Args:
            texts: Single text or list of texts
            return_probs: Whether to return individual model probabilities
            max_length: Maximum sequence length

        Returns:
            Dict or list of dicts with keys:
                - label: 0 (clean) or 1 (toxic)
                - confidence: Ensemble confidence score
                - toxic_prob: Probability of being toxic
                - prob1, prob2: Individual model probabilities (if return_probs=True)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        results = []

        with torch.no_grad():
            # Model 1 predictions
            enc1 = self.tokenizer1(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc1 = {k: v.to(self.device) for k, v in enc1.items()}
            out1 = self.model1(**enc1)
            probs1 = F.softmax(out1.logits, dim=-1)[:, 1].cpu().numpy()

            # Model 2 predictions
            enc2 = self.tokenizer2(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc2 = {k: v.to(self.device) for k, v in enc2.items()}
            out2 = self.model2(**enc2)
            probs2 = F.softmax(out2.logits, dim=-1)[:, 1].cpu().numpy()

            # Ensemble based on strategy
            for i in range(len(texts)):
                p1, p2 = float(probs1[i]), float(probs2[i])
                pred1 = p1 > self.threshold
                pred2 = p2 > self.threshold

                if self.strategy == "and":
                    # Both must predict toxic
                    label = 1 if (pred1 and pred2) else 0
                    toxic_prob = min(p1, p2)  # Conservative estimate
                elif self.strategy == "or":
                    # Either predicts toxic
                    label = 1 if (pred1 or pred2) else 0
                    toxic_prob = max(p1, p2)  # Aggressive estimate
                else:  # weighted
                    toxic_prob = self.weight1 * p1 + self.weight2 * p2
                    label = 1 if toxic_prob > self.threshold else 0

                confidence = toxic_prob if label == 1 else (1 - toxic_prob)

                result = {
                    "label": label,
                    "label_text": "toxic" if label == 1 else "clean",
                    "confidence": round(confidence, 4),
                    "toxic_prob": round(toxic_prob, 4),
                }

                if return_probs:
                    result["prob_phase2"] = round(p1, 4)
                    result["prob_coevo"] = round(p2, 4)

                results.append(result)

        return results[0] if single_input else results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        return_probs: bool = False,
    ) -> List[Dict]:
        """Predict toxicity for a batch of texts.

        Args:
            texts: List of texts
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            return_probs: Whether to return individual model probabilities

        Returns:
            List of prediction dicts
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = self.predict(batch, return_probs=return_probs, max_length=max_length)
            all_results.extend(results)

        return all_results

    def is_toxic(self, text: str) -> bool:
        """Simple check if text is toxic.

        Args:
            text: Input text

        Returns:
            True if toxic, False otherwise
        """
        return self.predict(text)["label"] == 1

    def get_toxicity_score(self, text: str) -> float:
        """Get toxicity probability score.

        Args:
            text: Input text

        Returns:
            Probability of being toxic (0.0 to 1.0)
        """
        return self.predict(text)["toxic_prob"]


def create_ensemble(
    model_dir: str = None,
    strategy: Literal["and", "or", "weighted", "pmf"] = "and",
    weight1: float = 0.6,
    weight2: float = 0.4,
    threshold: float = 0.5,
    pmf_strategy: str = "meta_learner",
    device: Optional[str] = None,
) -> Union["EnsembleClassifier", "PMFEnsemble"]:
    """Factory function to create ensemble classifier.

    Args:
        model_dir: Base directory for models (default: ml-service/models)
        strategy: Ensemble strategy ('and', 'or', 'weighted', 'pmf')
            - 'and': Both Phase2 + Coevo must predict toxic (low FP)
            - 'or': Either model predicts toxic (low FN)
            - 'weighted': Weighted average of Phase2 + Coevo
            - 'pmf': PMF ensemble with 3 models and meta-learner (highest F1)
        weight1: Weight for Phase 2 model (only for 'weighted')
        weight2: Weight for Coevolution model (only for 'weighted')
        threshold: Classification threshold
        pmf_strategy: Strategy for PMF ensemble ('meta_learner', 'weighted_avg', etc.)
        device: Device to use ('cuda', 'cpu', or None for auto)

    Returns:
        EnsembleClassifier or PMFEnsemble instance
    """
    if model_dir is None:
        # Try to find models directory
        possible_paths = [
            Path("models"),
            Path("ml-service/models"),
            Path(__file__).parent.parent.parent.parent / "models",
        ]
        for path in possible_paths:
            if (path / "phase2-combined").exists():
                model_dir = path
                break

    if model_dir is None:
        raise ValueError("Could not find models directory")

    model_dir = Path(model_dir)

    # Use PMF ensemble if requested
    if strategy == "pmf":
        from ml_service.inference.pmf_ensemble import create_pmf_ensemble
        return create_pmf_ensemble(
            model_dir=str(model_dir),
            strategy=pmf_strategy,
            threshold=threshold,
            device=device,
        )

    return EnsembleClassifier(
        model1_path=str(model_dir / "phase2-combined" / "best_model"),
        model2_path=str(model_dir / "coevolution-latest"),
        strategy=strategy,
        weight1=weight1,
        weight2=weight2,
        threshold=threshold,
        device=device,
    )


# Type alias for backwards compatibility
PMFEnsemble = None  # Will be imported lazily when needed


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ensemble toxic text classifier")
    parser.add_argument("texts", nargs="*", help="Texts to classify")
    parser.add_argument("--file", "-f", help="File with texts (one per line)")
    parser.add_argument("--strategy", "-s", choices=["and", "or", "weighted", "pmf"], default="and",
                        help="Ensemble strategy (default: and, use 'pmf' for 3-model ensemble)")
    parser.add_argument("--pmf-strategy", choices=["meta_learner", "weighted_avg", "voting", "and", "or"],
                        default="meta_learner", help="PMF sub-strategy (when --strategy=pmf)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--weight1", type=float, default=0.6, help="Phase 2 weight (for weighted strategy)")
    parser.add_argument("--weight2", type=float, default=0.4, help="Coevolution weight (for weighted strategy)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    # Get texts
    texts = args.texts
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if not texts:
        print("Usage: python ensemble_classifier.py 'text to classify'")
        print("       python ensemble_classifier.py -f input.txt")
        print("       python ensemble_classifier.py --strategy or 'text'  # low FN")
        print("       python ensemble_classifier.py --strategy and 'text'  # low FP (default)")
        print("       python ensemble_classifier.py --strategy pmf 'text'  # 3-model ensemble (highest F1)")
        sys.exit(1)

    # Create classifier
    classifier = create_ensemble(
        strategy=args.strategy,
        weight1=args.weight1,
        weight2=args.weight2,
        threshold=args.threshold,
        pmf_strategy=args.pmf_strategy if args.strategy == "pmf" else "meta_learner",
    )

    # Classify
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION RESULTS (Strategy: {args.strategy.upper()})")
    print("=" * 60)

    results = classifier.predict(texts, return_probs=args.verbose)
    if not isinstance(results, list):
        results = [results]

    for text, result in zip(texts, results):
        label = "TOXIC" if result["label"] == 1 else "CLEAN"
        print(f"\n[{label}] (conf: {result['confidence']:.2%})")
        print(f"  \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        if args.verbose:
            print(f"  Phase2: {result['prob_phase2']:.4f}, Coevo: {result['prob_coevo']:.4f}")
