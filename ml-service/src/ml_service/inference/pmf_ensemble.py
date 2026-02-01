"""PMF (Parallel Model Fusion) Ensemble Classifier.

Combines multiple transformer models using meta-learning for optimal weighting.

Architecture:
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  KcELECTRA      │  │  KLUE-BERT      │  │  KoELECTRA v3   │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └──────────┬─────────┴─────────┬──────────┘
                        │                   │
                  ┌─────▼─────┐       ┌─────▼─────┐
                  │ Meta-     │       │ Weighted  │
                  │ Learner   │       │ Average   │
                  └─────┬─────┘       └─────┬─────┘
                        │                   │
                        └─────────┬─────────┘
                                  │
                            ┌─────▼─────┐
                            │ Final     │
                            │ Prediction│
                            └───────────┘

Strategies:
- meta_learner: XGBoost/LogisticRegression trained on validation set
- weighted_avg: Simple weighted average of probabilities
- voting: Hard voting (majority vote)
- stacking: Stacked generalization with cross-validation
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Union, Literal, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class PMFEnsemble:
    """Parallel Model Fusion ensemble for toxic text classification."""

    DEFAULT_MODEL_PATHS = [
        "models/pmf/kcelectra/best_model",
        "models/pmf/klue-bert/best_model",
        "models/pmf/koelectra-v3/best_model",
    ]

    DEFAULT_WEIGHTS = [0.4, 0.3, 0.3]  # KcELECTRA gets higher weight

    def __init__(
        self,
        model_paths: List[str] = None,
        meta_learner_path: str = "models/pmf/meta_learner.pkl",
        strategy: Literal["meta_learner", "weighted_avg", "voting", "and", "or"] = "meta_learner",
        weights: List[float] = None,
        threshold: float = 0.5,
        device: str = None,
        lazy_load: bool = False,
    ):
        """Initialize PMF ensemble.

        Args:
            model_paths: List of paths to fine-tuned models
            meta_learner_path: Path to trained meta-learner
            strategy: Ensemble strategy
                - 'meta_learner': Use trained XGBoost/LogisticRegression
                - 'weighted_avg': Weighted average of probabilities
                - 'voting': Hard voting (majority)
                - 'and': All models must predict toxic (low FP)
                - 'or': Any model predicts toxic (low FN)
            weights: Weights for each model (for weighted_avg strategy)
            threshold: Classification threshold
            device: Device to use ('cuda', 'cpu', or None for auto)
            lazy_load: If True, don't load models until first prediction
        """
        self.model_paths = model_paths or self.DEFAULT_MODEL_PATHS
        self.meta_learner_path = meta_learner_path
        self.strategy = strategy
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        self.lazy_load = lazy_load

        # Validate weights
        if len(self.weights) != len(self.model_paths):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of models ({len(self.model_paths)})"
            )

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model containers
        self.models: List = []
        self.tokenizers: List = []
        self.model_names: List[str] = []
        self.meta_learner = None

        if not lazy_load:
            self._load_models()
            self._load_meta_learner()

    def _load_models(self):
        """Load all transformer models."""
        logger.info(f"Loading {len(self.model_paths)} models on {self.device}...")

        for path in self.model_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")

            model_name = path.parent.name if path.name == "best_model" else path.name
            logger.info(f"  Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(self.device)
            model.eval()

            self.tokenizers.append(tokenizer)
            self.models.append(model)
            self.model_names.append(model_name)

        logger.info(f"Loaded models: {', '.join(self.model_names)}")

    def _load_meta_learner(self):
        """Load trained meta-learner if strategy requires it."""
        if self.strategy != "meta_learner":
            return

        meta_path = Path(self.meta_learner_path)
        if not meta_path.exists():
            logger.warning(
                f"Meta-learner not found at {meta_path}. "
                "Run train_meta_learner.py first or use 'weighted_avg' strategy."
            )
            return

        with open(meta_path, "rb") as f:
            self.meta_learner = pickle.load(f)
        logger.info(f"Loaded meta-learner from {meta_path}")

    def _ensure_loaded(self):
        """Ensure models are loaded (for lazy loading)."""
        if not self.models:
            self._load_models()
            self._load_meta_learner()

    def get_individual_predictions(
        self,
        texts: Union[str, List[str]],
        max_length: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model.

        Args:
            texts: Single text or list of texts
            max_length: Maximum sequence length

        Returns:
            Dictionary with model names as keys and probability arrays as values
        """
        self._ensure_loaded()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        results = {}

        with torch.no_grad():
            for name, model, tokenizer in zip(self.model_names, self.models, self.tokenizers):
                encoding = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                outputs = model(**encoding)
                probs = F.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                results[name] = probs

        return results

    def _combine_predictions(
        self,
        probs_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine predictions from multiple models.

        Args:
            probs_dict: Dictionary of model_name -> probabilities

        Returns:
            Combined probability array
        """
        probs_list = [probs_dict[name] for name in self.model_names]
        probs_matrix = np.stack(probs_list, axis=1)  # Shape: (n_samples, n_models)

        if self.strategy == "meta_learner" and self.meta_learner is not None:
            # Use meta-learner for prediction
            combined = self.meta_learner.predict_proba(probs_matrix)[:, 1]

        elif self.strategy == "weighted_avg":
            # Weighted average
            weights = np.array(self.weights)
            combined = np.average(probs_matrix, axis=1, weights=weights)

        elif self.strategy == "voting":
            # Hard voting
            votes = (probs_matrix > self.threshold).astype(int)
            combined = votes.mean(axis=1)  # Proportion of positive votes

        elif self.strategy == "and":
            # All must predict toxic
            all_positive = np.all(probs_matrix > self.threshold, axis=1)
            combined = np.where(all_positive, probs_matrix.min(axis=1), probs_matrix.min(axis=1) * 0.5)

        elif self.strategy == "or":
            # Any predicts toxic
            any_positive = np.any(probs_matrix > self.threshold, axis=1)
            combined = np.where(any_positive, probs_matrix.max(axis=1), probs_matrix.max(axis=1))

        else:
            # Default to simple average
            combined = probs_matrix.mean(axis=1)

        return combined

    def predict(
        self,
        texts: Union[str, List[str]],
        return_individual: bool = False,
        max_length: int = 256,
    ) -> Union[Dict, List[Dict]]:
        """Predict toxicity using ensemble.

        Args:
            texts: Single text or list of texts
            return_individual: Include individual model predictions
            max_length: Maximum sequence length

        Returns:
            Dict or list of dicts with:
                - label: 0 (clean) or 1 (toxic)
                - label_text: 'clean' or 'toxic'
                - confidence: Ensemble confidence
                - toxic_prob: Probability of toxic
                - individual: Dict of model predictions (if return_individual=True)
        """
        self._ensure_loaded()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Get individual predictions
        probs_dict = self.get_individual_predictions(texts, max_length)

        # Combine predictions
        combined_probs = self._combine_predictions(probs_dict)

        # Build results
        results = []
        for i in range(len(texts)):
            prob = float(combined_probs[i])
            label = 1 if prob > self.threshold else 0
            confidence = prob if label == 1 else (1 - prob)

            result = {
                "label": label,
                "label_text": "toxic" if label == 1 else "clean",
                "confidence": round(confidence, 4),
                "toxic_prob": round(prob, 4),
            }

            if return_individual:
                result["individual"] = {
                    name: round(float(probs_dict[name][i]), 4)
                    for name in self.model_names
                }

            results.append(result)

        return results[0] if single_input else results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        return_individual: bool = False,
    ) -> List[Dict]:
        """Predict toxicity for a batch of texts.

        Args:
            texts: List of texts
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            return_individual: Include individual model predictions

        Returns:
            List of prediction dicts
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.predict(batch, return_individual=return_individual, max_length=max_length)
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

    def evaluate(
        self,
        valid_path: str,
        batch_size: int = 32,
    ) -> Dict:
        """Evaluate ensemble on validation set.

        Args:
            valid_path: Path to validation CSV with 'text' and 'label' columns
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        import pandas as pd
        from sklearn.metrics import (
            f1_score, accuracy_score, precision_score, recall_score,
            confusion_matrix, roc_auc_score
        )

        df = pd.read_csv(valid_path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        # Get predictions
        results = self.predict_batch(texts, batch_size=batch_size)
        preds = [r["label"] for r in results]
        probs = [r["toxic_prob"] for r in results]

        # Calculate metrics
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds, average="weighted")
        auc = roc_auc_score(labels, probs)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "auc_roc": float(auc),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
            "strategy": self.strategy,
            "threshold": self.threshold,
            "n_models": len(self.models),
            "model_names": self.model_names,
        }

        logger.info(f"PMF Ensemble Evaluation ({self.strategy}):")
        logger.info(f"  F1: {f1:.4f}, Acc: {acc:.4f}")
        logger.info(f"  FP: {fp}, FN: {fn}")
        logger.info(f"  AUC-ROC: {auc:.4f}")

        return metrics

    def save_config(self, path: str):
        """Save ensemble configuration."""
        config = {
            "model_paths": self.model_paths,
            "meta_learner_path": self.meta_learner_path,
            "strategy": self.strategy,
            "weights": self.weights,
            "threshold": self.threshold,
            "model_names": self.model_names,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, path: str, device: str = None) -> "PMFEnsemble":
        """Load ensemble from configuration file."""
        with open(path, "r") as f:
            config = json.load(f)

        return cls(
            model_paths=config["model_paths"],
            meta_learner_path=config.get("meta_learner_path", "models/pmf/meta_learner.pkl"),
            strategy=config.get("strategy", "meta_learner"),
            weights=config.get("weights"),
            threshold=config.get("threshold", 0.5),
            device=device,
        )


def create_pmf_ensemble(
    model_dir: str = None,
    strategy: Literal["meta_learner", "weighted_avg", "voting", "and", "or"] = "meta_learner",
    weights: List[float] = None,
    threshold: float = 0.5,
    device: str = None,
) -> PMFEnsemble:
    """Factory function to create PMF ensemble.

    Args:
        model_dir: Base directory for models (default: ml-service/models)
        strategy: Ensemble strategy
        weights: Model weights (for weighted_avg)
        threshold: Classification threshold
        device: Device to use

    Returns:
        PMFEnsemble instance
    """
    if model_dir is None:
        # Find models directory
        possible_paths = [
            Path("models"),
            Path("ml-service/models"),
            Path(__file__).parent.parent.parent.parent / "models",
        ]
        for path in possible_paths:
            if (path / "pmf").exists():
                model_dir = path
                break

    if model_dir is None:
        raise ValueError("Could not find models/pmf directory")

    model_dir = Path(model_dir)

    model_paths = [
        str(model_dir / "pmf" / "kcelectra" / "best_model"),
        str(model_dir / "pmf" / "klue-bert" / "best_model"),
        str(model_dir / "pmf" / "koelectra-v3" / "best_model"),
    ]

    meta_learner_path = str(model_dir / "pmf" / "meta_learner.pkl")

    return PMFEnsemble(
        model_paths=model_paths,
        meta_learner_path=meta_learner_path,
        strategy=strategy,
        weights=weights,
        threshold=threshold,
        device=device,
    )


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="PMF Ensemble toxic text classifier")
    parser.add_argument("texts", nargs="*", help="Texts to classify")
    parser.add_argument("--file", "-f", help="File with texts (one per line)")
    parser.add_argument("--strategy", "-s",
                        choices=["meta_learner", "weighted_avg", "voting", "and", "or"],
                        default="meta_learner", help="Ensemble strategy")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--evaluate", "-e", help="Evaluate on CSV file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show individual model predictions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create ensemble
    try:
        ensemble = create_pmf_ensemble(
            strategy=args.strategy,
            threshold=args.threshold,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("\nMake sure to train models first:")
        print("  python scripts/train_multi_model.py")
        print("  python scripts/train_meta_learner.py")
        sys.exit(1)

    if args.evaluate:
        # Evaluate on validation set
        metrics = ensemble.evaluate(args.evaluate)
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Strategy: {args.strategy}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"FP: {metrics['fp']}, FN: {metrics['fn']}")
        sys.exit(0)

    # Get texts
    texts = args.texts
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if not texts:
        print("Usage: python pmf_ensemble.py 'text to classify'")
        print("       python pmf_ensemble.py -f input.txt")
        print("       python pmf_ensemble.py -e validation.csv  # evaluate")
        sys.exit(1)

    # Classify
    print("\n" + "=" * 60)
    print(f"PMF ENSEMBLE RESULTS (Strategy: {args.strategy.upper()})")
    print("=" * 60)

    results = ensemble.predict(texts, return_individual=args.verbose)
    if not isinstance(results, list):
        results = [results]

    for text, result in zip(texts, results):
        label = "TOXIC" if result["label"] == 1 else "CLEAN"
        print(f"\n[{label}] (conf: {result['confidence']:.2%})")
        print(f'  "{text[:80]}{"..." if len(text) > 80 else ""}"')
        if args.verbose and "individual" in result:
            print("  Individual predictions:")
            for name, prob in result["individual"].items():
                print(f"    {name}: {prob:.4f}")
