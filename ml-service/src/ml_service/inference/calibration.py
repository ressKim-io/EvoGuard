"""Temperature Scaling and Threshold Optimization for Model Calibration.

References:
- "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- Post-hoc calibration without retraining

Usage:
    from ml_service.inference.calibration import TemperatureScaler, ThresholdOptimizer

    # Fit temperature on validation set
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)

    # Calibrate test logits
    calibrated_probs = scaler.calibrate(test_logits)

    # Find optimal threshold
    optimizer = ThresholdOptimizer()
    best_threshold = optimizer.find_optimal(calibrated_probs, val_labels)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaler:
    """Post-hoc temperature scaling for model calibration.

    Learns a single scalar T such that softmax(logits/T) produces
    well-calibrated probabilities. T > 1 softens overconfident predictions.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature parameter by minimizing NLL on validation set.

        Args:
            logits: Raw model logits [N, num_classes]
            labels: True labels [N]
            lr: Learning rate for L-BFGS
            max_iter: Maximum iterations

        Returns:
            Fitted temperature value
        """
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.long)

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        criterion = nn.CrossEntropyLoss()

        optimizer = LBFGS([temperature], lr=lr, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        self.temperature = temperature.item()
        return self.temperature

    def calibrate(self, logits: torch.Tensor | np.ndarray) -> np.ndarray:
        """Apply temperature scaling to get calibrated probabilities.

        Args:
            logits: Raw model logits [N, num_classes]

        Returns:
            Calibrated probabilities [N, num_classes]
        """
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32)

        with torch.no_grad():
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)

        return probs.numpy()


class ThresholdOptimizer:
    """Find optimal classification threshold for binary classification."""

    @staticmethod
    def find_optimal(
        probs: np.ndarray,
        labels: np.ndarray,
        metric: str = "f1",
        n_thresholds: int = 1000,
    ) -> tuple[float, float]:
        """Find threshold that maximizes the given metric.

        Args:
            probs: Positive class probabilities [N] or [N, 2]
            labels: True labels [N]
            metric: Metric to optimize ("f1", "balanced_accuracy")
            n_thresholds: Number of thresholds to search

        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        from sklearn.metrics import f1_score

        if probs.ndim == 2:
            probs = probs[:, 1]  # Take positive class probability

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        best_threshold = 0.5
        best_score = 0.0

        thresholds = np.linspace(0.01, 0.99, n_thresholds)

        for t in thresholds:
            preds = (probs >= t).astype(int)

            if metric == "f1":
                score = f1_score(labels, preds, average="weighted")
            else:
                from sklearn.metrics import balanced_accuracy_score
                score = balanced_accuracy_score(labels, preds)

            if score > best_score:
                best_score = score
                best_threshold = t

        return best_threshold, best_score


def collect_logits(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 128,
    batch_size: int = 64,
) -> torch.Tensor:
    """Collect raw logits from model for a set of texts.

    Args:
        model: Classification model
        tokenizer: Tokenizer
        texts: Input texts
        device: Compute device
        max_length: Max token length
        batch_size: Batch size

    Returns:
        Logits tensor [N, num_classes]
    """
    model.eval()
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())

    return torch.cat(all_logits, dim=0)


def calibrate_and_evaluate(
    model: nn.Module,
    tokenizer,
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    device: str = "cuda",
) -> dict:
    """Full calibration pipeline: fit on val, evaluate on test.

    Returns:
        Dict with temperature, threshold, and test metrics
    """
    from sklearn.metrics import confusion_matrix, f1_score

    # Collect logits
    print("Collecting validation logits...")
    val_logits = collect_logits(model, tokenizer, val_texts, device)
    print("Collecting test logits...")
    test_logits = collect_logits(model, tokenizer, test_texts, device)

    val_labels_t = torch.tensor(val_labels, dtype=torch.long)

    # Fit temperature
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels_t)
    print(f"Fitted temperature: {temperature:.4f}")

    # Calibrate
    val_probs = scaler.calibrate(val_logits)
    test_probs = scaler.calibrate(test_logits)

    # Find optimal threshold on validation set
    optimizer = ThresholdOptimizer()
    best_threshold, val_score = optimizer.find_optimal(
        val_probs, np.array(val_labels)
    )
    print(f"Optimal threshold: {best_threshold:.4f} (val F1: {val_score:.4f})")

    # Evaluate on test with optimal threshold
    test_preds_opt = (test_probs[:, 1] >= best_threshold).astype(int)
    f1_opt = f1_score(test_labels, test_preds_opt, average="weighted")
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(test_labels, test_preds_opt).ravel()

    # Also evaluate with default 0.5 threshold for comparison
    test_preds_def = (test_probs[:, 1] >= 0.5).astype(int)
    f1_def = f1_score(test_labels, test_preds_def, average="weighted")
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(test_labels, test_preds_def).ravel()

    return {
        "temperature": temperature,
        "optimal_threshold": best_threshold,
        "test_default_threshold": {
            "f1_weighted": f1_def,
            "fp": int(fp_d), "fn": int(fn_d),
            "tp": int(tp_d), "tn": int(tn_d),
        },
        "test_optimal_threshold": {
            "f1_weighted": f1_opt,
            "fp": int(fp_o), "fn": int(fn_o),
            "tp": int(tp_o), "tn": int(tn_o),
        },
    }
