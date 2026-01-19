"""Data Drift Detection - Statistical methods for detecting distribution shifts.

This module provides implementations for detecting data drift using:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) Test
- Text-specific drift monitoring
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PSIResult:
    """Result of PSI calculation."""

    psi: float
    n_bins: int
    alert_level: str  # "none", "minor", "significant"
    reference_size: int
    current_size: int
    timestamp: datetime


@dataclass
class KSTestResult:
    """Result of KS Test."""

    drifted: bool
    statistic: float
    p_value: float
    threshold: float
    reference_size: int
    current_size: int
    timestamp: datetime


@dataclass
class TextDriftResult:
    """Result of text data drift detection."""

    features: dict[str, dict[str, Any]]
    overall_drift_detected: bool
    drift_features: list[str]
    timestamp: datetime


def calculate_psi(
    reference: list[float],
    current: list[float],
    n_bins: int = 10,
) -> PSIResult:
    """Calculate Population Stability Index (PSI).

    PSI measures the shift in distribution between reference and current data.

    Interpretation:
    - PSI < 0.1: No significant change (stable)
    - 0.1 <= PSI < 0.2: Minor change (monitor)
    - PSI >= 0.2: Significant change (action required)

    Args:
        reference: Reference data distribution.
        current: Current data distribution.
        n_bins: Number of bins for histogram.

    Returns:
        PSIResult with calculated PSI and metadata.

    Example:
        >>> ref_data = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> cur_data = [0.2, 0.3, 0.4, 0.5, 0.6]
        >>> result = calculate_psi(ref_data, cur_data)
        >>> print(f"PSI: {result.psi:.3f}, Level: {result.alert_level}")
    """
    if not reference or not current:
        return PSIResult(
            psi=0.0,
            n_bins=n_bins,
            alert_level="none",
            reference_size=len(reference),
            current_size=len(current),
            timestamp=datetime.now(UTC),
        )

    # Calculate bin edges using reference data
    min_val = min(min(reference), min(current))
    max_val = max(max(reference), max(current))

    # Handle edge case where all values are the same
    if min_val == max_val:
        return PSIResult(
            psi=0.0,
            n_bins=n_bins,
            alert_level="none",
            reference_size=len(reference),
            current_size=len(current),
            timestamp=datetime.now(UTC),
        )

    bin_width = (max_val - min_val) / n_bins

    # Calculate histograms
    def histogram(data: list[float]) -> list[int]:
        counts = [0] * n_bins
        for val in data:
            bin_idx = min(int((val - min_val) / bin_width), n_bins - 1)
            counts[bin_idx] += 1
        return counts

    ref_hist = histogram(reference)
    cur_hist = histogram(current)

    # Convert to proportions with smoothing to avoid division by zero
    ref_total = len(reference)
    cur_total = len(current)

    # Add small constant for smoothing (Laplace smoothing)
    ref_pct = [(count + 1) / (ref_total + n_bins) for count in ref_hist]
    cur_pct = [(count + 1) / (cur_total + n_bins) for count in cur_hist]

    # Calculate PSI
    psi = sum((cur_pct[i] - ref_pct[i]) * math.log(cur_pct[i] / ref_pct[i]) for i in range(n_bins))

    # Determine alert level
    if psi >= 0.2:
        alert_level = "significant"
    elif psi >= 0.1:
        alert_level = "minor"
    else:
        alert_level = "none"

    return PSIResult(
        psi=psi,
        n_bins=n_bins,
        alert_level=alert_level,
        reference_size=ref_total,
        current_size=cur_total,
        timestamp=datetime.now(UTC),
    )


def ks_test(
    reference: list[float],
    current: list[float],
    threshold: float = 0.05,
) -> KSTestResult:
    """Perform Kolmogorov-Smirnov test for distribution comparison.

    The KS test compares the empirical cumulative distribution functions
    of the reference and current data.

    Args:
        reference: Reference data distribution.
        current: Current data distribution.
        threshold: P-value threshold for drift detection.

    Returns:
        KSTestResult with test results.

    Example:
        >>> ref_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> cur_data = [2.0, 3.0, 4.0, 5.0, 6.0]
        >>> result = ks_test(ref_data, cur_data)
        >>> print(f"Drifted: {result.drifted}, p-value: {result.p_value:.4f}")
    """
    if not reference or not current:
        return KSTestResult(
            drifted=False,
            statistic=0.0,
            p_value=1.0,
            threshold=threshold,
            reference_size=len(reference),
            current_size=len(current),
            timestamp=datetime.now(UTC),
        )

    n1 = len(reference)
    n2 = len(current)

    # Sort both datasets
    sorted_ref = sorted(reference)
    sorted_cur = sorted(current)

    # Combine and sort all unique values
    all_values = sorted(set(reference) | set(current))

    # Calculate empirical CDFs
    def ecdf(sorted_data: list[float], value: float) -> float:
        """Calculate empirical CDF at a given value."""
        count = sum(1 for x in sorted_data if x <= value)
        return count / len(sorted_data)

    # Find maximum difference between CDFs
    max_diff = 0.0
    for value in all_values:
        cdf_ref = ecdf(sorted_ref, value)
        cdf_cur = ecdf(sorted_cur, value)
        diff = abs(cdf_ref - cdf_cur)
        max_diff = max(max_diff, diff)

    statistic = max_diff

    # Approximate p-value using asymptotic formula
    # For large samples: sqrt(n*m/(n+m)) * D ~ Kolmogorov distribution
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * statistic

    # Approximate p-value using Kolmogorov distribution
    # P(K <= x) ≈ 1 - 2*sum((-1)^(i-1)*exp(-2*i^2*x^2)) for large x
    if lambda_val <= 0:
        p_value = 1.0
    elif lambda_val >= 3.0:
        p_value = 0.0
    else:
        # Use approximation for moderate values
        p_value = 0.0
        for i in range(1, 101):
            term = 2 * ((-1) ** (i - 1)) * math.exp(-2 * i * i * lambda_val * lambda_val)
            p_value += term
        p_value = min(max(p_value, 0.0), 1.0)

    return KSTestResult(
        drifted=p_value < threshold,
        statistic=statistic,
        p_value=p_value,
        threshold=threshold,
        reference_size=n1,
        current_size=n2,
        timestamp=datetime.now(UTC),
    )


class TextDataDriftMonitor:
    """Monitor drift in text data characteristics.

    Tracks text-specific features like length, word count, and unicode ratio
    to detect distribution shifts in incoming text data.

    Example:
        >>> reference_texts = ["Hello world", "How are you?", "Good morning!"]
        >>> monitor = TextDataDriftMonitor.from_reference(reference_texts)
        >>> current_texts = ["Привет мир", "Как дела?", "Доброе утро!"]
        >>> result = monitor.detect_drift(current_texts)
        >>> if result.overall_drift_detected:
        ...     print(f"Drift detected in: {result.drift_features}")
    """

    def __init__(
        self,
        reference_stats: dict[str, list[float]],
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
    ) -> None:
        """Initialize the text drift monitor.

        Args:
            reference_stats: Dictionary of feature name to reference values.
            psi_threshold: PSI threshold for drift detection.
            ks_threshold: KS test p-value threshold.
        """
        self.reference_stats = reference_stats
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    @classmethod
    def from_reference(
        cls,
        texts: list[str],
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
    ) -> "TextDataDriftMonitor":
        """Create a monitor from reference texts.

        Args:
            texts: Reference text data.
            psi_threshold: PSI threshold for drift detection.
            ks_threshold: KS test p-value threshold.

        Returns:
            TextDataDriftMonitor instance.
        """
        reference_stats = cls._compute_stats(texts)
        return cls(reference_stats, psi_threshold, ks_threshold)

    @staticmethod
    def _compute_stats(texts: list[str]) -> dict[str, list[float]]:
        """Compute text statistics.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary of feature name to values.
        """
        stats: dict[str, list[float]] = {
            "text_length": [],
            "word_count": [],
            "unicode_ratio": [],
            "avg_word_length": [],
        }

        for text in texts:
            text_length = len(text)
            words = text.split()
            word_count = len(words)

            # Unicode ratio (non-ASCII characters)
            unicode_chars = sum(1 for c in text if ord(c) > 127)
            unicode_ratio = unicode_chars / max(text_length, 1)

            # Average word length
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

            stats["text_length"].append(float(text_length))
            stats["word_count"].append(float(word_count))
            stats["unicode_ratio"].append(unicode_ratio)
            stats["avg_word_length"].append(avg_word_length)

        return stats

    def detect_drift(self, texts: list[str]) -> TextDriftResult:
        """Detect drift in the given texts.

        Args:
            texts: Current text data to analyze.

        Returns:
            TextDriftResult with drift detection results.
        """
        current_stats = self._compute_stats(texts)
        results: dict[str, dict[str, Any]] = {}
        drift_features: list[str] = []

        for feature in self.reference_stats:
            reference = self.reference_stats[feature]
            current = current_stats.get(feature, [])

            # Calculate PSI
            psi_result = calculate_psi(reference, current)

            # Calculate KS test
            ks_result = ks_test(reference, current, self.ks_threshold)

            # Determine if drift detected
            psi_drift = psi_result.psi >= self.psi_threshold
            ks_drift = ks_result.drifted

            results[feature] = {
                "psi": psi_result.psi,
                "psi_alert": psi_result.alert_level,
                "psi_drift": psi_drift,
                "ks_statistic": ks_result.statistic,
                "ks_p_value": ks_result.p_value,
                "ks_drift": ks_drift,
                "drift_detected": psi_drift or ks_drift,
            }

            if psi_drift or ks_drift:
                drift_features.append(feature)

        return TextDriftResult(
            features=results,
            overall_drift_detected=len(drift_features) > 0,
            drift_features=drift_features,
            timestamp=datetime.now(UTC),
        )

    def update_reference(self, texts: list[str]) -> None:
        """Update the reference statistics.

        Args:
            texts: New reference texts.
        """
        self.reference_stats = self._compute_stats(texts)


class StreamingDataDriftMonitor:
    """Monitor data drift in streaming fashion.

    Maintains sliding windows of reference and current data
    for continuous drift monitoring.

    Example:
        >>> monitor = StreamingDataDriftMonitor(
        ...     reference_window_size=1000,
        ...     current_window_size=100,
        ... )
        >>> # Add reference data during warm-up
        >>> for value in reference_values:
        ...     monitor.add_reference(value)
        >>> # Switch to monitoring mode
        >>> for value in incoming_values:
        ...     result = monitor.add_current(value)
        ...     if result and result.psi >= 0.2:
        ...         print("Drift detected!")
    """

    def __init__(
        self,
        reference_window_size: int = 1000,
        current_window_size: int = 100,
        check_interval: int = 50,
    ) -> None:
        """Initialize the streaming drift monitor.

        Args:
            reference_window_size: Size of the reference window.
            current_window_size: Size of the current window.
            check_interval: Number of samples between drift checks.
        """
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.check_interval = check_interval

        self._reference: deque[float] = deque(maxlen=reference_window_size)
        self._current: deque[float] = deque(maxlen=current_window_size)
        self._samples_since_check = 0
        self._last_result: PSIResult | None = None

    @property
    def reference_ready(self) -> bool:
        """Check if reference window is fully populated."""
        return len(self._reference) >= self.reference_window_size

    @property
    def current_ready(self) -> bool:
        """Check if current window has enough samples."""
        return len(self._current) >= self.current_window_size // 2

    def add_reference(self, value: float) -> None:
        """Add a value to the reference window.

        Args:
            value: Value to add.
        """
        self._reference.append(value)

    def add_reference_batch(self, values: list[float]) -> None:
        """Add multiple values to the reference window.

        Args:
            values: Values to add.
        """
        for value in values:
            self._reference.append(value)

    def add_current(self, value: float) -> PSIResult | None:
        """Add a value to the current window and optionally check for drift.

        Args:
            value: Value to add.

        Returns:
            PSIResult if drift check was performed, None otherwise.
        """
        self._current.append(value)
        self._samples_since_check += 1

        if self._samples_since_check >= self.check_interval and self.current_ready:
            return self.check_drift()

        return None

    def add_current_batch(self, values: list[float]) -> PSIResult | None:
        """Add multiple values to the current window.

        Args:
            values: Values to add.

        Returns:
            PSIResult if drift check was performed, None otherwise.
        """
        result = None
        for value in values:
            result = self.add_current(value) or result
        return result

    def check_drift(self) -> PSIResult:
        """Check for drift between reference and current windows.

        Returns:
            PSIResult with drift detection results.
        """
        result = calculate_psi(list(self._reference), list(self._current))
        self._last_result = result
        self._samples_since_check = 0
        return result

    def get_last_result(self) -> PSIResult | None:
        """Get the last drift check result.

        Returns:
            Last PSIResult or None if no check has been performed.
        """
        return self._last_result

    def reset_current(self) -> None:
        """Reset the current window."""
        self._current.clear()
        self._samples_since_check = 0

    def reset_all(self) -> None:
        """Reset both reference and current windows."""
        self._reference.clear()
        self._current.clear()
        self._samples_since_check = 0
        self._last_result = None
