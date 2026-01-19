"""Battle feature transformer for extracting battle statistics."""

from typing import Any

from ml_service.feature_store.compute.base import FeatureTransformer


class BattleFeatureTransformer(FeatureTransformer):
    """Extract statistical features from battle data.

    Features extracted:
        - detection_rate: Ratio of successful detections
        - evasion_rate: Ratio of successful evasions
        - avg_confidence: Average confidence score across rounds
        - round_count: Total number of rounds
        - success_streak: Maximum consecutive successful detections

    Example:
        >>> transformer = BattleFeatureTransformer()
        >>> battle = {
        ...     "rounds": [
        ...         {"detected": True, "evaded": False, "confidence": 0.9},
        ...         {"detected": True, "evaded": False, "confidence": 0.85},
        ...         {"detected": False, "evaded": True, "confidence": 0.4},
        ...     ]
        ... }
        >>> features = transformer.transform(battle)
        >>> print(features)
        {'detection_rate': 0.6667, 'evasion_rate': 0.3333, ...}
    """

    @property
    def feature_names(self) -> list[str]:
        """List of features produced by this transformer."""
        return [
            "detection_rate",
            "evasion_rate",
            "avg_confidence",
            "round_count",
            "success_streak",
        ]

    @property
    def entity_type(self) -> str:
        """Entity type for battle features."""
        return "battle"

    @property
    def version(self) -> str:
        """Transformer version."""
        return "1.0.0"

    def transform(self, data: Any) -> dict[str, Any]:
        """Extract features from battle data.

        Args:
            data: Battle dictionary with 'rounds' key containing list of round data.
                  Each round should have 'detected', 'evaded', and 'confidence' keys.

        Returns:
            Dictionary with computed feature values.
        """
        # Handle non-dict input
        if not isinstance(data, dict):
            return self._empty_features()

        rounds = data.get("rounds", [])
        if not rounds:
            return self._empty_features()

        total = len(rounds)

        # Count detections and evasions
        detected_count = sum(1 for r in rounds if r.get("detected", False))
        evaded_count = sum(1 for r in rounds if r.get("evaded", False))

        # Calculate average confidence
        confidences = [r.get("confidence", 0.0) for r in rounds]
        avg_confidence = sum(confidences) / total if total > 0 else 0.0

        # Calculate success streak
        success_streak = self._max_streak(rounds)

        return {
            "detection_rate": round(detected_count / total, 4),
            "evasion_rate": round(evaded_count / total, 4),
            "avg_confidence": round(avg_confidence, 4),
            "round_count": total,
            "success_streak": success_streak,
        }

    def _empty_features(self) -> dict[str, Any]:
        """Return features with zero/default values."""
        return {
            "detection_rate": 0.0,
            "evasion_rate": 0.0,
            "avg_confidence": 0.0,
            "round_count": 0,
            "success_streak": 0,
        }

    def _max_streak(self, rounds: list[dict[str, Any]]) -> int:
        """Calculate maximum consecutive successful detections.

        Args:
            rounds: List of round dictionaries.

        Returns:
            Maximum consecutive detections.
        """
        max_streak = 0
        current_streak = 0

        for r in rounds:
            if r.get("detected", False):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def get_feature_schema(self) -> dict[str, str]:
        """Get schema with feature types."""
        return {
            "detection_rate": "float",
            "evasion_rate": "float",
            "avg_confidence": "float",
            "round_count": "int",
            "success_streak": "int",
        }

    def transform_batch(self, data_list: list[Any]) -> list[dict[str, Any]]:
        """Transform a batch of battles.

        Args:
            data_list: List of battle dictionaries.

        Returns:
            List of feature dictionaries.
        """
        return [self.transform(battle) for battle in data_list]
