"""Tests for BattleFeatureTransformer."""

import pytest

from ml_service.feature_store.compute.battle_features import BattleFeatureTransformer


class TestBattleFeatureTransformer:
    """Tests for BattleFeatureTransformer."""

    @pytest.fixture
    def transformer(self) -> BattleFeatureTransformer:
        """Create a transformer instance."""
        return BattleFeatureTransformer()

    def test_feature_names(self, transformer: BattleFeatureTransformer) -> None:
        """Test that feature names are correct."""
        expected = [
            "detection_rate",
            "evasion_rate",
            "avg_confidence",
            "round_count",
            "success_streak",
        ]
        assert transformer.feature_names == expected

    def test_entity_type(self, transformer: BattleFeatureTransformer) -> None:
        """Test entity type."""
        assert transformer.entity_type == "battle"

    def test_version(self, transformer: BattleFeatureTransformer) -> None:
        """Test version."""
        assert transformer.version == "1.0.0"

    def test_transform_basic_battle(self, transformer: BattleFeatureTransformer) -> None:
        """Test transforming a basic battle."""
        battle = {
            "rounds": [
                {"detected": True, "evaded": False, "confidence": 0.9},
                {"detected": True, "evaded": False, "confidence": 0.85},
                {"detected": False, "evaded": True, "confidence": 0.4},
            ]
        }

        features = transformer.transform(battle)

        assert features["detection_rate"] == pytest.approx(0.6667, rel=1e-3)
        assert features["evasion_rate"] == pytest.approx(0.3333, rel=1e-3)
        assert features["avg_confidence"] == pytest.approx(0.7167, rel=1e-3)
        assert features["round_count"] == 3
        assert features["success_streak"] == 2

    def test_transform_all_detected(self, transformer: BattleFeatureTransformer) -> None:
        """Test battle where all rounds are detected."""
        battle = {
            "rounds": [
                {"detected": True, "evaded": False, "confidence": 0.95},
                {"detected": True, "evaded": False, "confidence": 0.90},
                {"detected": True, "evaded": False, "confidence": 0.92},
                {"detected": True, "evaded": False, "confidence": 0.88},
            ]
        }

        features = transformer.transform(battle)

        assert features["detection_rate"] == 1.0
        assert features["evasion_rate"] == 0.0
        assert features["success_streak"] == 4

    def test_transform_all_evaded(self, transformer: BattleFeatureTransformer) -> None:
        """Test battle where all rounds are evaded."""
        battle = {
            "rounds": [
                {"detected": False, "evaded": True, "confidence": 0.3},
                {"detected": False, "evaded": True, "confidence": 0.25},
                {"detected": False, "evaded": True, "confidence": 0.2},
            ]
        }

        features = transformer.transform(battle)

        assert features["detection_rate"] == 0.0
        assert features["evasion_rate"] == 1.0
        assert features["success_streak"] == 0

    def test_transform_empty_rounds(self, transformer: BattleFeatureTransformer) -> None:
        """Test battle with empty rounds."""
        battle = {"rounds": []}

        features = transformer.transform(battle)

        assert features["detection_rate"] == 0.0
        assert features["evasion_rate"] == 0.0
        assert features["avg_confidence"] == 0.0
        assert features["round_count"] == 0
        assert features["success_streak"] == 0

    def test_transform_no_rounds_key(self, transformer: BattleFeatureTransformer) -> None:
        """Test battle without rounds key."""
        battle = {"some_other_key": "value"}

        features = transformer.transform(battle)

        assert features["round_count"] == 0
        assert all(v == 0 or v == 0.0 for v in features.values())

    def test_transform_non_dict_input(self, transformer: BattleFeatureTransformer) -> None:
        """Test with non-dict input."""
        features = transformer.transform("not a dict")

        assert features["round_count"] == 0

    def test_transform_none_input(self, transformer: BattleFeatureTransformer) -> None:
        """Test with None input."""
        features = transformer.transform(None)

        assert features["round_count"] == 0

    def test_success_streak_multiple_streaks(
        self, transformer: BattleFeatureTransformer
    ) -> None:
        """Test success streak with multiple streaks."""
        battle = {
            "rounds": [
                {"detected": True, "evaded": False, "confidence": 0.9},
                {"detected": True, "evaded": False, "confidence": 0.9},
                {"detected": False, "evaded": True, "confidence": 0.3},
                {"detected": True, "evaded": False, "confidence": 0.9},
                {"detected": True, "evaded": False, "confidence": 0.9},
                {"detected": True, "evaded": False, "confidence": 0.9},
                {"detected": False, "evaded": True, "confidence": 0.3},
            ]
        }

        features = transformer.transform(battle)

        # Longest streak is 3 (rounds 4-6)
        assert features["success_streak"] == 3

    def test_missing_confidence(self, transformer: BattleFeatureTransformer) -> None:
        """Test rounds with missing confidence values."""
        battle = {
            "rounds": [
                {"detected": True, "evaded": False},
                {"detected": True, "evaded": False, "confidence": 0.9},
            ]
        }

        features = transformer.transform(battle)

        # Missing confidence defaults to 0
        assert features["avg_confidence"] == pytest.approx(0.45, rel=1e-3)

    def test_transform_batch(self, transformer: BattleFeatureTransformer) -> None:
        """Test batch transformation."""
        battles = [
            {"rounds": [{"detected": True, "evaded": False, "confidence": 0.9}]},
            {"rounds": [{"detected": False, "evaded": True, "confidence": 0.3}]},
            {"rounds": []},
        ]

        results = transformer.transform_batch(battles)

        assert len(results) == 3
        assert results[0]["detection_rate"] == 1.0
        assert results[1]["evasion_rate"] == 1.0
        assert results[2]["round_count"] == 0

    def test_get_feature_schema(self, transformer: BattleFeatureTransformer) -> None:
        """Test feature schema."""
        schema = transformer.get_feature_schema()

        assert schema["detection_rate"] == "float"
        assert schema["evasion_rate"] == "float"
        assert schema["avg_confidence"] == "float"
        assert schema["round_count"] == "int"
        assert schema["success_streak"] == "int"

    def test_repr(self, transformer: BattleFeatureTransformer) -> None:
        """Test string representation."""
        repr_str = repr(transformer)
        assert "BattleFeatureTransformer" in repr_str
        assert "battle" in repr_str
