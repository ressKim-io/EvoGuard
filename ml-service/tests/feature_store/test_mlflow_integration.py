"""Tests for MLflow Integration (Logger and Lineage)."""

from uuid import uuid4

import pytest

from ml_service.feature_store.mlflow_integration import (
    FeatureLineageTracker,
    FeatureLogger,
    FeatureSchemaValidator,
)
from ml_service.feature_store.mlflow_integration.lineage import (
    FeatureMissingError,
    FeatureTypeMismatchError,
)


class TestFeatureLogger:
    """Tests for FeatureLogger."""

    @pytest.fixture
    def logger(self) -> FeatureLogger:
        """Create a logger instance."""
        return FeatureLogger()

    def test_prepare_feature_metadata(self, logger: FeatureLogger) -> None:
        """Test preparing feature metadata."""
        metadata = logger.prepare_feature_metadata(
            feature_group="text_features",
            feature_group_version=1,
            feature_schema={"text_length": "int", "word_count": "int"},
        )

        assert "params" in metadata
        assert "schema" in metadata
        assert "tags" in metadata

        assert metadata["params"]["feature_group"] == "text_features"
        assert metadata["params"]["feature_group_version"] == "1"
        assert metadata["params"]["feature_count"] == "2"

        assert metadata["schema"]["feature_group"] == "text_features"
        assert "text_length" in metadata["schema"]["features"]

    def test_prepare_feature_metadata_with_optional_fields(
        self, logger: FeatureLogger
    ) -> None:
        """Test with optional fields."""
        group_id = uuid4()
        metadata = logger.prepare_feature_metadata(
            feature_group="text_features",
            feature_group_version=2,
            feature_schema={"text_length": "int"},
            feature_group_id=group_id,
            entity_type="text",
            transformer_version="1.0.0",
        )

        assert metadata["params"]["entity_type"] == "text"
        assert metadata["params"]["transformer_version"] == "1.0.0"
        assert metadata["params"]["feature_group_id"] == str(group_id)

    def test_prepare_training_features_info(self, logger: FeatureLogger) -> None:
        """Test preparing training features info."""
        info = logger.prepare_training_features_info(
            feature_group="text_features",
            feature_group_version=1,
            row_count=1000,
            entity_count=100,
            date_range=("2024-01-01", "2024-01-31"),
        )

        assert info["metrics"]["feature_row_count"] == 1000
        assert info["metrics"]["feature_entity_count"] == 100
        assert info["metrics"]["feature_date_span_days"] == 30

        assert info["artifact"]["row_count"] == 1000
        assert info["artifact"]["date_range"]["start"] == "2024-01-01"

    def test_prepare_feature_importance(self, logger: FeatureLogger) -> None:
        """Test preparing feature importance."""
        importance = logger.prepare_feature_importance(
            feature_names=["text_length", "word_count", "unicode_ratio"],
            importance_values=[0.5, 0.3, 0.2],
            method="permutation",
        )

        assert "metrics" in importance
        assert "artifact" in importance

        # Top feature should be text_length with 0.5
        assert importance["metrics"]["feature_importance_top1"] == 0.5
        assert importance["artifact"]["method"] == "permutation"

    def test_prepare_feature_importance_mismatch_length(
        self, logger: FeatureLogger
    ) -> None:
        """Test feature importance with mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            logger.prepare_feature_importance(
                feature_names=["a", "b"],
                importance_values=[0.5],
            )


class TestFeatureLineageTracker:
    """Tests for FeatureLineageTracker."""

    @pytest.fixture
    def tracker(self) -> FeatureLineageTracker:
        """Create a tracker instance."""
        return FeatureLineageTracker()

    def test_create_lineage_record(self, tracker: FeatureLineageTracker) -> None:
        """Test creating a lineage record."""
        group_id = uuid4()
        record = tracker.create_lineage_record(
            mlflow_run_id="run123",
            feature_group_id=group_id,
            feature_group_version=1,
            feature_schema={"text_length": "int", "word_count": "int"},
            mlflow_model_name="text_classifier",
            mlflow_model_version=1,
        )

        assert record["mlflow_run_id"] == "run123"
        assert record["mlflow_model_name"] == "text_classifier"
        assert record["mlflow_model_version"] == 1
        assert record["feature_group_id"] == str(group_id)
        assert record["feature_group_version"] == 1
        assert "feature_schema_snapshot" in record
        assert len(record["feature_schema_snapshot"]["features"]) == 2

    def test_create_lineage_record_with_used_features(
        self, tracker: FeatureLineageTracker
    ) -> None:
        """Test creating lineage record with specific used features."""
        record = tracker.create_lineage_record(
            mlflow_run_id="run123",
            feature_group_id=uuid4(),
            feature_group_version=1,
            feature_schema={"a": "int", "b": "int", "c": "int"},
            feature_names=["a", "b"],  # Only a and b used
        )

        assert record["feature_schema_snapshot"]["used_features"] == ["a", "b"]

    def test_extract_schema_from_lineage(
        self, tracker: FeatureLineageTracker
    ) -> None:
        """Test extracting schema from lineage record."""
        record = tracker.create_lineage_record(
            mlflow_run_id="run123",
            feature_group_id=uuid4(),
            feature_group_version=1,
            feature_schema={"text_length": "int", "word_count": "int"},
        )

        schema = tracker.extract_schema_from_lineage(record)

        assert schema == {"text_length": "int", "word_count": "int"}

    def test_get_used_features_explicit(
        self, tracker: FeatureLineageTracker
    ) -> None:
        """Test getting used features when explicitly specified."""
        record = tracker.create_lineage_record(
            mlflow_run_id="run123",
            feature_group_id=uuid4(),
            feature_group_version=1,
            feature_schema={"a": "int", "b": "int", "c": "int"},
            feature_names=["a", "c"],
        )

        used = tracker.get_used_features(record)

        assert used == ["a", "c"]

    def test_get_used_features_all(
        self, tracker: FeatureLineageTracker
    ) -> None:
        """Test getting used features when not specified (all features)."""
        record = tracker.create_lineage_record(
            mlflow_run_id="run123",
            feature_group_id=uuid4(),
            feature_group_version=1,
            feature_schema={"a": "int", "b": "int"},
        )

        used = tracker.get_used_features(record)

        assert set(used) == {"a", "b"}


class TestFeatureSchemaValidator:
    """Tests for FeatureSchemaValidator."""

    @pytest.fixture
    def validator(self) -> FeatureSchemaValidator:
        """Create a validator instance."""
        return FeatureSchemaValidator({
            "text_length": "int",
            "word_count": "int",
            "unicode_ratio": "float",
        })

    def test_validate_success(self, validator: FeatureSchemaValidator) -> None:
        """Test successful validation."""
        features = {
            "text_length": 100,
            "word_count": 20,
            "unicode_ratio": 0.15,
        }

        assert validator.validate(features) is True

    def test_validate_missing_feature_strict(
        self, validator: FeatureSchemaValidator
    ) -> None:
        """Test validation fails with missing feature in strict mode."""
        features = {
            "text_length": 100,
            # word_count missing
            "unicode_ratio": 0.15,
        }

        with pytest.raises(FeatureMissingError, match="word_count"):
            validator.validate(features, strict=True)

    def test_validate_missing_feature_non_strict(
        self, validator: FeatureSchemaValidator
    ) -> None:
        """Test validation returns False with missing feature in non-strict mode."""
        features = {
            "text_length": 100,
            "unicode_ratio": 0.15,
        }

        assert validator.validate(features, strict=False) is False

    def test_validate_type_mismatch_strict(
        self, validator: FeatureSchemaValidator
    ) -> None:
        """Test validation fails with type mismatch in strict mode."""
        features = {
            "text_length": "not an int",  # Wrong type
            "word_count": 20,
            "unicode_ratio": 0.15,
        }

        with pytest.raises(FeatureTypeMismatchError, match="text_length"):
            validator.validate(features, strict=True)

    def test_validate_type_compatibility(self) -> None:
        """Test type compatibility (int/float)."""
        validator = FeatureSchemaValidator({"value": "float"})

        # Int should be compatible with float
        assert validator.validate({"value": 10}) is True

    def test_validate_batch(self, validator: FeatureSchemaValidator) -> None:
        """Test batch validation."""
        batch = [
            {"text_length": 100, "word_count": 20, "unicode_ratio": 0.15},
            {"text_length": 200, "word_count": 40, "unicode_ratio": 0.1},
        ]

        results = validator.validate_batch(batch, strict=False)

        assert results == [True, True]

    def test_get_missing_features(self, validator: FeatureSchemaValidator) -> None:
        """Test getting missing features."""
        features = {"text_length": 100}

        missing = validator.get_missing_features(features)

        assert set(missing) == {"word_count", "unicode_ratio"}

    def test_get_extra_features(self, validator: FeatureSchemaValidator) -> None:
        """Test getting extra features."""
        features = {
            "text_length": 100,
            "word_count": 20,
            "unicode_ratio": 0.15,
            "extra_field": "value",
        }

        extra = validator.get_extra_features(features)

        assert extra == ["extra_field"]

    def test_from_lineage_record(self) -> None:
        """Test creating validator from lineage record."""
        tracker = FeatureLineageTracker()
        record = tracker.create_lineage_record(
            mlflow_run_id="run123",
            feature_group_id=uuid4(),
            feature_group_version=1,
            feature_schema={"a": "int", "b": "float"},
        )

        validator = FeatureSchemaValidator.from_lineage_record(record)

        assert validator.validate({"a": 1, "b": 0.5}) is True
