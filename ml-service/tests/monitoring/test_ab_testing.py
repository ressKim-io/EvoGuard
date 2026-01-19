"""Tests for A/B testing module."""

import pytest

from ml_service.monitoring.experimental.ab_testing import (
    ABExperiment,
    ExperimentManager,
    ExperimentResult,
    ExperimentStatus,
    StatisticalSignificance,
    StatisticalTestResult,
    TrafficSplitter,
    VariantAssignment,
    VariantMetrics,
)


class TestVariantMetrics:
    """Test VariantMetrics dataclass."""

    def test_initial_values(self) -> None:
        """Test default values."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.total_predictions == 0
        assert metrics.correct_predictions == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.total_confidence == 0.0

    def test_accuracy_no_predictions(self) -> None:
        """Test accuracy with no predictions."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.accuracy == 0.0

    def test_accuracy_with_predictions(self) -> None:
        """Test accuracy calculation."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            total_predictions=100,
            correct_predictions=80,
        )
        assert metrics.accuracy == 0.8

    def test_precision_no_positives(self) -> None:
        """Test precision with no positive predictions."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.precision == 0.0

    def test_precision_with_positives(self) -> None:
        """Test precision calculation."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            true_positives=80,
            false_positives=20,
        )
        assert metrics.precision == 0.8

    def test_recall_no_positives(self) -> None:
        """Test recall with no actual positives."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.recall == 0.0

    def test_recall_with_positives(self) -> None:
        """Test recall calculation."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            true_positives=80,
            false_negatives=20,
        )
        assert metrics.recall == 0.8

    def test_f1_score_no_data(self) -> None:
        """Test F1 with no data."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.f1_score == 0.0

    def test_f1_score_with_data(self) -> None:
        """Test F1 score calculation."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            true_positives=80,
            false_positives=10,
            false_negatives=10,
        )
        # precision = 80/90 = 0.888...
        # recall = 80/90 = 0.888...
        # f1 = 2 * 0.888 * 0.888 / (0.888 + 0.888) = 0.888
        assert 0.88 < metrics.f1_score < 0.90

    def test_mean_latency_no_predictions(self) -> None:
        """Test mean latency with no predictions."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.mean_latency_ms == 0.0

    def test_mean_latency_with_predictions(self) -> None:
        """Test mean latency calculation."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            total_predictions=100,
            total_latency_ms=5000.0,
        )
        assert metrics.mean_latency_ms == 50.0

    def test_mean_confidence_no_predictions(self) -> None:
        """Test mean confidence with no predictions."""
        metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        assert metrics.mean_confidence == 0.0

    def test_mean_confidence_with_predictions(self) -> None:
        """Test mean confidence calculation."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            total_predictions=100,
            total_confidence=85.0,
        )
        assert metrics.mean_confidence == 0.85

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = VariantMetrics(
            variant=VariantAssignment.CHAMPION,
            total_predictions=100,
            correct_predictions=80,
        )
        result = metrics.to_dict()
        assert result["variant"] == "champion"
        assert result["total_predictions"] == 100
        assert result["accuracy"] == 0.8


class TestTrafficSplitter:
    """Test TrafficSplitter class."""

    def test_default_ratio(self) -> None:
        """Test default challenger ratio."""
        splitter = TrafficSplitter()
        assert splitter.challenger_ratio == 0.1

    def test_custom_ratio(self) -> None:
        """Test custom challenger ratio."""
        splitter = TrafficSplitter(challenger_ratio=0.3)
        assert splitter.challenger_ratio == 0.3

    def test_invalid_ratio_negative(self) -> None:
        """Test that negative ratio raises error."""
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            TrafficSplitter(challenger_ratio=-0.1)

    def test_invalid_ratio_over_one(self) -> None:
        """Test that ratio > 1 raises error."""
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            TrafficSplitter(challenger_ratio=1.5)

    def test_consistent_assignment(self) -> None:
        """Test that same identifier gets same variant."""
        splitter = TrafficSplitter(challenger_ratio=0.5)
        variant1 = splitter.assign("user_123")
        variant2 = splitter.assign("user_123")
        assert variant1 == variant2

    def test_different_identifiers_vary(self) -> None:
        """Test that different identifiers can get different variants."""
        splitter = TrafficSplitter(challenger_ratio=0.5)
        variants = set()
        for i in range(100):
            variants.add(splitter.assign(f"user_{i}"))
        # With 50% ratio, we should see both variants
        assert len(variants) == 2

    def test_update_ratio(self) -> None:
        """Test updating the challenger ratio."""
        splitter = TrafficSplitter(challenger_ratio=0.1)
        splitter.update_ratio(0.5)
        assert splitter.challenger_ratio == 0.5

    def test_update_ratio_invalid(self) -> None:
        """Test that invalid ratio update raises error."""
        splitter = TrafficSplitter()
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            splitter.update_ratio(1.5)

    def test_zero_ratio_always_champion(self) -> None:
        """Test that 0 ratio always returns champion."""
        splitter = TrafficSplitter(challenger_ratio=0.0)
        for i in range(100):
            assert splitter.assign(f"user_{i}") == VariantAssignment.CHAMPION

    def test_full_ratio_always_challenger(self) -> None:
        """Test that 1.0 ratio always returns challenger."""
        splitter = TrafficSplitter(challenger_ratio=1.0)
        for i in range(100):
            assert splitter.assign(f"user_{i}") == VariantAssignment.CHALLENGER


class TestStatisticalSignificance:
    """Test StatisticalSignificance class."""

    def test_z_test_no_samples(self) -> None:
        """Test z-test with no samples."""
        result = StatisticalSignificance.z_test_proportions(
            p1=0.5, n1=0, p2=0.6, n2=0
        )
        assert result.is_significant is False
        assert result.p_value == 1.0
        assert "Insufficient" in result.message

    def test_z_test_significant_difference(self) -> None:
        """Test z-test with significant difference."""
        # Large sample, clear difference
        result = StatisticalSignificance.z_test_proportions(
            p1=0.50, n1=10000, p2=0.55, n2=10000
        )
        assert result.test_name == "two_proportion_z_test"
        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.challenger_better is True

    def test_z_test_no_significant_difference(self) -> None:
        """Test z-test with no significant difference."""
        # Small sample, small difference
        result = StatisticalSignificance.z_test_proportions(
            p1=0.50, n1=50, p2=0.52, n2=50
        )
        assert result.is_significant is False
        assert result.p_value > 0.05

    def test_z_test_champion_better(self) -> None:
        """Test z-test when champion is better."""
        result = StatisticalSignificance.z_test_proportions(
            p1=0.60, n1=10000, p2=0.50, n2=10000
        )
        assert result.is_significant is True
        assert result.challenger_better is False
        assert "Champion" in result.message

    def test_z_test_no_variance(self) -> None:
        """Test z-test with no variance (all same outcome)."""
        result = StatisticalSignificance.z_test_proportions(
            p1=0.0, n1=100, p2=0.0, n2=100
        )
        assert result.is_significant is False
        assert "no variance" in result.message

    def test_minimum_sample_size(self) -> None:
        """Test minimum sample size calculation."""
        n = StatisticalSignificance.minimum_sample_size(
            baseline_rate=0.5,
            minimum_detectable_effect=0.05,
        )
        assert n >= 100  # Minimum enforced
        assert isinstance(n, int)

    def test_minimum_sample_size_small_effect(self) -> None:
        """Test that smaller effect requires larger sample."""
        n_small = StatisticalSignificance.minimum_sample_size(
            baseline_rate=0.5, minimum_detectable_effect=0.01
        )
        n_large = StatisticalSignificance.minimum_sample_size(
            baseline_rate=0.5, minimum_detectable_effect=0.1
        )
        assert n_small > n_large


class TestABExperiment:
    """Test ABExperiment class."""

    def test_create_experiment(self) -> None:
        """Test experiment creation."""
        exp = ABExperiment(experiment_id="test_exp")
        assert exp.experiment_id == "test_exp"
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.challenger_ratio == 0.1
        assert exp.started_at is None

    def test_start_experiment(self) -> None:
        """Test starting an experiment."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None

    def test_start_already_running(self) -> None:
        """Test that starting a running experiment raises error."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        with pytest.raises(ValueError, match="Cannot start"):
            exp.start()

    def test_pause_experiment(self) -> None:
        """Test pausing an experiment."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        exp.pause()
        assert exp.status == ExperimentStatus.PAUSED

    def test_pause_not_running(self) -> None:
        """Test that pausing non-running experiment raises error."""
        exp = ABExperiment(experiment_id="test_exp")
        with pytest.raises(ValueError, match="only pause running"):
            exp.pause()

    def test_resume_experiment(self) -> None:
        """Test resuming an experiment."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        exp.pause()
        exp.resume()
        assert exp.status == ExperimentStatus.RUNNING

    def test_resume_not_paused(self) -> None:
        """Test that resuming non-paused experiment raises error."""
        exp = ABExperiment(experiment_id="test_exp")
        with pytest.raises(ValueError, match="only resume paused"):
            exp.resume()

    def test_stop_experiment(self) -> None:
        """Test stopping an experiment."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        exp.stop()
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.ended_at is not None

    def test_stop_with_cancelled_status(self) -> None:
        """Test stopping with cancelled status."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        exp.stop(status=ExperimentStatus.CANCELLED)
        assert exp.status == ExperimentStatus.CANCELLED

    def test_stop_not_running(self) -> None:
        """Test that stopping non-running experiment raises error."""
        exp = ABExperiment(experiment_id="test_exp")
        with pytest.raises(ValueError, match="only stop running"):
            exp.stop()

    def test_assign_variant_not_running(self) -> None:
        """Test variant assignment when not running defaults to champion."""
        exp = ABExperiment(experiment_id="test_exp")
        variant = exp.assign_variant("user_123")
        assert variant == VariantAssignment.CHAMPION

    def test_assign_variant_running(self) -> None:
        """Test variant assignment when running."""
        exp = ABExperiment(experiment_id="test_exp", challenger_ratio=0.5)
        exp.start()
        variants = set()
        for i in range(100):
            variants.add(exp.assign_variant(f"user_{i}"))
        # Should see both variants with 50% split
        assert len(variants) == 2

    def test_record_prediction_not_running(self) -> None:
        """Test that recording prediction when not running is ignored."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
        )
        result = exp.get_result()
        assert result.champion_metrics.total_predictions == 0

    def test_record_prediction_running(self) -> None:
        """Test recording predictions when running."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
            actual_label=1,
        )
        result = exp.get_result()
        assert result.champion_metrics.total_predictions == 1
        assert result.champion_metrics.correct_predictions == 1
        assert result.champion_metrics.true_positives == 1

    def test_record_prediction_confusion_matrix(self) -> None:
        """Test confusion matrix updates."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()

        # True positive
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
            actual_label=1,
        )
        # False positive
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
            actual_label=0,
        )
        # True negative
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=0,
            confidence=0.9,
            latency_ms=50.0,
            actual_label=0,
        )
        # False negative
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=0,
            confidence=0.9,
            latency_ms=50.0,
            actual_label=1,
        )

        result = exp.get_result()
        assert result.champion_metrics.true_positives == 1
        assert result.champion_metrics.false_positives == 1
        assert result.champion_metrics.true_negatives == 1
        assert result.champion_metrics.false_negatives == 1

    def test_get_statistical_result_insufficient_data(self) -> None:
        """Test statistical result with insufficient data."""
        exp = ABExperiment(experiment_id="test_exp")
        exp.start()
        result = exp.get_statistical_result()
        assert result is None

    def test_get_statistical_result_with_data(self) -> None:
        """Test statistical result with sufficient data."""
        exp = ABExperiment(experiment_id="test_exp", min_samples_per_variant=30)
        exp.start()

        # Add enough predictions
        for i in range(50):
            exp.record_prediction(
                variant=VariantAssignment.CHAMPION,
                prediction=1 if i < 40 else 0,
                confidence=0.9,
                latency_ms=50.0,
                actual_label=1 if i < 40 else 0,
            )
            exp.record_prediction(
                variant=VariantAssignment.CHALLENGER,
                prediction=1 if i < 35 else 0,
                confidence=0.9,
                latency_ms=50.0,
                actual_label=1 if i < 35 else 0,
            )

        result = exp.get_statistical_result()
        assert result is not None
        assert result.test_name == "two_proportion_z_test"

    def test_get_result(self) -> None:
        """Test getting experiment result."""
        exp = ABExperiment(experiment_id="test_exp")
        result = exp.get_result()
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == "test_exp"
        assert result.status == ExperimentStatus.DRAFT

    def test_get_result_recommendation_not_started(self) -> None:
        """Test recommendation when not started."""
        exp = ABExperiment(experiment_id="test_exp")
        result = exp.get_result()
        assert "not started" in result.recommendation

    def test_get_result_recommendation_need_samples(self) -> None:
        """Test recommendation when needing samples."""
        exp = ABExperiment(experiment_id="test_exp", min_samples_per_variant=100)
        exp.start()
        exp.record_prediction(
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
        )
        result = exp.get_result()
        assert "Need more" in result.recommendation

    def test_get_status(self) -> None:
        """Test getting experiment status."""
        exp = ABExperiment(experiment_id="test_exp", challenger_ratio=0.2)
        exp.start()
        status = exp.get_status()
        assert status["experiment_id"] == "test_exp"
        assert status["status"] == "running"
        assert status["challenger_ratio"] == 0.2

    def test_update_traffic_ratio(self) -> None:
        """Test updating traffic ratio."""
        exp = ABExperiment(experiment_id="test_exp", challenger_ratio=0.1)
        exp.update_traffic_ratio(0.5)
        assert exp.challenger_ratio == 0.5


class TestExperimentManager:
    """Test ExperimentManager class."""

    def test_create_experiment(self) -> None:
        """Test creating an experiment."""
        manager = ExperimentManager()
        exp = manager.create_experiment("exp_001")
        assert exp.experiment_id == "exp_001"
        assert manager.get_experiment("exp_001") is not None

    def test_create_duplicate_experiment(self) -> None:
        """Test that creating duplicate experiment raises error."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_experiment("exp_001")

    def test_get_experiment_not_found(self) -> None:
        """Test getting non-existent experiment."""
        manager = ExperimentManager()
        assert manager.get_experiment("nonexistent") is None

    def test_start_experiment(self) -> None:
        """Test starting an experiment via manager."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001")
        manager.start_experiment("exp_001")
        exp = manager.get_experiment("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.RUNNING

    def test_start_nonexistent_experiment(self) -> None:
        """Test starting non-existent experiment raises error."""
        manager = ExperimentManager()
        with pytest.raises(ValueError, match="not found"):
            manager.start_experiment("nonexistent")

    def test_stop_experiment(self) -> None:
        """Test stopping an experiment via manager."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001")
        manager.start_experiment("exp_001")
        manager.stop_experiment("exp_001")
        exp = manager.get_experiment("exp_001")
        assert exp is not None
        assert exp.status == ExperimentStatus.COMPLETED

    def test_stop_nonexistent_experiment(self) -> None:
        """Test stopping non-existent experiment raises error."""
        manager = ExperimentManager()
        with pytest.raises(ValueError, match="not found"):
            manager.stop_experiment("nonexistent")

    def test_assign_variant(self) -> None:
        """Test assigning variant via manager."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001", challenger_ratio=0.5)
        manager.start_experiment("exp_001")
        variant = manager.assign_variant("exp_001", "user_123")
        assert variant in (VariantAssignment.CHAMPION, VariantAssignment.CHALLENGER)

    def test_assign_variant_nonexistent(self) -> None:
        """Test assigning variant for non-existent experiment."""
        manager = ExperimentManager()
        variant = manager.assign_variant("nonexistent", "user_123")
        assert variant == VariantAssignment.CHAMPION

    def test_record_prediction(self) -> None:
        """Test recording prediction via manager."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001")
        manager.start_experiment("exp_001")
        manager.record_prediction(
            experiment_id="exp_001",
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
        )
        exp = manager.get_experiment("exp_001")
        assert exp is not None
        result = exp.get_result()
        assert result.champion_metrics.total_predictions == 1

    def test_record_prediction_nonexistent(self) -> None:
        """Test recording prediction for non-existent experiment is ignored."""
        manager = ExperimentManager()
        # Should not raise
        manager.record_prediction(
            experiment_id="nonexistent",
            variant=VariantAssignment.CHAMPION,
            prediction=1,
            confidence=0.9,
            latency_ms=50.0,
        )

    def test_get_all_results(self) -> None:
        """Test getting all experiment results."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001")
        manager.create_experiment("exp_002")
        results = manager.get_all_results()
        assert "exp_001" in results
        assert "exp_002" in results

    def test_get_running_experiments(self) -> None:
        """Test getting running experiments."""
        manager = ExperimentManager()
        manager.create_experiment("exp_001")
        manager.create_experiment("exp_002")
        manager.start_experiment("exp_001")
        running = manager.get_running_experiments()
        assert "exp_001" in running
        assert "exp_002" not in running

    def test_cleanup_completed(self) -> None:
        """Test cleaning up completed experiments."""
        manager = ExperimentManager()
        for i in range(15):
            manager.create_experiment(f"exp_{i:03d}")
            manager.start_experiment(f"exp_{i:03d}")
            manager.stop_experiment(f"exp_{i:03d}")

        removed = manager.cleanup_completed(keep_last=5)
        assert removed == 10
        results = manager.get_all_results()
        assert len(results) == 5


class TestExperimentResult:
    """Test ExperimentResult class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from datetime import UTC, datetime

        result = ExperimentResult(
            experiment_id="test_exp",
            status=ExperimentStatus.RUNNING,
            champion_metrics=VariantMetrics(variant=VariantAssignment.CHAMPION),
            challenger_metrics=VariantMetrics(variant=VariantAssignment.CHALLENGER),
            statistical_test=None,
            recommendation="Continue testing",
            started_at=datetime.now(UTC),
        )
        data = result.to_dict()
        assert data["experiment_id"] == "test_exp"
        assert data["status"] == "running"
        assert data["statistical_test"] is None

    def test_to_dict_with_statistical_test(self) -> None:
        """Test conversion with statistical test result."""
        from datetime import UTC, datetime

        stat_result = StatisticalTestResult(
            test_name="z_test",
            is_significant=True,
            p_value=0.01,
            confidence_level=0.95,
            effect_size=0.05,
            challenger_better=True,
            message="Significant",
        )
        result = ExperimentResult(
            experiment_id="test_exp",
            status=ExperimentStatus.COMPLETED,
            champion_metrics=VariantMetrics(variant=VariantAssignment.CHAMPION),
            challenger_metrics=VariantMetrics(variant=VariantAssignment.CHALLENGER),
            statistical_test=stat_result,
            recommendation="Promote challenger",
            started_at=datetime.now(UTC),
        )
        data = result.to_dict()
        assert data["statistical_test"]["is_significant"] is True
        assert data["statistical_test"]["p_value"] == 0.01


class TestIntegration:
    """Integration tests for A/B testing workflow."""

    def test_full_experiment_workflow(self) -> None:
        """Test complete experiment workflow."""
        manager = ExperimentManager()

        # Create experiment
        exp = manager.create_experiment(
            experiment_id="full_test",
            challenger_ratio=0.3,
            min_samples_per_variant=50,
            auto_stop=False,  # Disable auto-stop for test control
        )

        # Start experiment
        manager.start_experiment("full_test")

        # Simulate traffic
        for i in range(100):
            variant = manager.assign_variant("full_test", f"user_{i}")
            # Champion has 80% accuracy, Challenger has 85%
            is_champion = variant == VariantAssignment.CHAMPION
            prediction = 1
            actual = 1 if (i % 10 < (8 if is_champion else 8.5)) else 0

            manager.record_prediction(
                experiment_id="full_test",
                variant=variant,
                prediction=prediction,
                confidence=0.9,
                latency_ms=50.0,
                actual_label=actual,
            )

        # Get results
        results = manager.get_all_results()
        assert "full_test" in results

        # Stop experiment
        manager.stop_experiment("full_test")

        exp = manager.get_experiment("full_test")
        assert exp is not None
        assert exp.status == ExperimentStatus.COMPLETED
