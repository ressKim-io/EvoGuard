"""Tests for alert rules."""


from ml_service.monitoring.alerts.rules import (
    Alert,
    AlertRuleRegistry,
    AlertSeverity,
    AlertStatus,
    CompositeRule,
    DriftRule,
    ThresholdRule,
    create_default_rules,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self) -> None:
        """Test alert creation with all fields."""
        alert = Alert(
            name="TestAlert",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test message",
            model_name="test_model",
            metric_name="f1_score",
            metric_value=0.65,
            threshold=0.7,
        )

        assert alert.name == "TestAlert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING
        assert alert.message == "Test message"
        assert alert.model_name == "test_model"
        assert alert.metric_name == "f1_score"
        assert alert.metric_value == 0.65
        assert alert.threshold == 0.7

    def test_alert_to_dict(self) -> None:
        """Test alert serialization to dictionary."""
        alert = Alert(
            name="TestAlert",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Critical alert",
            model_name="test_model",
            metric_name="f1_score",
            metric_value=0.5,
            threshold=0.7,
            labels={"env": "production"},
            annotations={"description": "Test description"},
        )

        result = alert.to_dict()

        assert result["name"] == "TestAlert"
        assert result["severity"] == "critical"
        assert result["status"] == "firing"
        assert result["message"] == "Critical alert"
        assert result["labels"] == {"env": "production"}
        assert result["annotations"] == {"description": "Test description"}


class TestThresholdRule:
    """Tests for ThresholdRule."""

    def test_threshold_rule_gt_fires(self) -> None:
        """Test threshold rule fires when value exceeds threshold."""
        rule = ThresholdRule(
            name="HighLatency",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="latency",
            threshold=1.0,
            comparison="gt",
        )

        alert = rule.evaluate({"latency": 1.5})

        assert alert is not None
        assert alert.name == "HighLatency"
        assert alert.metric_value == 1.5
        assert rule.is_firing

    def test_threshold_rule_gt_does_not_fire(self) -> None:
        """Test threshold rule does not fire when value is below threshold."""
        rule = ThresholdRule(
            name="HighLatency",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="latency",
            threshold=1.0,
            comparison="gt",
        )

        alert = rule.evaluate({"latency": 0.5})

        assert alert is None
        assert not rule.is_firing

    def test_threshold_rule_lt_fires(self) -> None:
        """Test threshold rule fires when value is below threshold (lt comparison)."""
        rule = ThresholdRule(
            name="LowF1",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            metric_name="f1_score",
            threshold=0.7,
            comparison="lt",
        )

        alert = rule.evaluate({"f1_score": 0.65})

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "below" in alert.message

    def test_threshold_rule_gte_fires(self) -> None:
        """Test threshold rule with gte comparison."""
        rule = ThresholdRule(
            name="TestGte",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="gte",
        )

        alert = rule.evaluate({"value": 1.0})
        assert alert is not None

        alert = rule.evaluate({"value": 1.5})
        assert alert is not None

    def test_threshold_rule_lte_fires(self) -> None:
        """Test threshold rule with lte comparison."""
        rule = ThresholdRule(
            name="TestLte",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="lte",
        )

        alert = rule.evaluate({"value": 1.0})
        assert alert is not None

        alert = rule.evaluate({"value": 0.5})
        assert alert is not None

    def test_threshold_rule_eq_fires(self) -> None:
        """Test threshold rule with eq comparison."""
        rule = ThresholdRule(
            name="TestEq",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="eq",
        )

        alert = rule.evaluate({"value": 1.0})
        assert alert is not None

    def test_threshold_rule_missing_metric(self) -> None:
        """Test threshold rule returns None when metric is missing."""
        rule = ThresholdRule(
            name="TestRule",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="missing_metric",
            threshold=1.0,
            comparison="gt",
        )

        alert = rule.evaluate({"other_metric": 1.5})

        assert alert is None

    def test_threshold_rule_for_duration(self) -> None:
        """Test threshold rule with for_duration (first evaluation should not fire)."""
        rule = ThresholdRule(
            name="TestDuration",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="gt",
            for_duration_seconds=60,
        )

        # First evaluation - starts timer but doesn't fire
        alert = rule.evaluate({"value": 1.5})
        assert alert is None  # Duration not met yet

    def test_threshold_rule_resets_on_recovery(self) -> None:
        """Test threshold rule resets when condition is no longer met."""
        rule = ThresholdRule(
            name="TestReset",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="gt",
        )

        # Fire the rule
        rule.evaluate({"value": 1.5})
        assert rule.is_firing

        # Value returns to normal
        rule.evaluate({"value": 0.5})
        assert not rule.is_firing


class TestDriftRule:
    """Tests for DriftRule."""

    def test_data_drift_rule_fires(self) -> None:
        """Test data drift rule fires when PSI exceeds threshold."""
        rule = DriftRule(
            name="DataDrift",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            drift_type="data",
            psi_threshold=0.2,
        )

        alert = rule.evaluate({"data_drift_psi": 0.25})

        assert alert is not None
        assert "Data drift detected" in alert.message
        assert alert.metric_value == 0.25

    def test_data_drift_rule_does_not_fire(self) -> None:
        """Test data drift rule does not fire when PSI is below threshold."""
        rule = DriftRule(
            name="DataDrift",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            drift_type="data",
            psi_threshold=0.2,
        )

        alert = rule.evaluate({"data_drift_psi": 0.15})

        assert alert is None

    def test_concept_drift_rule_fires(self) -> None:
        """Test concept drift rule fires when F1 drop exceeds threshold."""
        rule = DriftRule(
            name="ConceptDrift",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            drift_type="concept",
            f1_drop_threshold=0.05,
        )

        alert = rule.evaluate({"f1_drop": 0.08})

        assert alert is not None
        assert "Concept drift detected" in alert.message
        assert "moderate" in alert.message

    def test_concept_drift_critical_severity(self) -> None:
        """Test concept drift with critical F1 drop."""
        rule = DriftRule(
            name="ConceptDrift",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            drift_type="concept",
            f1_drop_threshold=0.05,
        )

        alert = rule.evaluate({"f1_drop": 0.20})

        assert alert is not None
        assert "critical" in alert.message

    def test_concept_drift_significant_severity(self) -> None:
        """Test concept drift with significant F1 drop."""
        rule = DriftRule(
            name="ConceptDrift",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            drift_type="concept",
            f1_drop_threshold=0.05,
        )

        alert = rule.evaluate({"f1_drop": 0.12})

        assert alert is not None
        assert "significant" in alert.message

    def test_feature_drift_rule_fires(self) -> None:
        """Test feature drift rule fires when features drift."""
        rule = DriftRule(
            name="FeatureDrift",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            drift_type="feature",
            psi_threshold=0.2,
        )

        alert = rule.evaluate({
            "feature_drift_text_length": 0.25,
            "feature_drift_word_count": 0.30,
        })

        assert alert is not None
        assert "Feature drift detected" in alert.message
        assert "2 features" in alert.message

    def test_feature_drift_rule_no_drift(self) -> None:
        """Test feature drift rule does not fire when features are stable."""
        rule = DriftRule(
            name="FeatureDrift",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            drift_type="feature",
            psi_threshold=0.2,
        )

        alert = rule.evaluate({
            "feature_drift_text_length": 0.1,
            "feature_drift_word_count": 0.05,
        })

        assert alert is None

    def test_drift_rule_missing_metric(self) -> None:
        """Test drift rule returns None when metric is missing."""
        rule = DriftRule(
            name="DataDrift",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            drift_type="data",
        )

        alert = rule.evaluate({"other_metric": 1.0})

        assert alert is None


class TestCompositeRule:
    """Tests for CompositeRule."""

    def test_composite_rule_and_fires(self) -> None:
        """Test composite rule with AND operator fires when all rules fire."""
        rule1 = ThresholdRule(
            name="Rule1",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="Rule2",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        composite = CompositeRule(
            name="CompositeAnd",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            rules=[rule1, rule2],
            operator="and",
        )

        alert = composite.evaluate({"metric1": 1.5, "metric2": 2.5})

        assert alert is not None
        assert "Multiple conditions met" in alert.message

    def test_composite_rule_and_does_not_fire(self) -> None:
        """Test composite rule with AND operator does not fire when only one rule fires."""
        rule1 = ThresholdRule(
            name="Rule1",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="Rule2",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        composite = CompositeRule(
            name="CompositeAnd",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            rules=[rule1, rule2],
            operator="and",
        )

        alert = composite.evaluate({"metric1": 1.5, "metric2": 1.5})

        assert alert is None

    def test_composite_rule_or_fires(self) -> None:
        """Test composite rule with OR operator fires when any rule fires."""
        rule1 = ThresholdRule(
            name="Rule1",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="Rule2",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        composite = CompositeRule(
            name="CompositeOr",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            rules=[rule1, rule2],
            operator="or",
        )

        alert = composite.evaluate({"metric1": 1.5, "metric2": 1.5})

        assert alert is not None

    def test_composite_rule_or_does_not_fire(self) -> None:
        """Test composite rule with OR operator does not fire when no rules fire."""
        rule1 = ThresholdRule(
            name="Rule1",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="Rule2",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        composite = CompositeRule(
            name="CompositeOr",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            rules=[rule1, rule2],
            operator="or",
        )

        alert = composite.evaluate({"metric1": 0.5, "metric2": 1.5})

        assert alert is None

    def test_composite_rule_uses_highest_severity(self) -> None:
        """Test composite rule uses highest severity from child alerts."""
        rule1 = ThresholdRule(
            name="Rule1",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="Rule2",
            severity=AlertSeverity.CRITICAL,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        composite = CompositeRule(
            name="CompositeOr",
            severity=AlertSeverity.INFO,
            model_name="test_model",
            rules=[rule1, rule2],
            operator="or",
        )

        alert = composite.evaluate({"metric1": 1.5, "metric2": 2.5})

        # Should use CRITICAL from rule2
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL


class TestAlertRuleRegistry:
    """Tests for AlertRuleRegistry."""

    def test_register_and_get_rule(self) -> None:
        """Test registering and getting a rule."""
        registry = AlertRuleRegistry()
        rule = ThresholdRule(
            name="TestRule",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="gt",
        )

        registry.register(rule)
        retrieved = registry.get_rule("TestRule")

        assert retrieved is rule

    def test_unregister_rule(self) -> None:
        """Test unregistering a rule."""
        registry = AlertRuleRegistry()
        rule = ThresholdRule(
            name="TestRule",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="gt",
        )

        registry.register(rule)
        assert registry.unregister("TestRule")
        assert registry.get_rule("TestRule") is None

    def test_unregister_nonexistent_rule(self) -> None:
        """Test unregistering a nonexistent rule returns False."""
        registry = AlertRuleRegistry()
        assert not registry.unregister("NonexistentRule")

    def test_evaluate_all(self) -> None:
        """Test evaluating all registered rules."""
        registry = AlertRuleRegistry()

        rule1 = ThresholdRule(
            name="Rule1",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="Rule2",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        registry.register(rule1)
        registry.register(rule2)

        alerts = registry.evaluate_all({"metric1": 1.5, "metric2": 2.5})

        assert len(alerts) == 2

    def test_get_firing_rules(self) -> None:
        """Test getting list of firing rules."""
        registry = AlertRuleRegistry()

        rule1 = ThresholdRule(
            name="FiringRule",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric1",
            threshold=1.0,
            comparison="gt",
        )
        rule2 = ThresholdRule(
            name="NotFiringRule",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="metric2",
            threshold=2.0,
            comparison="gt",
        )

        registry.register(rule1)
        registry.register(rule2)

        registry.evaluate_all({"metric1": 1.5, "metric2": 1.5})

        firing = registry.get_firing_rules()

        assert "FiringRule" in firing
        assert "NotFiringRule" not in firing

    def test_list_rules(self) -> None:
        """Test listing all registered rules."""
        registry = AlertRuleRegistry()

        rule = ThresholdRule(
            name="TestRule",
            severity=AlertSeverity.WARNING,
            model_name="test_model",
            metric_name="value",
            threshold=1.0,
            comparison="gt",
            description="Test description",
        )

        registry.register(rule)
        rules_list = registry.list_rules()

        assert len(rules_list) == 1
        assert rules_list[0]["name"] == "TestRule"
        assert rules_list[0]["severity"] == "warning"
        assert rules_list[0]["description"] == "Test description"


class TestCreateDefaultRules:
    """Tests for create_default_rules function."""

    def test_create_default_rules(self) -> None:
        """Test creating default rules for a model."""
        rules = create_default_rules("test_model")

        assert len(rules) == 7

        rule_names = [r.name for r in rules]
        assert "ModelF1ScoreCritical" in rule_names
        assert "ModelF1ScoreWarning" in rule_names
        assert "LowConfidenceSpike" in rule_names
        assert "DataDriftDetected" in rule_names
        assert "ConceptDriftDetected" in rule_names
        assert "FeatureDriftDetected" in rule_names
        assert "HighPredictionLatency" in rule_names

    def test_default_rules_model_name(self) -> None:
        """Test default rules have correct model name."""
        rules = create_default_rules("my_model")

        for rule in rules:
            assert rule.model_name == "my_model"
