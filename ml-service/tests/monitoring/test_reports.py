"""Tests for reports module."""

import pytest

from ml_service.monitoring.reports.generator import (
    ActionReportData,
    ActionReportGenerator,
    DriftReportData,
    DriftReportGenerator,
    ExperimentReportData,
    ExperimentReportGenerator,
    MetricSummary,
    MonitoringReportGenerator,
    PerformanceReportData,
    PerformanceReportGenerator,
    ReportBuilder,
    ReportFormat,
    ReportSection,
    ReportType,
    TimeRange,
)


class TestTimeRange:
    """Test TimeRange class."""

    def test_last_hours(self) -> None:
        """Test creating time range for last N hours."""
        time_range = TimeRange.last_hours(24)
        assert time_range.end > time_range.start
        diff = time_range.end - time_range.start
        assert 23 * 3600 < diff.total_seconds() < 25 * 3600

    def test_last_days(self) -> None:
        """Test creating time range for last N days."""
        time_range = TimeRange.last_days(7)
        assert time_range.end > time_range.start
        diff = time_range.end - time_range.start
        assert 6 * 86400 < diff.total_seconds() < 8 * 86400

    def test_today(self) -> None:
        """Test creating time range for today."""
        time_range = TimeRange.today()
        assert time_range.start.hour == 0
        assert time_range.start.minute == 0
        assert time_range.start.second == 0


class TestMetricSummary:
    """Test MetricSummary class."""

    def test_from_empty_values(self) -> None:
        """Test creating summary from empty values."""
        summary = MetricSummary.from_values("test_metric", [])
        assert summary is None

    def test_from_values(self) -> None:
        """Test creating summary from values."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        summary = MetricSummary.from_values("latency_ms", values)
        assert summary is not None
        assert summary.name == "latency_ms"
        assert summary.count == 5
        assert summary.mean == 30.0
        assert summary.min_val == 10.0
        assert summary.max_val == 50.0

    def test_from_single_value(self) -> None:
        """Test creating summary from single value."""
        summary = MetricSummary.from_values("test", [42.0])
        assert summary is not None
        assert summary.count == 1
        assert summary.mean == 42.0
        assert summary.std == 0.0  # No std for single value


class TestReportSection:
    """Test ReportSection class."""

    def test_create_section(self) -> None:
        """Test creating a section."""
        section = ReportSection("Title", "Content")
        assert section.title == "Title"
        assert section.content == "Content"
        assert section.level == 2

    def test_add_subsection(self) -> None:
        """Test adding subsections."""
        section = ReportSection("Parent", "Parent content")
        subsection = section.add_subsection("Child", "Child content")
        assert len(section.subsections) == 1
        assert subsection.title == "Child"
        assert subsection.level == 3


class TestReportBuilder:
    """Test ReportBuilder class."""

    def test_create_builder(self) -> None:
        """Test creating a report builder."""
        builder = ReportBuilder("Test Report", ReportType.PERFORMANCE)
        assert builder.title == "Test Report"
        assert builder.report_type == ReportType.PERFORMANCE

    def test_add_section(self) -> None:
        """Test adding sections."""
        builder = ReportBuilder("Test", ReportType.PERFORMANCE)
        section = builder.add_section("Section 1", "Content 1")
        assert len(builder.sections) == 1
        assert section.title == "Section 1"

    def test_add_metadata(self) -> None:
        """Test adding metadata."""
        builder = ReportBuilder("Test", ReportType.PERFORMANCE)
        builder.add_metadata("custom_key", "custom_value")
        assert builder.metadata["custom_key"] == "custom_value"

    def test_render_markdown(self) -> None:
        """Test rendering to Markdown."""
        builder = ReportBuilder("Test Report", ReportType.PERFORMANCE)
        builder.add_section("Overview", "This is the overview.")
        result = builder.render(ReportFormat.MARKDOWN)
        assert "# Test Report" in result
        assert "## Overview" in result
        assert "This is the overview." in result

    def test_render_html(self) -> None:
        """Test rendering to HTML."""
        builder = ReportBuilder("Test Report", ReportType.PERFORMANCE)
        builder.add_section("Overview", "This is the overview.")
        result = builder.render(ReportFormat.HTML)
        assert "<h1>Test Report</h1>" in result
        assert "<h2>Overview</h2>" in result
        assert "This is the overview." in result

    def test_render_json(self) -> None:
        """Test rendering to JSON."""
        import json

        builder = ReportBuilder("Test Report", ReportType.PERFORMANCE)
        builder.add_section("Overview", "Content")
        result = builder.render(ReportFormat.JSON)
        data = json.loads(result)
        assert data["title"] == "Test Report"
        assert len(data["sections"]) == 1

    def test_render_text(self) -> None:
        """Test rendering to plain text."""
        builder = ReportBuilder("Test Report", ReportType.PERFORMANCE)
        builder.add_section("Overview", "Content")
        result = builder.render(ReportFormat.TEXT)
        assert "Test Report" in result
        assert "Overview" in result


class TestPerformanceReportGenerator:
    """Test PerformanceReportGenerator class."""

    @pytest.fixture
    def sample_data(self) -> PerformanceReportData:
        """Create sample performance data."""
        return PerformanceReportData(
            model_name="text_classifier",
            model_version="v1.0",
            time_range=TimeRange.last_hours(24),
            total_predictions=10000,
            accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            latency_summary=MetricSummary(
                name="latency_ms",
                count=10000,
                mean=45.0,
                std=10.0,
                min_val=20.0,
                max_val=200.0,
                p50=42.0,
                p95=70.0,
                p99=95.0,
            ),
            confidence_summary=None,
            label_distribution={"spam": 3000, "ham": 7000},
            error_rate=0.001,
        )

    def test_generate_markdown(self, sample_data: PerformanceReportData) -> None:
        """Test generating markdown report."""
        generator = PerformanceReportGenerator()
        report = generator.generate(sample_data, ReportFormat.MARKDOWN)
        assert "Model Performance Report" in report
        assert "text_classifier" in report
        assert "Accuracy" in report
        assert "0.92" in report

    def test_generate_html(self, sample_data: PerformanceReportData) -> None:
        """Test generating HTML report."""
        generator = PerformanceReportGenerator()
        report = generator.generate(sample_data, ReportFormat.HTML)
        assert "<html>" in report
        assert "text_classifier" in report

    def test_generate_with_distribution(self, sample_data: PerformanceReportData) -> None:
        """Test that label distribution is included."""
        generator = PerformanceReportGenerator()
        report = generator.generate(sample_data, ReportFormat.MARKDOWN)
        assert "spam" in report
        assert "ham" in report

    def test_generate_with_latency(self, sample_data: PerformanceReportData) -> None:
        """Test that latency stats are included."""
        generator = PerformanceReportGenerator()
        report = generator.generate(sample_data, ReportFormat.MARKDOWN)
        assert "P95" in report
        assert "70.00ms" in report


class TestDriftReportGenerator:
    """Test DriftReportGenerator class."""

    @pytest.fixture
    def sample_drift_data(self) -> DriftReportData:
        """Create sample drift data."""
        return DriftReportData(
            model_name="text_classifier",
            time_range=TimeRange.last_hours(24),
            drift_detected=True,
            drift_score=0.15,
            drift_type="concept_drift",
            affected_features=["feature_1", "feature_2"],
            baseline_stats={"mean": 0.5, "std": 0.1},
            current_stats={"mean": 0.6, "std": 0.15},
            drift_history=[
                {"timestamp": "2024-01-01T10:00:00", "drift_score": 0.05, "drift_detected": False},
                {"timestamp": "2024-01-01T11:00:00", "drift_score": 0.15, "drift_detected": True},
            ],
        )

    def test_generate_drift_detected(self, sample_drift_data: DriftReportData) -> None:
        """Test generating report when drift is detected."""
        generator = DriftReportGenerator()
        report = generator.generate(sample_drift_data, ReportFormat.MARKDOWN)
        assert "Drift Detection Report" in report
        assert "DETECTED" in report
        assert "feature_1" in report

    def test_generate_no_drift(self) -> None:
        """Test generating report when no drift."""
        data = DriftReportData(
            model_name="test_model",
            time_range=TimeRange.last_hours(24),
            drift_detected=False,
            drift_score=0.02,
            drift_type="none",
        )
        generator = DriftReportGenerator()
        report = generator.generate(data, ReportFormat.MARKDOWN)
        assert "Not Detected" in report

    def test_generate_with_history(self, sample_drift_data: DriftReportData) -> None:
        """Test that history is included."""
        generator = DriftReportGenerator()
        report = generator.generate(sample_drift_data, ReportFormat.MARKDOWN)
        assert "History" in report
        assert "2024-01-01" in report


class TestExperimentReportGenerator:
    """Test ExperimentReportGenerator class."""

    @pytest.fixture
    def sample_experiment_data(self) -> ExperimentReportData:
        """Create sample experiment data."""
        return ExperimentReportData(
            experiment_id="exp_001",
            experiment_name="New Model Test",
            time_range=TimeRange.last_days(7),
            status="completed",
            champion_metrics={"accuracy": 0.85, "latency_ms": 50.0},
            challenger_metrics={"accuracy": 0.88, "latency_ms": 45.0},
            statistical_significance=True,
            p_value=0.02,
            winner="challenger",
            recommendation="Promote challenger model to production",
        )

    def test_generate_experiment_report(
        self, sample_experiment_data: ExperimentReportData
    ) -> None:
        """Test generating experiment report."""
        generator = ExperimentReportGenerator()
        report = generator.generate(sample_experiment_data, ReportFormat.MARKDOWN)
        assert "A/B Experiment Report" in report
        assert "exp_001" in report
        assert "Champion" in report
        assert "Challenger" in report

    def test_generate_with_significance(
        self, sample_experiment_data: ExperimentReportData
    ) -> None:
        """Test that significance is shown."""
        generator = ExperimentReportGenerator()
        report = generator.generate(sample_experiment_data, ReportFormat.MARKDOWN)
        assert "Yes" in report  # Significant
        assert "0.02" in report  # P-value

    def test_generate_recommendation(
        self, sample_experiment_data: ExperimentReportData
    ) -> None:
        """Test that recommendation is included."""
        generator = ExperimentReportGenerator()
        report = generator.generate(sample_experiment_data, ReportFormat.MARKDOWN)
        assert "Promote challenger" in report


class TestActionReportGenerator:
    """Test ActionReportGenerator class."""

    @pytest.fixture
    def sample_action_data(self) -> ActionReportData:
        """Create sample action data."""
        return ActionReportData(
            time_range=TimeRange.last_days(7),
            retrain_triggers=[
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "model": "text_classifier",
                    "reason": "drift_detected",
                    "status": "completed",
                },
                {
                    "timestamp": "2024-01-02T10:00:00",
                    "model": "text_classifier",
                    "reason": "scheduled",
                    "status": "completed",
                },
            ],
            rollbacks=[
                {
                    "timestamp": "2024-01-01T12:00:00",
                    "model": "text_classifier",
                    "from_version": "v1.1",
                    "to_version": "v1.0",
                    "reason": "performance_degradation",
                },
            ],
            alerts_sent=5,
            actions_taken=3,
        )

    def test_generate_action_report(self, sample_action_data: ActionReportData) -> None:
        """Test generating action report."""
        generator = ActionReportGenerator()
        report = generator.generate(sample_action_data, ReportFormat.MARKDOWN)
        assert "Automated Actions Report" in report
        assert "Retrain Triggers" in report
        assert "Rollbacks" in report

    def test_generate_summary(self, sample_action_data: ActionReportData) -> None:
        """Test that summary is included."""
        generator = ActionReportGenerator()
        report = generator.generate(sample_action_data, ReportFormat.MARKDOWN)
        assert "Total Actions Taken" in report
        assert "Alerts Sent" in report

    def test_generate_empty_report(self) -> None:
        """Test generating report with no actions."""
        data = ActionReportData(
            time_range=TimeRange.last_hours(24),
            alerts_sent=0,
            actions_taken=0,
        )
        generator = ActionReportGenerator()
        report = generator.generate(data, ReportFormat.MARKDOWN)
        assert "Actions Summary" in report
        assert "0" in report


class TestMonitoringReportGenerator:
    """Test MonitoringReportGenerator class."""

    def test_generate_performance_report(self) -> None:
        """Test generating performance report via main generator."""
        generator = MonitoringReportGenerator()
        data = PerformanceReportData(
            model_name="test_model",
            model_version="v1.0",
            time_range=TimeRange.last_hours(24),
            total_predictions=1000,
            accuracy=0.9,
            precision=None,
            recall=None,
            f1_score=None,
            latency_summary=None,
            confidence_summary=None,
        )
        report = generator.generate_performance_report(data)
        assert "Model Performance Report" in report

    def test_generate_drift_report(self) -> None:
        """Test generating drift report via main generator."""
        generator = MonitoringReportGenerator()
        data = DriftReportData(
            model_name="test_model",
            time_range=TimeRange.last_hours(24),
            drift_detected=False,
            drift_score=0.01,
            drift_type="none",
        )
        report = generator.generate_drift_report(data)
        assert "Drift Detection Report" in report

    def test_generate_experiment_report(self) -> None:
        """Test generating experiment report via main generator."""
        generator = MonitoringReportGenerator()
        data = ExperimentReportData(
            experiment_id="exp_001",
            experiment_name="Test",
            time_range=TimeRange.last_days(1),
            status="running",
            champion_metrics={"accuracy": 0.8},
            challenger_metrics={"accuracy": 0.82},
            statistical_significance=False,
            p_value=0.1,
            winner=None,
            recommendation="Continue testing",
        )
        report = generator.generate_experiment_report(data)
        assert "A/B Experiment Report" in report

    def test_generate_action_report(self) -> None:
        """Test generating action report via main generator."""
        generator = MonitoringReportGenerator()
        data = ActionReportData(
            time_range=TimeRange.last_hours(24),
            alerts_sent=0,
            actions_taken=0,
        )
        report = generator.generate_action_report(data)
        assert "Automated Actions Report" in report

    def test_generate_comprehensive_report(self) -> None:
        """Test generating comprehensive report."""
        generator = MonitoringReportGenerator()

        perf_data = PerformanceReportData(
            model_name="test_model",
            model_version="v1.0",
            time_range=TimeRange.last_hours(24),
            total_predictions=1000,
            accuracy=0.9,
            precision=None,
            recall=None,
            f1_score=None,
            latency_summary=None,
            confidence_summary=None,
        )

        drift_data = DriftReportData(
            model_name="test_model",
            time_range=TimeRange.last_hours(24),
            drift_detected=False,
            drift_score=0.01,
            drift_type="none",
        )

        report = generator.generate_comprehensive_report(
            performance_data=perf_data,
            drift_data=drift_data,
        )

        assert "Comprehensive Model Monitoring Report" in report
        assert "Model Performance" in report
        assert "Drift Detection" in report

    def test_generate_comprehensive_all_sections(self) -> None:
        """Test comprehensive report with all sections."""
        generator = MonitoringReportGenerator()

        perf_data = PerformanceReportData(
            model_name="test_model",
            model_version="v1.0",
            time_range=TimeRange.last_hours(24),
            total_predictions=1000,
            accuracy=0.9,
            precision=None,
            recall=None,
            f1_score=None,
            latency_summary=None,
            confidence_summary=None,
        )

        drift_data = DriftReportData(
            model_name="test_model",
            time_range=TimeRange.last_hours(24),
            drift_detected=False,
            drift_score=0.01,
            drift_type="none",
        )

        exp_data = ExperimentReportData(
            experiment_id="exp_001",
            experiment_name="Test",
            time_range=TimeRange.last_days(1),
            status="running",
            champion_metrics={},
            challenger_metrics={},
            statistical_significance=False,
            p_value=None,
            winner=None,
            recommendation="Continue",
        )

        action_data = ActionReportData(
            time_range=TimeRange.last_hours(24),
            alerts_sent=0,
            actions_taken=0,
        )

        report = generator.generate_comprehensive_report(
            performance_data=perf_data,
            drift_data=drift_data,
            experiment_data=exp_data,
            action_data=action_data,
        )

        assert "Model Performance" in report
        assert "Drift Detection" in report
        assert "A/B Experiments" in report
        assert "Automated Actions" in report


class TestReportFormat:
    """Test ReportFormat enum."""

    def test_format_values(self) -> None:
        """Test format enum values."""
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.TEXT.value == "text"


class TestReportType:
    """Test ReportType enum."""

    def test_type_values(self) -> None:
        """Test type enum values."""
        assert ReportType.PERFORMANCE.value == "performance"
        assert ReportType.DRIFT.value == "drift"
        assert ReportType.EXPERIMENT.value == "experiment"
        assert ReportType.ACTION.value == "action"
        assert ReportType.COMPREHENSIVE.value == "comprehensive"
