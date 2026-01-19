"""Automatic report generation for model monitoring (Phase 5).

This module provides comprehensive report generation capabilities for:
- Model performance summaries
- Drift detection reports
- A/B experiment results
- Automated action logs
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class ReportFormat(str, Enum):
    """Supported report output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


class ReportType(str, Enum):
    """Types of monitoring reports."""

    PERFORMANCE = "performance"
    DRIFT = "drift"
    EXPERIMENT = "experiment"
    ACTION = "action"
    COMPREHENSIVE = "comprehensive"


@dataclass
class TimeRange:
    """Time range for report data."""

    start: datetime
    end: datetime

    @classmethod
    def last_hours(cls, hours: int) -> TimeRange:
        """Create time range for last N hours."""
        end = datetime.now()
        start = end - timedelta(hours=hours)
        return cls(start=start, end=end)

    @classmethod
    def last_days(cls, days: int) -> TimeRange:
        """Create time range for last N days."""
        end = datetime.now()
        start = end - timedelta(days=days)
        return cls(start=start, end=end)

    @classmethod
    def today(cls) -> TimeRange:
        """Create time range for today."""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return cls(start=start, end=now)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> MetricSummary | None:
        """Create summary from list of values."""
        if not values:
            return None

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return cls(
            name=name,
            count=n,
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0.0,
            min_val=min(values),
            max_val=max(values),
            p50=sorted_vals[int(n * 0.50)] if n > 0 else 0.0,
            p95=sorted_vals[int(n * 0.95)] if n > 0 else 0.0,
            p99=sorted_vals[int(n * 0.99)] if n > 0 else 0.0,
        )


@dataclass
class PerformanceReportData:
    """Data for performance report."""

    model_name: str
    model_version: str
    time_range: TimeRange
    total_predictions: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None
    latency_summary: MetricSummary | None
    confidence_summary: MetricSummary | None
    label_distribution: dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0


@dataclass
class DriftReportData:
    """Data for drift report."""

    model_name: str
    time_range: TimeRange
    drift_detected: bool
    drift_score: float
    drift_type: str
    affected_features: list[str] = field(default_factory=list)
    baseline_stats: dict[str, float] = field(default_factory=dict)
    current_stats: dict[str, float] = field(default_factory=dict)
    drift_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentReportData:
    """Data for experiment report."""

    experiment_id: str
    experiment_name: str
    time_range: TimeRange
    status: str
    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    statistical_significance: bool
    p_value: float | None
    winner: str | None
    recommendation: str


@dataclass
class ActionReportData:
    """Data for automated action report."""

    time_range: TimeRange
    retrain_triggers: list[dict[str, Any]] = field(default_factory=list)
    rollbacks: list[dict[str, Any]] = field(default_factory=list)
    alerts_sent: int = 0
    actions_taken: int = 0


class ReportSection:
    """A section of a report."""

    def __init__(self, title: str, content: str, level: int = 2) -> None:
        self.title = title
        self.content = content
        self.level = level
        self.subsections: list[ReportSection] = []

    def add_subsection(self, title: str, content: str) -> ReportSection:
        """Add a subsection."""
        subsection = ReportSection(title, content, self.level + 1)
        self.subsections.append(subsection)
        return subsection


class ReportBuilder:
    """Builder for constructing reports."""

    def __init__(self, title: str, report_type: ReportType) -> None:
        self.title = title
        self.report_type = report_type
        self.sections: list[ReportSection] = []
        self.metadata: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "report_type": report_type.value,
        }

    def add_section(self, title: str, content: str) -> ReportSection:
        """Add a section to the report."""
        section = ReportSection(title, content)
        self.sections.append(section)
        return section

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the report."""
        self.metadata[key] = value

    def render(self, format_type: ReportFormat) -> str:
        """Render the report in the specified format."""
        if format_type == ReportFormat.MARKDOWN:
            return self._render_markdown()
        elif format_type == ReportFormat.HTML:
            return self._render_html()
        elif format_type == ReportFormat.JSON:
            return self._render_json()
        else:
            return self._render_text()

    def _render_markdown(self) -> str:
        """Render as Markdown."""
        lines = [f"# {self.title}", ""]

        # Metadata
        lines.append("**Report Metadata**")
        lines.append("")
        for key, value in self.metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Sections
        for section in self.sections:
            lines.extend(self._render_section_md(section))

        return "\n".join(lines)

    def _render_section_md(self, section: ReportSection) -> list[str]:
        """Render a section as Markdown."""
        lines = []
        prefix = "#" * section.level
        lines.append(f"{prefix} {section.title}")
        lines.append("")
        lines.append(section.content)
        lines.append("")

        for subsection in section.subsections:
            lines.extend(self._render_section_md(subsection))

        return lines

    def _render_html(self) -> str:
        """Render as HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #333; }",
            "h2 { color: #555; border-bottom: 1px solid #ddd; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f4f4f4; }",
            ".metric { background-color: #e7f3ff; padding: 10px; margin: 5px 0; }",
            ".alert { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }",
            ".success { background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; }",
            ".danger { background-color: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.title}</h1>",
        ]

        # Metadata
        html_parts.append("<div class='metric'>")
        html_parts.append("<strong>Report Metadata</strong><br>")
        for key, value in self.metadata.items():
            html_parts.append(f"<em>{key}</em>: {value}<br>")
        html_parts.append("</div>")

        # Sections
        for section in self.sections:
            html_parts.extend(self._render_section_html(section))

        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def _render_section_html(self, section: ReportSection) -> list[str]:
        """Render a section as HTML."""
        tag = f"h{min(section.level, 6)}"
        html_parts = [
            f"<{tag}>{section.title}</{tag}>",
            f"<div>{section.content}</div>",
        ]

        for subsection in section.subsections:
            html_parts.extend(self._render_section_html(subsection))

        return html_parts

    def _render_json(self) -> str:
        """Render as JSON."""
        import json

        data = {
            "title": self.title,
            "metadata": self.metadata,
            "sections": [self._section_to_dict(s) for s in self.sections],
        }
        return json.dumps(data, indent=2, default=str)

    def _section_to_dict(self, section: ReportSection) -> dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": section.title,
            "content": section.content,
            "level": section.level,
            "subsections": [self._section_to_dict(s) for s in section.subsections],
        }

    def _render_text(self) -> str:
        """Render as plain text."""
        lines = [self.title, "=" * len(self.title), ""]

        for key, value in self.metadata.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

        for section in self.sections:
            lines.extend(self._render_section_text(section))

        return "\n".join(lines)

    def _render_section_text(self, section: ReportSection) -> list[str]:
        """Render a section as plain text."""
        lines = [section.title, "-" * len(section.title), "", section.content, ""]

        for subsection in section.subsections:
            lines.extend(self._render_section_text(subsection))

        return lines


class PerformanceReportGenerator:
    """Generate model performance reports."""

    def generate(
        self,
        data: PerformanceReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a performance report."""
        builder = ReportBuilder(
            f"Model Performance Report - {data.model_name}",
            ReportType.PERFORMANCE,
        )

        builder.add_metadata("model_name", data.model_name)
        builder.add_metadata("model_version", data.model_version)
        builder.add_metadata("time_range_start", data.time_range.start.isoformat())
        builder.add_metadata("time_range_end", data.time_range.end.isoformat())

        # Overview section
        overview = self._build_overview(data)
        builder.add_section("Overview", overview)

        # Metrics section
        metrics = self._build_metrics(data)
        builder.add_section("Performance Metrics", metrics)

        # Latency section
        if data.latency_summary:
            latency = self._build_latency(data.latency_summary)
            builder.add_section("Latency Statistics", latency)

        # Distribution section
        if data.label_distribution:
            dist = self._build_distribution(data.label_distribution)
            builder.add_section("Label Distribution", dist)

        return builder.render(format_type)

    def _build_overview(self, data: PerformanceReportData) -> str:
        """Build overview section content."""
        return f"""
- **Total Predictions**: {data.total_predictions:,}
- **Error Rate**: {data.error_rate:.2%}
- **Time Period**: {data.time_range.start.strftime('%Y-%m-%d %H:%M')} to {data.time_range.end.strftime('%Y-%m-%d %H:%M')}
""".strip()

    def _build_metrics(self, data: PerformanceReportData) -> str:
        """Build metrics section content."""
        metrics_lines = []
        if data.accuracy is not None:
            metrics_lines.append(f"- **Accuracy**: {data.accuracy:.4f}")
        if data.precision is not None:
            metrics_lines.append(f"- **Precision**: {data.precision:.4f}")
        if data.recall is not None:
            metrics_lines.append(f"- **Recall**: {data.recall:.4f}")
        if data.f1_score is not None:
            metrics_lines.append(f"- **F1 Score**: {data.f1_score:.4f}")

        if not metrics_lines:
            return "No metric data available."

        return "\n".join(metrics_lines)

    def _build_latency(self, summary: MetricSummary) -> str:
        """Build latency section content."""
        return f"""
| Statistic | Value |
|-----------|-------|
| Count | {summary.count:,} |
| Mean | {summary.mean:.2f}ms |
| Std Dev | {summary.std:.2f}ms |
| Min | {summary.min_val:.2f}ms |
| Max | {summary.max_val:.2f}ms |
| P50 | {summary.p50:.2f}ms |
| P95 | {summary.p95:.2f}ms |
| P99 | {summary.p99:.2f}ms |
""".strip()

    def _build_distribution(self, distribution: dict[str, int]) -> str:
        """Build distribution section content."""
        total = sum(distribution.values())
        lines = ["| Label | Count | Percentage |", "|-------|-------|------------|"]
        for label, count in sorted(distribution.items()):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"| {label} | {count:,} | {pct:.1f}% |")
        return "\n".join(lines)


class DriftReportGenerator:
    """Generate drift detection reports."""

    def generate(
        self,
        data: DriftReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a drift detection report."""
        builder = ReportBuilder(
            f"Drift Detection Report - {data.model_name}",
            ReportType.DRIFT,
        )

        builder.add_metadata("model_name", data.model_name)
        builder.add_metadata("drift_detected", data.drift_detected)
        builder.add_metadata("time_range_start", data.time_range.start.isoformat())
        builder.add_metadata("time_range_end", data.time_range.end.isoformat())

        # Status section
        status = self._build_status(data)
        builder.add_section("Drift Status", status)

        # Details section
        if data.drift_detected:
            details = self._build_details(data)
            builder.add_section("Drift Details", details)

        # History section
        if data.drift_history:
            history = self._build_history(data.drift_history)
            builder.add_section("Drift History", history)

        return builder.render(format_type)

    def _build_status(self, data: DriftReportData) -> str:
        """Build status section content."""
        status_emoji = "⚠️" if data.drift_detected else "✅"
        status_text = "DETECTED" if data.drift_detected else "Not Detected"
        return f"""
{status_emoji} **Drift Status**: {status_text}

- **Drift Score**: {data.drift_score:.4f}
- **Drift Type**: {data.drift_type}
""".strip()

    def _build_details(self, data: DriftReportData) -> str:
        """Build details section content."""
        lines = ["**Affected Features**:", ""]
        for feature in data.affected_features:
            lines.append(f"- {feature}")

        if data.baseline_stats and data.current_stats:
            lines.extend(["", "**Statistical Comparison**:", ""])
            lines.append("| Feature | Baseline | Current | Change |")
            lines.append("|---------|----------|---------|--------|")
            for key in data.baseline_stats:
                baseline = data.baseline_stats.get(key, 0)
                current = data.current_stats.get(key, 0)
                change = ((current - baseline) / baseline * 100) if baseline != 0 else 0
                lines.append(f"| {key} | {baseline:.4f} | {current:.4f} | {change:+.1f}% |")

        return "\n".join(lines)

    def _build_history(self, history: list[dict[str, Any]]) -> str:
        """Build history section content."""
        lines = ["| Timestamp | Drift Score | Detected |", "|-----------|-------------|----------|"]
        for entry in history[-10:]:  # Last 10 entries
            ts = entry.get("timestamp", "N/A")
            score = entry.get("drift_score", 0)
            detected = "Yes" if entry.get("drift_detected", False) else "No"
            lines.append(f"| {ts} | {score:.4f} | {detected} |")
        return "\n".join(lines)


class ExperimentReportGenerator:
    """Generate A/B experiment reports."""

    def generate(
        self,
        data: ExperimentReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate an experiment report."""
        builder = ReportBuilder(
            f"A/B Experiment Report - {data.experiment_name}",
            ReportType.EXPERIMENT,
        )

        builder.add_metadata("experiment_id", data.experiment_id)
        builder.add_metadata("status", data.status)
        builder.add_metadata("time_range_start", data.time_range.start.isoformat())
        builder.add_metadata("time_range_end", data.time_range.end.isoformat())

        # Summary section
        summary = self._build_summary(data)
        builder.add_section("Experiment Summary", summary)

        # Comparison section
        comparison = self._build_comparison(data)
        builder.add_section("Variant Comparison", comparison)

        # Results section
        results = self._build_results(data)
        builder.add_section("Statistical Results", results)

        # Recommendation section
        recommendation = self._build_recommendation(data)
        builder.add_section("Recommendation", recommendation)

        return builder.render(format_type)

    def _build_summary(self, data: ExperimentReportData) -> str:
        """Build summary section content."""
        return f"""
- **Experiment ID**: {data.experiment_id}
- **Experiment Name**: {data.experiment_name}
- **Status**: {data.status}
- **Winner**: {data.winner or "Not determined yet"}
""".strip()

    def _build_comparison(self, data: ExperimentReportData) -> str:
        """Build comparison section content."""
        lines = ["| Metric | Champion | Challenger | Difference |"]
        lines.append("|--------|----------|------------|------------|")

        all_metrics = set(data.champion_metrics.keys()) | set(data.challenger_metrics.keys())
        for metric in sorted(all_metrics):
            champ = data.champion_metrics.get(metric, 0)
            chall = data.challenger_metrics.get(metric, 0)
            diff = chall - champ
            diff_str = f"{diff:+.4f}" if isinstance(diff, float) else str(diff)
            lines.append(f"| {metric} | {champ:.4f} | {chall:.4f} | {diff_str} |")

        return "\n".join(lines)

    def _build_results(self, data: ExperimentReportData) -> str:
        """Build results section content."""
        sig_text = "Yes ✅" if data.statistical_significance else "No ❌"
        p_value_text = f"{data.p_value:.4f}" if data.p_value is not None else "N/A"

        return f"""
- **Statistically Significant**: {sig_text}
- **P-Value**: {p_value_text}
""".strip()

    def _build_recommendation(self, data: ExperimentReportData) -> str:
        """Build recommendation section content."""
        return data.recommendation


class ActionReportGenerator:
    """Generate automated action reports."""

    def generate(
        self,
        data: ActionReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate an action report."""
        builder = ReportBuilder(
            "Automated Actions Report",
            ReportType.ACTION,
        )

        builder.add_metadata("time_range_start", data.time_range.start.isoformat())
        builder.add_metadata("time_range_end", data.time_range.end.isoformat())
        builder.add_metadata("total_actions", data.actions_taken)

        # Summary section
        summary = self._build_summary(data)
        builder.add_section("Actions Summary", summary)

        # Retrain triggers section
        if data.retrain_triggers:
            retrain = self._build_retrain_triggers(data.retrain_triggers)
            builder.add_section("Retrain Triggers", retrain)

        # Rollbacks section
        if data.rollbacks:
            rollbacks = self._build_rollbacks(data.rollbacks)
            builder.add_section("Model Rollbacks", rollbacks)

        return builder.render(format_type)

    def _build_summary(self, data: ActionReportData) -> str:
        """Build summary section content."""
        return f"""
- **Total Actions Taken**: {data.actions_taken}
- **Alerts Sent**: {data.alerts_sent}
- **Retrain Triggers**: {len(data.retrain_triggers)}
- **Rollbacks**: {len(data.rollbacks)}
""".strip()

    def _build_retrain_triggers(self, triggers: list[dict[str, Any]]) -> str:
        """Build retrain triggers section content."""
        lines = ["| Timestamp | Model | Reason | Status |"]
        lines.append("|-----------|-------|--------|--------|")

        for trigger in triggers:
            ts = trigger.get("timestamp", "N/A")
            model = trigger.get("model", "N/A")
            reason = trigger.get("reason", "N/A")
            status = trigger.get("status", "N/A")
            lines.append(f"| {ts} | {model} | {reason} | {status} |")

        return "\n".join(lines)

    def _build_rollbacks(self, rollbacks: list[dict[str, Any]]) -> str:
        """Build rollbacks section content."""
        lines = ["| Timestamp | Model | From Version | To Version | Reason |"]
        lines.append("|-----------|-------|--------------|------------|--------|")

        for rollback in rollbacks:
            ts = rollback.get("timestamp", "N/A")
            model = rollback.get("model", "N/A")
            from_ver = rollback.get("from_version", "N/A")
            to_ver = rollback.get("to_version", "N/A")
            reason = rollback.get("reason", "N/A")
            lines.append(f"| {ts} | {model} | {from_ver} | {to_ver} | {reason} |")

        return "\n".join(lines)


class MonitoringReportGenerator:
    """Main class for generating comprehensive monitoring reports."""

    def __init__(self) -> None:
        self.performance_generator = PerformanceReportGenerator()
        self.drift_generator = DriftReportGenerator()
        self.experiment_generator = ExperimentReportGenerator()
        self.action_generator = ActionReportGenerator()

    def generate_performance_report(
        self,
        data: PerformanceReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a performance report."""
        return self.performance_generator.generate(data, format_type)

    def generate_drift_report(
        self,
        data: DriftReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a drift report."""
        return self.drift_generator.generate(data, format_type)

    def generate_experiment_report(
        self,
        data: ExperimentReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate an experiment report."""
        return self.experiment_generator.generate(data, format_type)

    def generate_action_report(
        self,
        data: ActionReportData,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate an action report."""
        return self.action_generator.generate(data, format_type)

    def generate_comprehensive_report(
        self,
        performance_data: PerformanceReportData | None = None,
        drift_data: DriftReportData | None = None,
        experiment_data: ExperimentReportData | None = None,
        action_data: ActionReportData | None = None,
        format_type: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a comprehensive report combining all aspects."""
        builder = ReportBuilder(
            "Comprehensive Model Monitoring Report",
            ReportType.COMPREHENSIVE,
        )

        builder.add_metadata("generated_at", datetime.now().isoformat())

        # Performance section
        if performance_data:
            perf_content = self.performance_generator.generate(
                performance_data, ReportFormat.MARKDOWN
            )
            builder.add_section("Model Performance", perf_content)

        # Drift section
        if drift_data:
            drift_content = self.drift_generator.generate(
                drift_data, ReportFormat.MARKDOWN
            )
            builder.add_section("Drift Detection", drift_content)

        # Experiment section
        if experiment_data:
            exp_content = self.experiment_generator.generate(
                experiment_data, ReportFormat.MARKDOWN
            )
            builder.add_section("A/B Experiments", exp_content)

        # Actions section
        if action_data:
            action_content = self.action_generator.generate(
                action_data, ReportFormat.MARKDOWN
            )
            builder.add_section("Automated Actions", action_content)

        return builder.render(format_type)
