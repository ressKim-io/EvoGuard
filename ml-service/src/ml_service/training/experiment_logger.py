"""Simple experiment logger for tracking training runs.

Saves experiment results to JSON and generates markdown reports.
Lightweight alternative when MLflow is not available.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Log experiments to JSON and generate markdown reports.

    Example:
        >>> exp_logger = ExperimentLogger()
        >>> exp_logger.log_experiment(
        ...     name="qlora-bert-jigsaw",
        ...     config={"model": "bert-base-uncased", "epochs": 3},
        ...     metrics={"f1": 0.913, "accuracy": 0.913},
        ... )
        >>> exp_logger.generate_report()
    """

    def __init__(
        self,
        experiments_dir: Path | str = "experiments",
        project_name: str = "evoguard",
    ) -> None:
        """Initialize experiment logger.

        Args:
            experiments_dir: Directory to store experiment logs.
            project_name: Name of the project for reports.
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name

        # Paths
        self.experiments_file = self.experiments_dir / "experiments.json"
        self.reports_dir = self.experiments_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Load existing experiments
        self.experiments: list[dict[str, Any]] = self._load_experiments()

    def _load_experiments(self) -> list[dict[str, Any]]:
        """Load existing experiments from JSON file."""
        if self.experiments_file.exists():
            with open(self.experiments_file) as f:
                return json.load(f)
        return []

    def _save_experiments(self) -> None:
        """Save experiments to JSON file."""
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def log_experiment(
        self,
        name: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        dataset: str | None = None,
        model_path: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Log a training experiment.

        Args:
            name: Experiment name.
            config: Training configuration (hyperparameters).
            metrics: Training metrics (f1, accuracy, etc.).
            dataset: Dataset used for training.
            model_path: Path to saved model.
            notes: Additional notes.
            tags: Tags for categorization.

        Returns:
            The logged experiment record.
        """
        experiment = {
            "id": len(self.experiments) + 1,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "dataset": dataset,
            "model_path": model_path,
            "notes": notes,
            "tags": tags or [],
        }

        self.experiments.append(experiment)
        self._save_experiments()

        logger.info(f"Logged experiment #{experiment['id']}: {name}")
        return experiment

    def get_best_experiment(self, metric: str = "eval_f1") -> dict[str, Any] | None:
        """Get the best experiment by a metric.

        Args:
            metric: Metric to compare (default: eval_f1).

        Returns:
            Best experiment or None if no experiments.
        """
        if not self.experiments:
            return None

        return max(
            self.experiments,
            key=lambda x: x.get("metrics", {}).get(metric, 0),
        )

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate a markdown report of all experiments.

        Args:
            output_file: Optional output file path.

        Returns:
            Markdown report content.
        """
        report_lines = [
            f"# {self.project_name.upper()} Training Experiments Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Experiments:** {len(self.experiments)}",
            "",
        ]

        # Best experiment
        best = self.get_best_experiment()
        if best:
            report_lines.extend([
                "## 🏆 Best Experiment",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| **Name** | {best['name']} |",
                f"| **F1 Score** | {best['metrics'].get('eval_f1', 'N/A'):.4f} |",
                f"| **Accuracy** | {best['metrics'].get('eval_accuracy', 'N/A'):.4f} |",
                f"| **Date** | {best['timestamp'][:10]} |",
                "",
            ])

        # Summary table
        report_lines.extend([
            "## 📊 Experiments Summary",
            "",
            "| # | Name | Dataset | F1 | Accuracy | Date |",
            "|---|------|---------|-----|----------|------|",
        ])

        for exp in sorted(self.experiments, key=lambda x: x["id"], reverse=True):
            metrics = exp.get("metrics", {})
            f1 = metrics.get("eval_f1", metrics.get("f1", "N/A"))
            acc = metrics.get("eval_accuracy", metrics.get("accuracy", "N/A"))

            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else f1
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else acc

            report_lines.append(
                f"| {exp['id']} | {exp['name']} | {exp.get('dataset', 'N/A')} | "
                f"{f1_str} | {acc_str} | {exp['timestamp'][:10]} |"
            )

        report_lines.append("")

        # Detailed experiments
        report_lines.extend([
            "## 📝 Experiment Details",
            "",
        ])

        for exp in sorted(self.experiments, key=lambda x: x["id"], reverse=True):
            report_lines.extend([
                f"### Experiment #{exp['id']}: {exp['name']}",
                "",
                f"**Timestamp:** {exp['timestamp']}",
                "",
            ])

            if exp.get("dataset"):
                report_lines.append(f"**Dataset:** {exp['dataset']}")

            if exp.get("tags"):
                report_lines.append(f"**Tags:** {', '.join(exp['tags'])}")

            report_lines.append("")

            # Config
            report_lines.extend([
                "#### Configuration",
                "```json",
                json.dumps(exp.get("config", {}), indent=2),
                "```",
                "",
            ])

            # Metrics
            report_lines.extend([
                "#### Metrics",
                "| Metric | Value |",
                "|--------|-------|",
            ])

            for key, value in exp.get("metrics", {}).items():
                if isinstance(value, float):
                    report_lines.append(f"| {key} | {value:.6f} |")
                else:
                    report_lines.append(f"| {key} | {value} |")

            report_lines.append("")

            if exp.get("notes"):
                report_lines.extend([
                    "#### Notes",
                    exp["notes"],
                    "",
                ])

            if exp.get("model_path"):
                report_lines.append(f"**Model Path:** `{exp['model_path']}`")
                report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        report = "\n".join(report_lines)

        # Save report
        if output_file is None:
            output_file = self.reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Report saved to: {output_path}")

        # Also save latest report
        latest_path = self.reports_dir / "LATEST_REPORT.md"
        with open(latest_path, "w") as f:
            f.write(report)

        return report

    def generate_comparison_chart_data(self) -> dict[str, Any]:
        """Generate data for comparison charts (JSON format).

        Returns:
            Chart data in a format suitable for visualization.
        """
        if not self.experiments:
            return {"experiments": [], "metrics": []}

        chart_data = {
            "experiments": [],
            "metrics": ["eval_f1", "eval_accuracy", "eval_precision", "eval_recall"],
        }

        for exp in self.experiments:
            chart_data["experiments"].append({
                "id": exp["id"],
                "name": exp["name"],
                "timestamp": exp["timestamp"],
                "values": {
                    metric: exp.get("metrics", {}).get(metric, 0)
                    for metric in chart_data["metrics"]
                },
            })

        # Save chart data
        chart_file = self.experiments_dir / "chart_data.json"
        with open(chart_file, "w") as f:
            json.dump(chart_data, f, indent=2)

        return chart_data


def log_training_run(
    name: str,
    config: dict[str, Any],
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    dataset: str | None = None,
    model_path: str | None = None,
    experiments_dir: Path | str = "experiments",
) -> dict[str, Any]:
    """Convenience function to log a training run.

    Args:
        name: Experiment name.
        config: Training configuration.
        train_metrics: Training metrics.
        eval_metrics: Validation metrics.
        test_metrics: Test metrics (optional).
        dataset: Dataset name.
        model_path: Path to saved model.
        experiments_dir: Directory for experiments.

    Returns:
        Logged experiment record.
    """
    exp_logger = ExperimentLogger(experiments_dir=experiments_dir)

    # Combine metrics
    all_metrics = {
        "train_loss": train_metrics.get("train_loss"),
        **{f"eval_{k}" if not k.startswith("eval_") else k: v for k, v in eval_metrics.items()},
    }

    if test_metrics:
        all_metrics.update({
            f"test_{k}" if not k.startswith("test_") else k: v
            for k, v in test_metrics.items()
        })

    experiment = exp_logger.log_experiment(
        name=name,
        config=config,
        metrics=all_metrics,
        dataset=dataset,
        model_path=model_path,
        tags=["qlora", "toxic-classification"],
    )

    # Generate updated report
    exp_logger.generate_report()

    return experiment
