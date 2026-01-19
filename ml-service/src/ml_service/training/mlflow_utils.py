"""MLflow integration for experiment tracking and model registry."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from ml_service.training.config import TrainingConfig

logger = logging.getLogger(__name__)

# Optional imports for MLflow
HAS_MLFLOW = False
try:
    import mlflow
    from mlflow.models import infer_signature

    HAS_MLFLOW = True
except ImportError:
    pass

if TYPE_CHECKING:
    import mlflow


class MLflowTracker:
    """MLflow experiment tracking and model registry integration.

    Handles:
    - Experiment creation and management
    - Metric and parameter logging
    - Model artifact logging
    - Champion/Challenger model aliases
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize MLflow tracker.

        Args:
            config: Training configuration with MLflow settings.
        """
        if not HAS_MLFLOW:
            raise ImportError(
                "MLflow not installed. Install with: uv pip install --group training"
            )

        self.config = config
        self._run_id: str | None = None
        self._experiment_id: str | None = None

        # Set tracking URI
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI: {config.mlflow_tracking_uri}")

    def setup_experiment(self) -> str:
        """Set up or get existing experiment.

        Returns:
            Experiment ID.
        """
        experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment_name)

        if experiment is None:
            self._experiment_id = mlflow.create_experiment(
                self.config.mlflow_experiment_name,
                tags={"project": "evoguard", "task": "toxic-classification"},
            )
            logger.info(f"Created experiment: {self.config.mlflow_experiment_name}")
        else:
            self._experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.config.mlflow_experiment_name}")

        mlflow.set_experiment(self.config.mlflow_experiment_name)
        return self._experiment_id

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator["mlflow.ActiveRun", None, None]:
        """Start an MLflow run.

        Args:
            run_name: Optional name for the run.
            tags: Optional tags for the run.

        Yields:
            Active MLflow run.
        """
        self.setup_experiment()

        default_tags = {
            "model_name": self.config.model_name,
            "use_qlora": str(self.config.use_4bit_quantization),
        }
        if tags:
            default_tags.update(tags)

        with mlflow.start_run(run_name=run_name, tags=default_tags) as run:
            self._run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self._run_id}")

            # Log configuration
            self.log_config()

            yield run

            logger.info(f"Completed MLflow run: {self._run_id}")

    def log_config(self) -> None:
        """Log training configuration as parameters."""
        params = {
            "model_name": self.config.model_name,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "max_length": self.config.max_length,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "use_gradient_checkpointing": self.config.use_gradient_checkpointing,
            "use_4bit_quantization": self.config.use_4bit_quantization,
            "lora_r": self.config.lora.r,
            "lora_alpha": self.config.lora.lora_alpha,
            "lora_dropout": self.config.lora.lora_dropout,
            "seed": self.config.seed,
        }
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Optional step number for time series metrics.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        register_name: str | None = None,
    ) -> str:
        """Log model to MLflow.

        Args:
            model: The model to log (supports transformers, sklearn, etc.).
            artifact_path: Path within artifacts to store the model.
            register_name: Optional name to register in Model Registry.

        Returns:
            Model URI.
        """
        try:
            # Try to log as transformers model
            from mlflow.transformers import log_model as log_transformers

            model_info = log_transformers(
                transformers_model=model,
                artifact_path=artifact_path,
                registered_model_name=register_name,
            )
            logger.info(f"Logged transformers model: {model_info.model_uri}")
            return model_info.model_uri
        except Exception:
            # Fall back to generic logging
            model_info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=register_name,
            )
            logger.info(f"Logged pyfunc model: {model_info.model_uri}")
            return model_info.model_uri

    def log_artifact(self, local_path: Path | str, artifact_path: str | None = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Local path to the file.
            artifact_path: Optional path within artifacts.
        """
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(self, local_dir: Path | str, artifact_path: str | None = None) -> None:
        """Log all artifacts in a directory.

        Args:
            local_dir: Local directory path.
            artifact_path: Optional path within artifacts.
        """
        mlflow.log_artifacts(str(local_dir), artifact_path)

    def set_model_alias(
        self,
        name: str,
        version: int | str,
        alias: str,
    ) -> None:
        """Set an alias for a model version.

        Aliases can be 'champion' or 'challenger' for A/B testing.

        Args:
            name: Registered model name.
            version: Model version number.
            alias: Alias to set (e.g., 'champion', 'challenger').
        """
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(name, alias, str(version))
        logger.info(f"Set alias '{alias}' for {name} version {version}")

    def get_model_by_alias(self, name: str, alias: str) -> str:
        """Get model URI by alias.

        Args:
            name: Registered model name.
            alias: Model alias (e.g., 'champion').

        Returns:
            Model URI.
        """
        return f"models:/{name}@{alias}"

    def promote_to_champion(self, name: str, version: int | str) -> None:
        """Promote a model version to champion.

        Args:
            name: Registered model name.
            version: Version to promote.
        """
        self.set_model_alias(name, version, "champion")
        logger.info(f"Promoted {name} version {version} to champion")

    def register_challenger(self, name: str, version: int | str) -> None:
        """Register a model version as challenger.

        Args:
            name: Registered model name.
            version: Version to register as challenger.
        """
        self.set_model_alias(name, version, "challenger")
        logger.info(f"Registered {name} version {version} as challenger")

    def get_best_run(
        self,
        metric: str = "f1",
        ascending: bool = False,
    ) -> dict[str, Any] | None:
        """Get the best run from the experiment.

        Args:
            metric: Metric to sort by.
            ascending: Sort order.

        Returns:
            Best run info or None if no runs.
        """
        if self._experiment_id is None:
            self.setup_experiment()

        runs = mlflow.search_runs(
            experiment_ids=[self._experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if runs.empty:
            return None

        return runs.iloc[0].to_dict()

    def compare_runs(
        self,
        run_ids: list[str],
        metrics: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare.
            metrics: List of metrics to compare (all if None).

        Returns:
            Dictionary mapping run_id -> metric -> value.
        """
        client = mlflow.MlflowClient()
        results: dict[str, dict[str, float]] = {}

        for run_id in run_ids:
            run = client.get_run(run_id)
            run_metrics = run.data.metrics

            if metrics:
                run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}

            results[run_id] = run_metrics

        return results
