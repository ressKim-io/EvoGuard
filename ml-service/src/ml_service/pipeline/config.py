"""Pipeline configuration for Adversarial MLOps Pipeline."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AttackConfig(BaseModel):
    """Configuration for attack generation."""

    corpus_path: Path = Field(
        default=Path("data/toxic_corpus.csv"),
        description="Path to the toxic corpus for attacks",
    )
    num_variants: int = Field(
        default=5,
        description="Number of variants to generate per input text",
    )
    batch_size: int = Field(
        default=100,
        description="Number of samples to attack per cycle",
    )
    include_llm_strategies: bool = Field(
        default=False,
        description="Include LLM-based attack strategies (slower but more effective)",
    )
    text_column: str = Field(
        default="text",
        description="Column name for text in corpus",
    )
    label_column: str = Field(
        default="label",
        description="Column name for label in corpus",
    )


class QualityGateConfig(BaseModel):
    """Configuration for quality gate thresholds."""

    max_evasion_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable evasion rate (above this triggers retraining)",
    )
    min_f1_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable F1 score",
    )
    min_f1_drop: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable F1 drop from baseline",
    )
    min_precision: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable precision",
    )
    min_recall: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable recall",
    )


class ScheduleConfig(BaseModel):
    """Configuration for pipeline scheduling."""

    enabled: bool = Field(
        default=False,
        description="Enable automatic scheduling",
    )
    manual_trigger_only: bool = Field(
        default=True,
        description="Only allow manual triggers (no automatic runs)",
    )
    interval_hours: int = Field(
        default=4,
        description="Interval between automatic runs (if enabled)",
    )


class RetrainConfig(BaseModel):
    """Configuration for model retraining."""

    min_failed_samples: int = Field(
        default=50,
        description="Minimum number of failed samples required to trigger retraining",
    )
    augmentation_multiplier: int = Field(
        default=3,
        description="Number of augmented variants per failed sample",
    )
    merge_with_original: bool = Field(
        default=True,
        description="Merge augmented data with original training data",
    )
    original_data_path: Path | None = Field(
        default=None,
        description="Path to original training data for merging",
    )
    max_augmented_samples: int = Field(
        default=5000,
        description="Maximum number of augmented samples to include",
    )


class EvaluationConfig(BaseModel):
    """Configuration for Champion/Challenger evaluation."""

    traffic_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Ratio of traffic to route to challenger",
    )
    min_samples: int = Field(
        default=1000,
        description="Minimum samples before making promotion decision",
    )
    significance_level: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Statistical significance level (p-value threshold)",
    )
    min_improvement: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum improvement required for promotion (2%)",
    )
    max_evaluation_hours: int = Field(
        default=48,
        description="Maximum hours to run A/B test before decision",
    )


class StorageConfig(BaseModel):
    """Configuration for data storage."""

    failed_samples_dir: Path = Field(
        default=Path("data/failed_samples"),
        description="Directory to store failed sample files",
    )
    augmented_data_dir: Path = Field(
        default=Path("data/augmented"),
        description="Directory to store augmented datasets",
    )
    model_output_dir: Path = Field(
        default=Path("models/pipeline"),
        description="Directory for trained models",
    )
    history_file: Path = Field(
        default=Path("data/pipeline_history.json"),
        description="File to store pipeline execution history",
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration."""

    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    experiment_name: str = Field(
        default="adversarial-pipeline",
        description="MLflow experiment name",
    )
    model_name: str = Field(
        default="evoguard-toxic-classifier",
        description="Registered model name in MLflow",
    )


class PipelineConfig(BaseModel):
    """Main configuration for Adversarial MLOps Pipeline."""

    attack: AttackConfig = Field(default_factory=AttackConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Model settings
    model_name: str = Field(
        default="bert-base-uncased",
        description="Base model for training/inference",
    )

    # Logging
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging",
    )

    def model_dump_yaml(self) -> str:
        """Export configuration as YAML string."""
        import yaml

        return yaml.dump(self.model_dump(), default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Load configuration from dictionary."""
        return cls(**data)
