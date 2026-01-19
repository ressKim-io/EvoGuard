"""Tests for training configuration."""

from pathlib import Path

import pytest

from ml_service.training.config import DataConfig, LoRAConfig, TrainingConfig


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LoRAConfig()

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.target_modules == ["query", "key", "value", "dense"]
        assert config.bias == "none"
        assert config.task_type == "SEQ_CLS"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.model_name == "bert-base-uncased"
        assert config.num_labels == 2
        assert config.max_length == 256
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.num_epochs == 3
        assert config.use_gradient_checkpointing is True
        assert config.use_4bit_quantization is True
        assert config.seed == 42

    def test_custom_model(self) -> None:
        """Test custom model configuration."""
        config = TrainingConfig(
            model_name="distilbert-base-uncased",
            num_labels=3,
            max_length=128,
        )

        assert config.model_name == "distilbert-base-uncased"
        assert config.num_labels == 3
        assert config.max_length == 128

    def test_memory_optimization_settings(self) -> None:
        """Test memory optimization settings."""
        config = TrainingConfig(
            use_gradient_checkpointing=False,
            use_4bit_quantization=False,
            gradient_accumulation_steps=8,
        )

        assert config.use_gradient_checkpointing is False
        assert config.use_4bit_quantization is False
        assert config.gradient_accumulation_steps == 8

    def test_lora_config_nested(self) -> None:
        """Test nested LoRA configuration."""
        config = TrainingConfig(
            lora=LoRAConfig(r=32, lora_alpha=64),
        )

        assert config.lora.r == 32
        assert config.lora.lora_alpha == 64

    def test_paths_are_paths(self) -> None:
        """Test that path fields are Path objects."""
        config = TrainingConfig()

        assert isinstance(config.output_dir, Path)
        assert isinstance(config.logging_dir, Path)

    def test_mlflow_settings(self) -> None:
        """Test MLflow configuration."""
        config = TrainingConfig(
            mlflow_experiment_name="test-experiment",
            mlflow_tracking_uri="http://mlflow:5000",
        )

        assert config.mlflow_experiment_name == "test-experiment"
        assert config.mlflow_tracking_uri == "http://mlflow:5000"

    def test_early_stopping(self) -> None:
        """Test early stopping configuration."""
        config = TrainingConfig(early_stopping_patience=5)
        assert config.early_stopping_patience == 5

    def test_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        config = TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=4,
        )

        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        assert effective_batch_size == 16


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self) -> None:
        """Test default data configuration."""
        config = DataConfig()

        assert config.train_file is None
        assert config.text_column == "text"
        assert config.label_column == "label"
        assert config.train_split == 0.8
        assert config.eval_split == 0.1
        assert config.test_split == 0.1
        assert config.shuffle is True

    def test_split_ratios_sum(self) -> None:
        """Test that split ratios sum to 1.0."""
        config = DataConfig()
        total = config.train_split + config.eval_split + config.test_split
        assert total == pytest.approx(1.0)

    def test_custom_columns(self) -> None:
        """Test custom column names."""
        config = DataConfig(
            text_column="content",
            label_column="toxic",
        )

        assert config.text_column == "content"
        assert config.label_column == "toxic"

    def test_max_samples_for_debugging(self) -> None:
        """Test max_samples for debugging."""
        config = DataConfig(max_samples=100)
        assert config.max_samples == 100

    def test_file_paths(self) -> None:
        """Test file path configuration."""
        config = DataConfig(
            train_file=Path("data/train.csv"),
            eval_file=Path("data/eval.csv"),
        )

        assert config.train_file == Path("data/train.csv")
        assert config.eval_file == Path("data/eval.csv")
