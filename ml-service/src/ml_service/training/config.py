"""Training configuration for QLoRA fine-tuning."""

from pathlib import Path

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA-specific configuration.

    QLoRA uses 4-bit quantization with LoRA adapters for memory-efficient
    fine-tuning on consumer GPUs (8GB VRAM).
    """

    r: int = Field(default=16, description="LoRA rank (lower = less memory)")
    lora_alpha: int = Field(default=32, description="LoRA scaling factor")
    lora_dropout: float = Field(default=0.05, description="Dropout probability")
    target_modules: list[str] = Field(
        default=["query", "key", "value", "dense"],
        description="Modules to apply LoRA to",
    )
    bias: str = Field(default="none", description="Bias handling: none, all, lora_only")
    task_type: str = Field(default="SEQ_CLS", description="Task type for PEFT")


class TrainingConfig(BaseModel):
    """Configuration for QLoRA fine-tuning.

    Optimized for 8GB VRAM with gradient checkpointing and small batch sizes.
    """

    # Model settings
    model_name: str = Field(
        default="bert-base-uncased",
        description="Base model from HuggingFace",
    )
    num_labels: int = Field(default=2, description="Number of classification labels")
    max_length: int = Field(default=256, description="Maximum sequence length")

    # Training hyperparameters
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    batch_size: int = Field(default=4, description="Training batch size (keep small for VRAM)")
    eval_batch_size: int = Field(default=8, description="Evaluation batch size")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_ratio: float = Field(default=0.1, description="Warmup ratio for scheduler")
    weight_decay: float = Field(default=0.01, description="Weight decay for regularization")
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Gradient accumulation steps (effective batch = batch_size * this)",
    )

    # Memory optimization
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing to reduce memory",
    )
    use_4bit_quantization: bool = Field(
        default=True,
        description="Enable 4-bit quantization for QLoRA",
    )
    bnb_4bit_compute_dtype: str = Field(
        default="float16",
        description="Compute dtype for 4-bit quantization",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="Quantization type: nf4 or fp4",
    )

    # LoRA configuration
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Paths
    output_dir: Path = Field(
        default=Path("models/checkpoints"),
        description="Directory for saving checkpoints",
    )
    logging_dir: Path = Field(
        default=Path("logs/training"),
        description="Directory for training logs",
    )

    # Evaluation
    eval_strategy: str = Field(default="epoch", description="Evaluation strategy")
    save_strategy: str = Field(default="epoch", description="Save strategy")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end")
    metric_for_best_model: str = Field(default="f1", description="Metric for best model selection")

    # MLflow
    mlflow_experiment_name: str = Field(
        default="evoguard-toxic-classifier",
        description="MLflow experiment name",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )

    # Early stopping
    early_stopping_patience: int = Field(
        default=3,
        description="Early stopping patience (epochs without improvement)",
    )

    # Seed
    seed: int = Field(default=42, description="Random seed for reproducibility")


class DataConfig(BaseModel):
    """Configuration for data processing."""

    train_file: Path | None = Field(default=None, description="Path to training data")
    eval_file: Path | None = Field(default=None, description="Path to evaluation data")
    test_file: Path | None = Field(default=None, description="Path to test data")

    text_column: str = Field(default="text", description="Column name for text data")
    label_column: str = Field(default="label", description="Column name for labels")

    train_split: float = Field(default=0.8, description="Training split ratio")
    eval_split: float = Field(default=0.1, description="Evaluation split ratio")
    test_split: float = Field(default=0.1, description="Test split ratio")

    max_samples: int | None = Field(default=None, description="Max samples to use (for debugging)")
    shuffle: bool = Field(default=True, description="Shuffle data before splitting")
