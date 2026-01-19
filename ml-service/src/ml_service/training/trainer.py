"""QLoRA trainer for fine-tuning transformer models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_service.training.config import TrainingConfig

logger = logging.getLogger(__name__)

# Optional imports for training
HAS_TRAINING_DEPS = False
try:
    import torch
    from datasets import DatasetDict
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    HAS_TRAINING_DEPS = True
except ImportError:
    pass

if TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers import Trainer

    from ml_service.training.mlflow_utils import MLflowTracker


class QLoRATrainer:
    """QLoRA fine-tuning trainer for text classification.

    Uses 4-bit quantization with LoRA adapters for memory-efficient
    training on consumer GPUs.

    Example:
        >>> config = TrainingConfig(model_name="bert-base-uncased")
        >>> trainer = QLoRATrainer(config)
        >>> trainer.setup_model()
        >>> results = trainer.train(tokenized_datasets)
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the QLoRA trainer.

        Args:
            config: Training configuration.
        """
        if not HAS_TRAINING_DEPS:
            raise ImportError(
                "Training dependencies not installed. "
                "Install with: uv pip install --group training"
            )

        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Trainer | None = None
        self.mlflow_tracker: MLflowTracker | None = None

    def setup_model(self) -> None:
        """Set up model with QLoRA configuration.

        Loads the base model with 4-bit quantization and applies LoRA adapters.
        """
        logger.info(f"Setting up model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )

        # Set up quantization config for QLoRA
        if self.config.use_4bit_quantization:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization (QLoRA)")
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Prepare model for k-bit training
        if self.config.use_4bit_quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        # Set up LoRA config
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=TaskType.SEQ_CLS,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("Model setup complete")

    def compute_metrics(self, eval_pred: Any) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            eval_pred: Evaluation predictions from Trainer.

        Returns:
            Dictionary of metrics.
        """
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
        }

    def train(
        self,
        datasets: "DatasetDict",
        run_name: str | None = None,
        use_mlflow: bool = True,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            datasets: Tokenized DatasetDict with train/validation splits.
            run_name: Optional name for the training run.
            use_mlflow: Whether to use MLflow tracking.

        Returns:
            Training results including metrics.
        """
        if self.model is None:
            self.setup_model()

        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.logging_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            logging_dir=str(self.config.logging_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=True,
            logging_steps=10,
            save_total_limit=3,
            seed=self.config.seed,
            fp16=torch.cuda.is_available(),
            report_to="mlflow" if use_mlflow else "none",
            remove_unused_columns=True,
        )

        # Callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        # Train with MLflow tracking
        if use_mlflow:
            self.mlflow_tracker = MLflowTracker(self.config)
            with self.mlflow_tracker.start_run(run_name=run_name):
                logger.info("Starting training...")
                train_result = self.trainer.train()

                # Log final metrics
                metrics = self.trainer.evaluate()
                self.mlflow_tracker.log_metrics(metrics)

                # Save and log model
                self.save_model()
                self.mlflow_tracker.log_artifacts(self.config.output_dir, "model")
        else:
            logger.info("Starting training (without MLflow)...")
            train_result = self.trainer.train()
            metrics = self.trainer.evaluate()
            self.save_model()

        results = {
            "train_loss": train_result.training_loss,
            "eval_metrics": metrics,
            "model_path": str(self.config.output_dir),
        }

        logger.info(f"Training complete. Results: {results}")
        return results

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            dataset: Dataset to evaluate on.

        Returns:
            Evaluation metrics.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")

        return self.trainer.evaluate(eval_dataset=dataset)

    def save_model(self, path: Path | str | None = None) -> Path:
        """Save the trained model.

        Args:
            path: Optional path to save to. Uses config output_dir if not specified.

        Returns:
            Path where model was saved.
        """
        save_path = Path(path) if path else self.config.output_dir

        # Save PEFT model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to: {save_path}")
        return save_path

    def load_model(self, path: Path | str) -> None:
        """Load a trained model.

        Args:
            path: Path to the saved model.
        """
        from peft import PeftModel

        path = Path(path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            device_map="auto",
        )

        # Load PEFT model
        self.model = PeftModel.from_pretrained(base_model, path)

        logger.info(f"Model loaded from: {path}")

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Make predictions on a list of texts.

        Args:
            texts: List of texts to classify.

        Returns:
            List of prediction dictionaries with label and confidence.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call setup_model() or load_model() first.")

        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i, prob in enumerate(probs):
            label = prob.argmax().item()
            confidence = prob[label].item()
            results.append(
                {
                    "text": texts[i],
                    "label": label,
                    "label_name": "toxic" if label == 1 else "non-toxic",
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "non-toxic": round(prob[0].item(), 4),
                        "toxic": round(prob[1].item(), 4),
                    },
                }
            )

        return results
