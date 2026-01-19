"""Data preprocessing pipeline for training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_service.training.config import DataConfig, TrainingConfig

logger = logging.getLogger(__name__)

# These imports are optional (only needed when training)
HAS_TRAINING_DEPS = False
try:
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, PreTrainedTokenizer

    HAS_TRAINING_DEPS = True
except ImportError:
    pass

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from transformers import PreTrainedTokenizer


class TextDataset:
    """Wrapper for text classification dataset.

    Handles loading from various formats (CSV, JSON, Parquet) and
    preprocessing for transformer models.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: "PreTrainedTokenizer | None" = None,
        max_length: int = 256,
    ) -> None:
        """Initialize the dataset.

        Args:
            texts: List of text samples.
            labels: List of integer labels.
            tokenizer: HuggingFace tokenizer (optional).
            max_length: Maximum sequence length for tokenization.
        """
        if not HAS_TRAINING_DEPS:
            raise ImportError(
                "Training dependencies not installed. "
                "Install with: uv pip install --group training"
            )

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": label,
            }

        return {"text": text, "labels": label}

    def to_hf_dataset(self) -> "Dataset":
        """Convert to HuggingFace Dataset format."""
        return Dataset.from_dict({"text": self.texts, "label": self.labels})


class DataProcessor:
    """Data preprocessing pipeline for QLoRA training.

    Handles:
    - Loading data from various file formats
    - Tokenization with HuggingFace tokenizers
    - Train/eval/test splitting
    - Batch processing
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        data_config: DataConfig | None = None,
    ) -> None:
        """Initialize the data processor.

        Args:
            training_config: Training configuration.
            data_config: Data configuration (optional).
        """
        if not HAS_TRAINING_DEPS:
            raise ImportError(
                "Training dependencies not installed. "
                "Install with: uv pip install --group training"
            )

        self.training_config = training_config
        self.data_config = data_config or DataConfig()
        self.tokenizer: PreTrainedTokenizer | None = None

    def load_tokenizer(self) -> "PreTrainedTokenizer":
        """Load tokenizer from HuggingFace.

        Returns:
            Loaded tokenizer.
        """
        logger.info(f"Loading tokenizer: {self.training_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.training_config.model_name,
            use_fast=True,
        )
        return self.tokenizer

    def load_data(
        self,
        file_path: Path | str,
        text_column: str | None = None,
        label_column: str | None = None,
    ) -> pd.DataFrame:
        """Load data from file.

        Supports CSV, JSON, and Parquet formats.

        Args:
            file_path: Path to the data file.
            text_column: Name of the text column.
            label_column: Name of the label column.

        Returns:
            DataFrame with text and label columns.
        """
        file_path = Path(file_path)
        text_col = text_column or self.data_config.text_column
        label_col = label_column or self.data_config.label_column

        logger.info(f"Loading data from: {file_path}")

        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Validate columns
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")

        # Rename to standard columns
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]

        # Convert labels to integers if needed
        if df["label"].dtype == "object":
            unique_labels = df["label"].unique()
            label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
            df["label"] = df["label"].map(label_map)
            logger.info(f"Label mapping: {label_map}")

        logger.info(f"Loaded {len(df)} samples")
        return df

    def prepare_datasets(
        self,
        df: pd.DataFrame,
        shuffle: bool | None = None,
    ) -> "DatasetDict":
        """Prepare train/eval/test datasets.

        Args:
            df: DataFrame with text and label columns.
            shuffle: Whether to shuffle before splitting.

        Returns:
            DatasetDict with train, validation, and test splits.
        """
        shuffle = shuffle if shuffle is not None else self.data_config.shuffle

        if shuffle:
            df = df.sample(frac=1, random_state=self.training_config.seed).reset_index(drop=True)

        # Limit samples if specified
        if self.data_config.max_samples is not None:
            df = df.head(self.data_config.max_samples)
            logger.info(f"Limited to {len(df)} samples")

        # Split data
        train_df, temp_df = train_test_split(
            df,
            train_size=self.data_config.train_split,
            random_state=self.training_config.seed,
            stratify=df["label"],
        )

        # Calculate relative sizes for eval/test from remaining data
        eval_ratio = self.data_config.eval_split / (
            self.data_config.eval_split + self.data_config.test_split
        )

        eval_df, test_df = train_test_split(
            temp_df,
            train_size=eval_ratio,
            random_state=self.training_config.seed,
            stratify=temp_df["label"],
        )

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Eval: {len(eval_df)}, Test: {len(test_df)}"
        )

        # Convert to HuggingFace datasets
        datasets = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, preserve_index=False),
                "validation": Dataset.from_pandas(eval_df, preserve_index=False),
                "test": Dataset.from_pandas(test_df, preserve_index=False),
            }
        )

        return datasets

    def tokenize_datasets(self, datasets: "DatasetDict") -> "DatasetDict":
        """Tokenize all datasets.

        Args:
            datasets: DatasetDict with raw text data.

        Returns:
            DatasetDict with tokenized data.
        """
        if self.tokenizer is None:
            self.load_tokenizer()

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return self.tokenizer(  # type: ignore[misc]
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.training_config.max_length,
            )

        logger.info("Tokenizing datasets...")
        tokenized = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        return tokenized

    def prepare_from_file(self, file_path: Path | str) -> "DatasetDict":
        """Full pipeline: load, split, and tokenize data from file.

        Args:
            file_path: Path to the data file.

        Returns:
            Tokenized DatasetDict ready for training.
        """
        df = self.load_data(file_path)
        datasets = self.prepare_datasets(df)
        tokenized = self.tokenize_datasets(datasets)
        return tokenized

    def prepare_from_battle_data(self, battle_records: list[dict[str, Any]]) -> "DatasetDict":
        """Prepare training data from battle records.

        Battle records contain adversarial examples that were misclassified,
        which are valuable for improving model robustness.

        Args:
            battle_records: List of battle records with 'original_text',
                          'mutated_text', and 'defender_detected' fields.

        Returns:
            Tokenized DatasetDict ready for training.
        """
        texts = []
        labels = []

        for record in battle_records:
            # Original text (toxic)
            if "original_text" in record:
                texts.append(record["original_text"])
                labels.append(1)  # toxic

            # Mutated text that evaded detection (still toxic)
            if "mutated_text" in record and not record.get("defender_detected", True):
                texts.append(record["mutated_text"])
                labels.append(1)  # toxic (misclassified as non-toxic)

        df = pd.DataFrame({"text": texts, "label": labels})
        logger.info(f"Prepared {len(df)} samples from battle data")

        datasets = self.prepare_datasets(df)
        tokenized = self.tokenize_datasets(datasets)
        return tokenized
