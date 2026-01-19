"""Dataset loaders for public toxic text datasets.

Supports loading from HuggingFace Hub:
- google/jigsaw_toxicity_pred (Official Jigsaw dataset)
- Arsive/toxicity_classification_jigsaw (Balanced version)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Optional imports
HAS_DATASETS = False
try:
    from datasets import Dataset, DatasetDict, load_dataset

    HAS_DATASETS = True
except ImportError:
    pass

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


class JigsawDatasetLoader:
    """Load Jigsaw Toxic Comment datasets from HuggingFace.

    Available datasets:
    - 'jigsaw': google/jigsaw_toxicity_pred (159k train samples)
    - 'jigsaw_balanced': Arsive/toxicity_classification_jigsaw (balanced)

    Example:
        >>> loader = JigsawDatasetLoader()
        >>> datasets = loader.load('jigsaw', max_samples=10000)
        >>> print(datasets['train'][0])
    """

    DATASETS = {
        "jigsaw": {
            "name": "Arsive/toxicity_classification_jigsaw",
            "text_column": "comment_text",
            "label_column": "toxic",
            "split_mapping": {"train": "train", "test": "test"},
        },
        "jigsaw_balanced": {
            "name": "SetFit/toxic_conversations_50k",
            "text_column": "text",
            "label_column": "label",
            "split_mapping": {"train": "train", "test": "test"},
        },
        "toxic_tweets": {
            "name": "cardiffnlp/tweet_topic_single",
            "text_column": "text",
            "label_column": "label",
            "split_mapping": {"train": "train_coling2022", "test": "test_coling2022"},
        },
    }

    def __init__(self) -> None:
        """Initialize the dataset loader."""
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library not installed. "
                "Install with: uv pip install --group training"
            )

    def list_available(self) -> list[str]:
        """List available dataset names.

        Returns:
            List of dataset identifiers.
        """
        return list(self.DATASETS.keys())

    def load(
        self,
        dataset_name: str = "jigsaw",
        max_samples: int | None = None,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """Load a toxic text dataset.

        Args:
            dataset_name: Name of dataset ('jigsaw', 'jigsaw_balanced', etc.)
            max_samples: Maximum samples to load (None for all).
            test_size: Fraction for test split if not provided.
            seed: Random seed for shuffling/splitting.

        Returns:
            DatasetDict with train/test splits.
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {self.list_available()}"
            )

        config = self.DATASETS[dataset_name]
        logger.info(f"Loading dataset: {config['name']}")

        # Load from HuggingFace
        raw_dataset = load_dataset(config["name"])

        # Process each split
        processed_splits: dict[str, Dataset] = {}

        for target_split, source_split in config["split_mapping"].items():
            if source_split not in raw_dataset:
                logger.warning(f"Split '{source_split}' not found, skipping")
                continue

            split_data = raw_dataset[source_split]

            # Limit samples if specified
            if max_samples is not None:
                split_data = split_data.shuffle(seed=seed).select(
                    range(min(max_samples, len(split_data)))
                )

            # Standardize columns
            split_data = self._standardize_columns(split_data, config)

            processed_splits[target_split] = split_data
            logger.info(f"Loaded {target_split}: {len(split_data)} samples")

        # Create test split from train if not provided
        if "test" not in processed_splits and "train" in processed_splits:
            logger.info(f"Creating test split ({test_size*100:.0f}%)")
            split = processed_splits["train"].train_test_split(
                test_size=test_size, seed=seed
            )
            processed_splits["train"] = split["train"]
            processed_splits["test"] = split["test"]

        # Create validation split from train
        if "train" in processed_splits:
            split = processed_splits["train"].train_test_split(
                test_size=0.1, seed=seed
            )
            processed_splits["train"] = split["train"]
            processed_splits["validation"] = split["test"]

        return DatasetDict(processed_splits)

    def _standardize_columns(
        self, dataset: Dataset, config: dict[str, Any]
    ) -> Dataset:
        """Standardize column names and types.

        Args:
            dataset: Raw dataset.
            config: Dataset configuration.

        Returns:
            Dataset with standardized 'text' and 'label' columns.
        """
        text_col = config["text_column"]
        label_col = config["label_column"]

        def process_example(example: dict[str, Any]) -> dict[str, Any]:
            text = example.get(text_col, "")
            label = example.get(label_col, 0)

            # Handle None or missing text
            if text is None:
                text = ""

            # Handle float labels (binarize at 0.5)
            if isinstance(label, float):
                label = 1 if label >= 0.5 else 0
            elif label is None:
                label = 0
            else:
                label = int(label)

            # Ensure label is binary (0 or 1)
            label = 1 if label > 0 else 0

            return {"text": str(text), "label": label}

        # Apply transformation
        dataset = dataset.map(process_example, remove_columns=dataset.column_names)

        # Filter out empty texts
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

        return dataset

    def load_and_tokenize(
        self,
        dataset_name: str = "jigsaw",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> DatasetDict:
        """Load and tokenize a dataset.

        Args:
            dataset_name: Name of dataset to load.
            tokenizer_name: HuggingFace tokenizer name.
            max_length: Maximum sequence length.
            max_samples: Maximum samples to load.
            seed: Random seed.

        Returns:
            Tokenized DatasetDict ready for training.
        """
        from transformers import AutoTokenizer

        # Load dataset
        datasets = self.load(dataset_name, max_samples=max_samples, seed=seed)

        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Tokenize
        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        logger.info("Tokenizing dataset...")
        tokenized = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        return tokenized


def get_sample_data(n_samples: int = 100, seed: int = 42) -> DatasetDict:
    """Get a small sample dataset for testing.

    Creates synthetic toxic/non-toxic examples.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        DatasetDict with train/validation/test splits.
    """
    import random

    random.seed(seed)

    toxic_templates = [
        "I hate you and everything about you",
        "You are so stupid and worthless",
        "Go away you idiot",
        "This is the worst thing ever",
        "You should be ashamed of yourself",
        "What a terrible person you are",
        "I can't believe how dumb this is",
        "You're an absolute moron",
        "This makes me so angry",
        "How can anyone be this stupid",
    ]

    non_toxic_templates = [
        "Thank you for sharing this",
        "I appreciate your perspective",
        "This is an interesting point",
        "I agree with your analysis",
        "Great work on this project",
        "Looking forward to more updates",
        "This is helpful information",
        "I learned something new today",
        "Thanks for the explanation",
        "This is well written",
    ]

    texts = []
    labels = []

    for i in range(n_samples):
        if i % 2 == 0:
            text = random.choice(toxic_templates)
            # Add some variation
            text = text + f" (comment {i})"
            labels.append(1)
        else:
            text = random.choice(non_toxic_templates)
            text = text + f" (comment {i})"
            labels.append(0)
        texts.append(text)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    # Create dataset
    from datasets import Dataset

    full_dataset = Dataset.from_dict({"text": list(texts), "label": list(labels)})

    # Split
    split1 = full_dataset.train_test_split(test_size=0.2, seed=seed)
    split2 = split1["train"].train_test_split(test_size=0.125, seed=seed)  # 0.1 of original

    return DatasetDict({
        "train": split2["train"],
        "validation": split2["test"],
        "test": split1["test"],
    })
