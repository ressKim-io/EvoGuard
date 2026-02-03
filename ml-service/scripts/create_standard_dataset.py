#!/usr/bin/env python3
"""Create standardized dataset for fair model comparison.

This script creates korean_standard_v1 dataset by combining:
- KOTOX (verified, high-quality)
- BEEP (verified, high-quality)
- UnSmile (verified, high-quality)
- curse_dataset (small, toxic-only)
- korean_hate_speech_balanced (balanced)

EXCLUDES:
- K-HATERS (label conversion noise)
- K-MHaS (label conversion noise)

Output:
- korean_standard_v1_train.csv (80%)
- korean_standard_v1_valid.csv (10%)
- korean_standard_v1_test.csv (10%)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def load_kotox(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load KOTOX dataset (pre-split)."""
    kotox_dir = data_dir / "KOTOX" / "data" / "KOTOX_classification" / "total"

    if not kotox_dir.exists():
        logger.warning("KOTOX not found.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(kotox_dir / "train.csv")[["text", "label"]]
    valid_df = pd.read_csv(kotox_dir / "valid.csv")[["text", "label"]]

    logger.info(f"KOTOX: {len(train_df)} train, {len(valid_df)} valid")
    return train_df, valid_df


def load_beep(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BEEP dataset."""
    train_path = data_dir / "beep_train.tsv"
    dev_path = data_dir / "beep_dev.tsv"

    if not train_path.exists():
        logger.warning("BEEP not found.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(train_path, sep="\t")
    dev_df = pd.read_csv(dev_path, sep="\t")

    def to_binary(x):
        return 0 if x == "none" else 1

    train_df["label"] = train_df["hate"].apply(to_binary)
    dev_df["label"] = dev_df["hate"].apply(to_binary)

    train_df = train_df.rename(columns={"comments": "text"})[["text", "label"]]
    dev_df = dev_df.rename(columns={"comments": "text"})[["text", "label"]]

    logger.info(f"BEEP: {len(train_df)} train, {len(dev_df)} valid")
    return train_df, dev_df


def load_unsmile(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UnSmile dataset."""
    train_path = data_dir / "unsmile_train.tsv"
    valid_path = data_dir / "unsmile_valid.tsv"

    if not train_path.exists():
        logger.warning("UnSmile not found.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(train_path, sep="\t")
    valid_df = pd.read_csv(valid_path, sep="\t")

    train_df["label"] = 1 - train_df["clean"]
    valid_df["label"] = 1 - valid_df["clean"]

    train_df = train_df.rename(columns={"문장": "text"})[["text", "label"]]
    valid_df = valid_df.rename(columns={"문장": "text"})[["text", "label"]]

    logger.info(f"UnSmile: {len(train_df)} train, {len(valid_df)} valid")
    return train_df, valid_df


def load_curse(data_dir: Path) -> pd.DataFrame:
    """Load curse dataset (toxic only, no split)."""
    curse_path = data_dir / "curse_dataset.txt"

    if not curse_path.exists():
        logger.warning("Curse dataset not found.")
        return pd.DataFrame()

    with open(curse_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({"text": lines, "label": [1] * len(lines)})
    logger.info(f"Curse: {len(df)} samples (all toxic)")
    return df


def load_korean_hate_balanced(data_dir: Path) -> pd.DataFrame:
    """Load korean hate speech balanced dataset (no pre-split)."""
    path = data_dir / "korean_hate_speech_balanced.csv"

    if not path.exists():
        logger.warning("Korean hate speech balanced not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)[["text", "label"]]
    logger.info(f"Korean Hate Balanced: {len(df)} samples")
    return df


def load_slang_toxic(data_dir: Path) -> pd.DataFrame:
    """Load slang toxic dataset (generated during coevolution)."""
    path = data_dir / "slang_toxic.csv"

    if not path.exists():
        logger.warning("Slang toxic not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)[["text", "label"]]
    logger.info(f"Slang Toxic: {len(df)} samples")
    return df


def main():
    parser = argparse.ArgumentParser(description="Create standardized dataset")
    parser.add_argument("--data-dir", default="data/korean", help="Data directory")
    parser.add_argument("--output-prefix", default="korean_standard_v1", help="Output filename prefix")
    parser.add_argument("--include-slang", action="store_true", help="Include slang_toxic.csv")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation set ratio")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / args.data_dir

    logger.info("=" * 60)
    logger.info("Creating Standardized Dataset")
    logger.info("=" * 60)
    logger.info(f"Seed: {SEED}")
    logger.info(f"Test ratio: {args.test_ratio}")
    logger.info(f"Valid ratio: {args.valid_ratio}")
    logger.info("")

    # Load datasets with pre-existing splits
    kotox_train, kotox_valid = load_kotox(data_dir)
    beep_train, beep_valid = load_beep(data_dir)
    unsmile_train, unsmile_valid = load_unsmile(data_dir)

    # Load datasets without splits
    curse_df = load_curse(data_dir)
    khate_df = load_korean_hate_balanced(data_dir)

    # Optionally include slang toxic
    slang_df = pd.DataFrame()
    if args.include_slang:
        slang_df = load_slang_toxic(data_dir)

    # Combine pre-split valid sets
    valid_from_sources = pd.concat(
        [df for df in [kotox_valid, beep_valid, unsmile_valid] if not df.empty],
        ignore_index=True
    )

    # Combine datasets that need splitting
    needs_split = pd.concat(
        [df for df in [curse_df, khate_df, slang_df] if not df.empty],
        ignore_index=True
    )

    # Split needs_split into train/valid/test
    if not needs_split.empty:
        # First split: train vs (valid+test)
        train_part, temp = train_test_split(
            needs_split,
            test_size=args.test_ratio + args.valid_ratio,
            random_state=SEED,
            stratify=needs_split["label"]
        )

        # Second split: valid vs test
        valid_ratio_adjusted = args.valid_ratio / (args.test_ratio + args.valid_ratio)
        valid_part, test_part = train_test_split(
            temp,
            test_size=1 - valid_ratio_adjusted,
            random_state=SEED,
            stratify=temp["label"]
        )
    else:
        train_part = pd.DataFrame()
        valid_part = pd.DataFrame()
        test_part = pd.DataFrame()

    # Also create test set from pre-split sources (take from valid)
    if not valid_from_sources.empty:
        valid_keep, test_from_valid = train_test_split(
            valid_from_sources,
            test_size=0.5,  # Half of valid becomes test
            random_state=SEED,
            stratify=valid_from_sources["label"]
        )
    else:
        valid_keep = pd.DataFrame()
        test_from_valid = pd.DataFrame()

    # Combine all training data
    train_dfs = [kotox_train, beep_train, unsmile_train, train_part]
    combined_train = pd.concat([df for df in train_dfs if not df.empty], ignore_index=True)

    # Combine all validation data
    valid_dfs = [valid_keep, valid_part]
    combined_valid = pd.concat([df for df in valid_dfs if not df.empty], ignore_index=True)

    # Combine all test data
    test_dfs = [test_from_valid, test_part]
    combined_test = pd.concat([df for df in test_dfs if not df.empty], ignore_index=True)

    # Clean data: remove NaN and duplicates
    logger.info("\nCleaning data...")

    combined_train = combined_train.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined_valid = combined_valid.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined_test = combined_test.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Remove valid/test samples from training (prevent leakage)
    valid_texts = set(combined_valid["text"].tolist())
    test_texts = set(combined_test["text"].tolist())

    combined_train = combined_train[~combined_train["text"].isin(valid_texts | test_texts)].reset_index(drop=True)

    # Remove test samples from valid
    combined_valid = combined_valid[~combined_valid["text"].isin(test_texts)].reset_index(drop=True)

    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined_valid = combined_valid.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined_test = combined_test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save datasets
    train_path = data_dir / f"{args.output_prefix}_train.csv"
    valid_path = data_dir / f"{args.output_prefix}_valid.csv"
    test_path = data_dir / f"{args.output_prefix}_test.csv"

    combined_train.to_csv(train_path, index=False)
    combined_valid.to_csv(valid_path, index=False)
    combined_test.to_csv(test_path, index=False)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET CREATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train: {len(combined_train):,} samples")
    logger.info(f"  Label 0 (clean): {(combined_train['label'] == 0).sum():,}")
    logger.info(f"  Label 1 (toxic): {(combined_train['label'] == 1).sum():,}")
    logger.info(f"Valid: {len(combined_valid):,} samples")
    logger.info(f"  Label 0 (clean): {(combined_valid['label'] == 0).sum():,}")
    logger.info(f"  Label 1 (toxic): {(combined_valid['label'] == 1).sum():,}")
    logger.info(f"Test: {len(combined_test):,} samples")
    logger.info(f"  Label 0 (clean): {(combined_test['label'] == 0).sum():,}")
    logger.info(f"  Label 1 (toxic): {(combined_test['label'] == 1).sum():,}")
    logger.info("")
    logger.info(f"Output files:")
    logger.info(f"  {train_path}")
    logger.info(f"  {valid_path}")
    logger.info(f"  {test_path}")
    logger.info("=" * 60)

    # Save metadata
    metadata = {
        "version": "v1",
        "seed": SEED,
        "sources": ["KOTOX", "BEEP", "UnSmile", "curse_dataset", "korean_hate_speech_balanced"],
        "excluded": ["K-HATERS", "K-MHaS"],
        "train_size": len(combined_train),
        "valid_size": len(combined_valid),
        "test_size": len(combined_test),
        "train_label_dist": combined_train["label"].value_counts().to_dict(),
        "valid_label_dist": combined_valid["label"].value_counts().to_dict(),
        "test_label_dist": combined_test["label"].value_counts().to_dict(),
    }

    import json
    metadata_path = data_dir / f"{args.output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
