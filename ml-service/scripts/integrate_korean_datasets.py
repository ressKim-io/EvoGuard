#!/usr/bin/env python3
"""Integrate all Korean hate speech datasets into a unified format.

Combines:
- K-HATERS (192K): 4-level labels -> binary
- K-MHaS (109K): 8 multi-labels -> binary
- Existing datasets: KOTOX, BEEP, UnSmile, curse, korean_hate_speech

Output: korean_combined_v2.csv (~370K samples)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_khaters(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load K-HATERS dataset and convert to binary labels.

    Label mapping:
    - L2_hate, L1_hate, Offensive -> 1 (toxic)
    - Normal -> 0 (clean)
    """
    khaters_dir = data_dir / "K-HATERS"

    if not khaters_dir.exists():
        logger.warning("K-HATERS not found. Run download_korean_datasets.py first.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(khaters_dir / "train.csv")
    valid_df = pd.read_csv(khaters_dir / "validation.csv")

    def to_binary(label: str) -> int:
        """Convert K-HATERS label to binary."""
        if label in ["L2_hate", "L1_hate", "Offensive"]:
            return 1
        return 0

    train_df["label"] = train_df["label"].apply(to_binary)
    valid_df["label"] = valid_df["label"].apply(to_binary)

    train_df = train_df.rename(columns={"text": "text"})[["text", "label"]]
    valid_df = valid_df.rename(columns={"text": "text"})[["text", "label"]]

    return train_df, valid_df


def load_kmhas(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load K-MHaS dataset and convert to binary labels.

    Label mapping (multi-label format, e.g., "2,4"):
    - Any of 0-7 hate categories -> 1 (toxic)
    - Only "8" (Not Hate Speech) -> 0 (clean)

    Categories: 0=Politics, 1=Origin, 2=Physical, 3=Age, 4=Gender, 5=Religion, 6=Race, 7=Profanity, 8=Not Hate
    """
    kmhas_dir = data_dir / "K-MHaS"

    if not kmhas_dir.exists():
        logger.warning("K-MHaS not found. Run download_korean_datasets.py first.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(kmhas_dir / "train.csv")
    valid_df = pd.read_csv(kmhas_dir / "validation.csv")

    def to_binary(label_str) -> int:
        """Convert K-MHaS multi-label to binary.

        Label is comma-separated string (e.g., "2,4" or "8").
        If only "8" -> 0 (clean), otherwise -> 1 (toxic).
        """
        if pd.isna(label_str):
            return 0
        labels = str(label_str).split(",")
        # If only label is "8", it's clean
        if labels == ["8"]:
            return 0
        return 1

    train_df["label"] = train_df["label"].apply(to_binary)
    valid_df["label"] = valid_df["label"].apply(to_binary)

    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]

    return train_df, valid_df


def load_kotox(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load KOTOX dataset."""
    kotox_dir = data_dir / "KOTOX" / "data" / "KOTOX_classification" / "total"

    if not kotox_dir.exists():
        logger.warning("KOTOX not found.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.read_csv(kotox_dir / "train.csv")
    valid_df = pd.read_csv(kotox_dir / "valid.csv")

    return train_df[["text", "label"]], valid_df[["text", "label"]]


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

    return train_df, valid_df


def load_curse(data_dir: Path) -> pd.DataFrame:
    """Load curse dataset (toxic only)."""
    curse_path = data_dir / "curse_dataset.txt"

    if not curse_path.exists():
        logger.warning("Curse dataset not found.")
        return pd.DataFrame()

    with open(curse_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return pd.DataFrame({"text": lines, "label": [1] * len(lines)})


def load_korean_hate(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load korean hate speech balanced dataset."""
    path = data_dir / "korean_hate_speech_balanced.csv"

    if not path.exists():
        logger.warning("Korean hate speech balanced not found.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * 0.9)
    train_df = df[:split_idx]
    valid_df = df[split_idx:]

    return train_df[["text", "label"]], valid_df[["text", "label"]]


def load_slang_toxic(data_dir: Path) -> pd.DataFrame:
    """Load slang toxic dataset."""
    path = data_dir / "slang_toxic.csv"

    if not path.exists():
        logger.warning("Slang toxic not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df[["text", "label"]]


def main():
    parser = argparse.ArgumentParser(description="Integrate Korean hate speech datasets")
    parser.add_argument("--output", default="korean_combined_v2.csv", help="Output filename")
    parser.add_argument("--data_dir", default="data/korean", help="Data directory")
    parser.add_argument("--include_new_only", action="store_true", help="Only include K-HATERS and K-MHaS")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / args.data_dir

    logger.info("=" * 60)
    logger.info("Korean Hate Speech Dataset Integration")
    logger.info("=" * 60)

    train_dfs = []
    valid_dfs = []
    stats = {}

    # Load new datasets
    khaters_train, khaters_valid = load_khaters(data_dir)
    if not khaters_train.empty:
        train_dfs.append(khaters_train)
        valid_dfs.append(khaters_valid)
        stats["K-HATERS"] = {"train": len(khaters_train), "valid": len(khaters_valid)}
        logger.info(f"K-HATERS: {len(khaters_train)} train, {len(khaters_valid)} valid")

    kmhas_train, kmhas_valid = load_kmhas(data_dir)
    if not kmhas_train.empty:
        train_dfs.append(kmhas_train)
        valid_dfs.append(kmhas_valid)
        stats["K-MHaS"] = {"train": len(kmhas_train), "valid": len(kmhas_valid)}
        logger.info(f"K-MHaS: {len(kmhas_train)} train, {len(kmhas_valid)} valid")

    # Load existing datasets (unless --include_new_only)
    if not args.include_new_only:
        kotox_train, kotox_valid = load_kotox(data_dir)
        if not kotox_train.empty:
            train_dfs.append(kotox_train)
            valid_dfs.append(kotox_valid)
            stats["KOTOX"] = {"train": len(kotox_train), "valid": len(kotox_valid)}
            logger.info(f"KOTOX: {len(kotox_train)} train, {len(kotox_valid)} valid")

        beep_train, beep_valid = load_beep(data_dir)
        if not beep_train.empty:
            train_dfs.append(beep_train)
            valid_dfs.append(beep_valid)
            stats["BEEP"] = {"train": len(beep_train), "valid": len(beep_valid)}
            logger.info(f"BEEP: {len(beep_train)} train, {len(beep_valid)} valid")

        unsmile_train, unsmile_valid = load_unsmile(data_dir)
        if not unsmile_train.empty:
            train_dfs.append(unsmile_train)
            valid_dfs.append(unsmile_valid)
            stats["UnSmile"] = {"train": len(unsmile_train), "valid": len(unsmile_valid)}
            logger.info(f"UnSmile: {len(unsmile_train)} train, {len(unsmile_valid)} valid")

        curse_df = load_curse(data_dir)
        if not curse_df.empty:
            train_dfs.append(curse_df)
            stats["Curse"] = {"train": len(curse_df), "valid": 0}
            logger.info(f"Curse: {len(curse_df)} samples")

        khate_train, khate_valid = load_korean_hate(data_dir)
        if not khate_train.empty:
            train_dfs.append(khate_train)
            valid_dfs.append(khate_valid)
            stats["KoreanHate"] = {"train": len(khate_train), "valid": len(khate_valid)}
            logger.info(f"KoreanHate: {len(khate_train)} train, {len(khate_valid)} valid")

        slang_df = load_slang_toxic(data_dir)
        if not slang_df.empty:
            train_dfs.append(slang_df)
            stats["SlangToxic"] = {"train": len(slang_df), "valid": 0}
            logger.info(f"SlangToxic: {len(slang_df)} samples")

    # Combine all datasets
    if not train_dfs:
        logger.error("No datasets found. Run download_korean_datasets.py first.")
        return

    combined_train = pd.concat(train_dfs, ignore_index=True)
    combined_valid = pd.concat([df for df in valid_dfs if not df.empty], ignore_index=True)

    # Clean data
    logger.info("Cleaning data...")
    combined_train = combined_train.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined_valid = combined_valid.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Remove validation samples from training
    valid_texts = set(combined_valid["text"].tolist())
    combined_train = combined_train[~combined_train["text"].isin(valid_texts)].reset_index(drop=True)

    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_valid = combined_valid.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save combined dataset
    output_path = data_dir / args.output
    combined_all = pd.concat([combined_train, combined_valid], ignore_index=True)
    combined_all.to_csv(output_path, index=False)

    # Save train/valid separately
    combined_train.to_csv(data_dir / "korean_combined_v2_train.csv", index=False)
    combined_valid.to_csv(data_dir / "korean_combined_v2_valid.csv", index=False)

    logger.info("=" * 60)
    logger.info("Integration Summary:")
    logger.info(f"  Total Train: {len(combined_train):,}")
    logger.info(f"  Total Valid: {len(combined_valid):,}")
    logger.info(f"  Total Combined: {len(combined_all):,}")
    logger.info(f"  Train label distribution: {combined_train['label'].value_counts().to_dict()}")
    logger.info(f"  Valid label distribution: {combined_valid['label'].value_counts().to_dict()}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    # Dataset breakdown
    logger.info("\nDataset Breakdown:")
    for name, s in stats.items():
        logger.info(f"  {name}: train={s['train']:,}, valid={s['valid']:,}")


if __name__ == "__main__":
    main()
