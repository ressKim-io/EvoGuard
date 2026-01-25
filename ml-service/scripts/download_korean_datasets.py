#!/usr/bin/env python3
"""Download K-HATERS and K-MHaS Korean hate speech datasets.

K-HATERS: 192K news comments with 4-level labels
K-MHaS: 109K utterances with 8 multi-label categories
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_khaters(output_dir: Path) -> dict:
    """Download K-HATERS dataset from HuggingFace.

    Labels:
    - L2_hate: Severe hate speech
    - L1_hate: Mild hate speech
    - Offensive: Offensive language
    - Normal: Normal text

    Args:
        output_dir: Directory to save the dataset

    Returns:
        dict with dataset statistics
    """
    from datasets import load_dataset

    logger.info("Downloading K-HATERS dataset...")
    dataset = load_dataset("humane-lab/K-HATERS")

    stats = {
        "train": len(dataset["train"]),
        "validation": len(dataset["validation"]),
        "test": len(dataset["test"]),
    }

    # Save as CSV
    khaters_dir = output_dir / "K-HATERS"
    khaters_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        df = dataset[split].to_pandas()
        df.to_csv(khaters_dir / f"{split}.csv", index=False)
        logger.info(f"K-HATERS {split}: {len(df)} samples saved")

    logger.info(f"K-HATERS total: {sum(stats.values())} samples")
    return stats


def download_kmhas(output_dir: Path) -> dict:
    """Download K-MHaS dataset from GitHub.

    Labels (multi-label, tab-separated .txt files):
    - 0: Politics
    - 1: Origin
    - 2: Physical
    - 3: Age
    - 4: Gender
    - 5: Religion
    - 6: Race
    - 7: Profanity
    - 8: Not Hate Speech

    Args:
        output_dir: Directory to save the dataset

    Returns:
        dict with dataset statistics
    """
    import urllib.request
    import pandas as pd

    logger.info("Downloading K-MHaS dataset from GitHub...")

    base_url = "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data"
    kmhas_dir = output_dir / "K-MHaS"
    kmhas_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    files = {
        "train": "kmhas_train.txt",
        "validation": "kmhas_valid.txt",
    }

    for split, filename in files.items():
        url = f"{base_url}/{filename}"
        temp_path = kmhas_dir / filename
        csv_path = kmhas_dir / f"{split}.csv"

        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, temp_path)

        # Parse tab-separated file
        df = pd.read_csv(temp_path, sep="\t", names=["text", "label"], skiprows=1)

        # Clean text (remove quotes)
        df["text"] = df["text"].str.strip('"')

        # Save as CSV
        df.to_csv(csv_path, index=False)
        temp_path.unlink()  # Remove temp file

        stats[split] = len(df)
        logger.info(f"K-MHaS {split}: {len(df)} samples saved")

    logger.info(f"K-MHaS total: {sum(stats.values())} samples")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download Korean hate speech datasets")
    parser.add_argument(
        "--output_dir",
        default="data/korean",
        help="Directory to save datasets",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "khaters", "kmhas"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Korean Hate Speech Dataset Downloader")
    logger.info("=" * 60)

    total_stats = {}

    if args.dataset in ["all", "khaters"]:
        try:
            total_stats["K-HATERS"] = download_khaters(output_dir)
        except Exception as e:
            logger.error(f"Failed to download K-HATERS: {e}")

    if args.dataset in ["all", "kmhas"]:
        try:
            total_stats["K-MHaS"] = download_kmhas(output_dir)
        except Exception as e:
            logger.error(f"Failed to download K-MHaS: {e}")

    logger.info("=" * 60)
    logger.info("Download Summary:")
    for name, stats in total_stats.items():
        total = sum(stats.values())
        logger.info(f"  {name}: {total:,} samples")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
