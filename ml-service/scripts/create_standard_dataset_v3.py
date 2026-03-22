#!/usr/bin/env python3
"""Create korean_standard_v3 dataset.

Priority order (highest quality first):
1. AI Hub Ethics (408K, government-verified, 국가 단위 검증)
2. KOLD (40K, EMNLP 2022, balanced, Naver+YouTube)
3. KPHC (20K, 2024, political hate)
4. KoMultiText (40K, NeurIPS 2023, SNS)
5. AI Hub Harmful Query (10K, government, all toxic)
6. Existing v1 data (after decontamination)

Steps:
1. Fix |0/|1 contamination in existing v1 data
2. Clean each new dataset (dedup, filter non-Korean, etc.)
3. Merge all with deduplication (new data takes priority)
4. Split into train/valid/test (preserve existing test set for fair comparison)
5. Generate metadata

Usage:
    python scripts/create_standard_dataset_v3.py
    python scripts/create_standard_dataset_v3.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "korean"

# Korean character detection
KOREAN_PATTERN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")


def has_korean(text: str, min_ratio: float = 0.1) -> bool:
    """Check if text contains sufficient Korean characters."""
    if not text or len(text.strip()) == 0:
        return False
    korean_chars = len(KOREAN_PATTERN.findall(text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return False
    return korean_chars / total_chars >= min_ratio


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove null bytes and control characters (keep newlines, tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


# =============================================================================
# 1. Fix existing v1 data
# =============================================================================
def fix_v1_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """Fix |0/|1 suffix contamination in v1 data.

    Texts ending with |0 or |1 had their label concatenated into the text.
    Strip the suffix and correct the label.
    """
    fixed_count = 0
    relabeled_count = 0

    for idx, row in df.iterrows():
        text = str(row["text"])
        if text.endswith("|0"):
            df.at[idx, "text"] = text[:-2].rstrip()
            if row["label"] == 1:
                df.at[idx, "label"] = 0
                relabeled_count += 1
            fixed_count += 1
        elif text.endswith("|1"):
            df.at[idx, "text"] = text[:-2].rstrip()
            # label=1 is likely correct for |1 suffix
            fixed_count += 1

    logger.info(f"  [v1 fix] Stripped |0/|1 suffix from {fixed_count} texts")
    logger.info(f"  [v1 fix] Relabeled {relabeled_count} texts from toxic→clean")
    return df


def load_and_fix_v1() -> pd.DataFrame:
    """Load and fix existing v1 training data."""
    path = DATA_DIR / "korean_standard_v1_train.csv"
    if not path.exists():
        logger.warning(f"v1 train not found: {path}")
        return pd.DataFrame(columns=["text", "label"])

    df = pd.read_csv(path)
    logger.info(f"[v1] Loaded {len(df)} samples")

    # Fix contamination
    df = fix_v1_contamination(df)

    # Clean texts
    df["text"] = df["text"].apply(clean_text)

    # Remove empty/short texts
    df = df[df["text"].str.len() >= 3].reset_index(drop=True)

    # Remove duplicates (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info(f"  [v1] Removed {before - len(df)} duplicates after fix")

    # Remove non-Korean
    korean_mask = df["text"].apply(has_korean)
    removed = (~korean_mask).sum()
    if removed > 0:
        df = df[korean_mask].reset_index(drop=True)
        logger.info(f"  [v1] Removed {removed} non-Korean texts")

    df["source"] = "v1_existing"
    logger.info(f"  [v1] Final: {len(df)} samples (toxic={df['label'].sum()}, clean={(df['label']==0).sum()})")
    return df[["text", "label", "source"]]


# =============================================================================
# 2. Load new datasets
# =============================================================================
def load_ethics_verification() -> pd.DataFrame:
    """Load AI Hub Ethics Verification dataset (국가 단위 검증, 최고 품질).

    451K sentences with is_immoral boolean label and 7 unethical types.
    """
    ethics_dir = DATA_DIR / "ethics_verification"
    if not ethics_dir.exists():
        logger.warning(f"Ethics verification dir not found: {ethics_dir}")
        return pd.DataFrame(columns=["text", "label", "source"])

    rows = []
    for jf in sorted(ethics_dir.rglob("*.json")):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            talksets = data if isinstance(data, list) else data.get("data", data.get("talksets", [data]))
            if isinstance(talksets, dict):
                talksets = [talksets]

            for ts in talksets:
                for s in ts.get("sentences", []):
                    # Prefer non-anonymized text, fall back to anonymized
                    text = clean_text(s.get("text", s.get("origin_text", "")))
                    is_immoral = s.get("is_immoral")

                    if text and len(text) >= 3 and is_immoral is not None:
                        rows.append({
                            "text": text,
                            "label": 1 if is_immoral else 0,
                        })
        except Exception as e:
            logger.warning(f"  Failed to parse {jf.name}: {e}")
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("[Ethics] No data loaded")
        return pd.DataFrame(columns=["text", "label", "source"])

    # Filter non-Korean
    korean_mask = df["text"].apply(has_korean)
    removed_kr = (~korean_mask).sum()
    df = df[korean_mask].reset_index(drop=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    df["source"] = "AIHub_Ethics"
    logger.info(
        f"[Ethics] {len(df)} samples "
        f"(toxic={df['label'].sum()}, clean={(df['label']==0).sum()}, "
        f"removed: {removed_kr} non-Korean, {before - len(df)} dupes)"
    )
    return df


def load_kold() -> pd.DataFrame:
    """Load KOLD dataset (EMNLP 2022, Naver+YouTube comments)."""
    path = DATA_DIR / "KOLD" / "data" / "kold_v1.json"
    if not path.exists():
        logger.warning(f"KOLD not found: {path}")
        return pd.DataFrame(columns=["text", "label", "source"])

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        text = clean_text(item.get("comment", ""))
        off = item.get("OFF")
        if text and len(text) >= 3 and off is not None:
            rows.append({"text": text, "label": 1 if off else 0})

    df = pd.DataFrame(rows)

    # Remove non-Korean
    korean_mask = df["text"].apply(has_korean)
    removed = (~korean_mask).sum()
    df = df[korean_mask].reset_index(drop=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    df["source"] = "KOLD"
    logger.info(
        f"[KOLD] {len(df)} samples "
        f"(toxic={df['label'].sum()}, clean={(df['label']==0).sum()}, "
        f"removed: {removed} non-Korean, {before - len(df)} dupes)"
    )
    return df


def load_kphc() -> pd.DataFrame:
    """Load KPHC dataset (political hate speech, 2024)."""
    path = DATA_DIR / "KPHC" / "KPHC dataset.csv"
    if not path.exists():
        logger.warning(f"KPHC not found: {path}")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.read_csv(path)

    # Use 'hate' column for label, also mark 'incitement' as toxic
    result = pd.DataFrame()
    result["text"] = df["contents"].apply(clean_text)
    result["label"] = ((df["hate"] == 1) | (df["incitement"] == 1)).astype(int)

    # Filter
    result = result[result["text"].str.len() >= 3].reset_index(drop=True)
    korean_mask = result["text"].apply(has_korean)
    removed_kr = (~korean_mask).sum()
    result = result[korean_mask].reset_index(drop=True)

    before = len(result)
    result = result.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    result["source"] = "KPHC"
    logger.info(
        f"[KPHC] {len(result)} samples "
        f"(toxic={result['label'].sum()}, clean={(result['label']==0).sum()}, "
        f"removed: {removed_kr} non-Korean, {before - len(result)} dupes)"
    )
    return result


def load_komultitext() -> pd.DataFrame:
    """Load KoMultiText (NeurIPS 2023, DC Inside comments)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Dasool/KoMultiText")
    except Exception as e:
        logger.warning(f"KoMultiText load failed: {e}")
        return pd.DataFrame(columns=["text", "label", "source"])

    # Combine train + test
    dfs = []
    for split in ["train", "test"]:
        split_df = ds[split].to_pandas()
        dfs.append(split_df)
    raw = pd.concat(dfs, ignore_index=True)

    # toxic = profanity OR any bias
    bias_cols = [
        "gender", "politics", "nation", "race", "region",
        "generation", "social_hierarchy", "appearance", "others",
    ]
    result = pd.DataFrame()
    result["text"] = raw["comment"].apply(clean_text)
    result["label"] = ((raw["profanity"] == 1) | (raw[bias_cols].max(axis=1) == 1)).astype(int)

    # Filter
    result = result[result["text"].str.len() >= 3].reset_index(drop=True)
    korean_mask = result["text"].apply(has_korean)
    removed_kr = (~korean_mask).sum()
    result = result[korean_mask].reset_index(drop=True)

    before = len(result)
    result = result.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    result["source"] = "KoMultiText"
    logger.info(
        f"[KoMultiText] {len(result)} samples "
        f"(toxic={result['label'].sum()}, clean={(result['label']==0).sum()}, "
        f"removed: {removed_kr} non-Korean, {before - len(result)} dupes)"
    )
    return result


def load_aihub() -> pd.DataFrame:
    """Load AI Hub harmful expression data (all toxic)."""
    aihub_dir = DATA_DIR / "aihub_harmful"
    if not aihub_dir.exists():
        logger.warning(f"AI Hub dir not found: {aihub_dir}")
        return pd.DataFrame(columns=["text", "label", "source"])

    texts = []

    # Parse JSON files
    for jf in aihub_dir.glob("*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
            for item in d.get("data", []):
                text = clean_text(item.get("instruct_text", ""))
                if text:
                    texts.append(text)
        except Exception:
            continue

    # Parse TXT files
    for tf in aihub_dir.glob("*.txt"):
        try:
            with open(tf, encoding="utf-8") as f:
                text = clean_text(f.read())
            if text:
                texts.append(text)
        except Exception:
            continue

    df = pd.DataFrame({"text": texts, "label": 1})  # All toxic

    # Filter
    df = df[df["text"].str.len() >= 3].reset_index(drop=True)
    korean_mask = df["text"].apply(has_korean)
    removed_kr = (~korean_mask).sum()
    df = df[korean_mask].reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    df["source"] = "AIHub"
    logger.info(
        f"[AIHub] {len(df)} samples (all toxic, "
        f"removed: {removed_kr} non-Korean, {before - len(df)} dupes)"
    )
    return df


# =============================================================================
# 3. Merge and create v3
# =============================================================================
def merge_datasets(
    new_dfs: list[pd.DataFrame],
    existing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all datasets with deduplication.

    New datasets take priority over existing data.
    When duplicate texts exist, keep the version from the higher-priority source.
    """
    # Concat new data first (priority order), then existing
    all_new = pd.concat(new_dfs, ignore_index=True)

    logger.info(f"\n[Merge] New data: {len(all_new)}, Existing: {len(existing_df)}")

    # Deduplicate within new data (keep first = higher priority)
    before = len(all_new)
    all_new = all_new.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info(f"  Cross-dataset dedup (new): removed {before - len(all_new)}")

    # Remove existing texts that overlap with new data
    new_texts = set(all_new["text"])
    existing_clean = existing_df[~existing_df["text"].isin(new_texts)].reset_index(drop=True)
    removed_overlap = len(existing_df) - len(existing_clean)
    logger.info(f"  Overlap with existing: removed {removed_overlap} from existing")

    # Combine
    merged = pd.concat([all_new, existing_clean], ignore_index=True)

    # Final dedup
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info(f"  Final dedup: removed {before - len(merged)}")

    logger.info(f"  Final merged: {len(merged)} samples")
    return merged


def create_splits(
    merged_df: pd.DataFrame,
    test_path: Path | None = None,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/valid/test splits.

    Preserves existing test set for fair comparison with previous models.
    """
    # Keep existing test set for consistency
    if test_path and test_path.exists():
        existing_test = pd.read_csv(test_path)
        existing_test["text"] = existing_test["text"].apply(clean_text)
        test_texts = set(existing_test["text"])

        # Remove test texts from merged data
        train_valid = merged_df[~merged_df["text"].isin(test_texts)].reset_index(drop=True)
        test_df = existing_test[["text", "label"]].copy()

        logger.info(f"  Preserved existing test set: {len(test_df)} samples")
        logger.info(f"  Remaining for train/valid: {len(train_valid)}")
    else:
        # Create new test split
        train_valid, test_df = train_test_split(
            merged_df, test_size=test_ratio,
            stratify=merged_df["label"], random_state=seed,
        )
        train_valid = train_valid.reset_index(drop=True)
        test_df = test_df[["text", "label"]].reset_index(drop=True)

    # Split train/valid
    train_df, valid_df = train_test_split(
        train_valid, test_size=valid_ratio,
        stratify=train_valid["label"], random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df[["text", "label"]].reset_index(drop=True)

    # Shuffle train
    train_out = train_df[["text", "label"]].sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_out, valid_df, test_df


def print_report(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> None:
    """Print dataset statistics report."""
    print("\n" + "=" * 70)
    print("KOREAN STANDARD V3 DATASET REPORT")
    print("=" * 70)

    for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        toxic = (df["label"] == 1).sum()
        clean = (df["label"] == 0).sum()
        print(f"\n  {name}: {len(df)} samples")
        print(f"    Toxic: {toxic} ({toxic/len(df):.1%})")
        print(f"    Clean: {clean} ({clean/len(df):.1%})")
        print(f"    Text length: min={df['text'].str.len().min()}, "
              f"max={df['text'].str.len().max()}, "
              f"mean={df['text'].str.len().mean():.1f}")

    total = len(train_df) + len(valid_df) + len(test_df)
    print(f"\n  Total: {total} samples")

    # Source distribution (from merged_df which has source column)
    if "source" in merged_df.columns:
        print("\n  Source distribution (merged):")
        for source, count in merged_df["source"].value_counts().items():
            pct = count / len(merged_df) * 100
            print(f"    {source}: {count} ({pct:.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Create korean_standard_v3 dataset")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't save")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation ratio")
    args = parser.parse_args()

    print("=" * 70)
    print("CREATING KOREAN STANDARD V3 DATASET")
    print("=" * 70)
    print("Priority: Ethics > KOLD > KPHC > KoMultiText > AIHub_Harmful > v1_existing")
    print()

    # Load all datasets (priority order)
    ethics_df = load_ethics_verification()
    kold_df = load_kold()
    kphc_df = load_kphc()
    kmt_df = load_komultitext()
    aihub_df = load_aihub()
    v1_df = load_and_fix_v1()

    # Merge with priority
    new_dfs = [ethics_df, kold_df, kphc_df, kmt_df, aihub_df]  # Priority order
    merged = merge_datasets(new_dfs, v1_df)

    # Create splits (preserve existing test set)
    test_path = DATA_DIR / "korean_standard_v1_test.csv"
    train_df, valid_df, test_df = create_splits(
        merged, test_path=test_path,
        valid_ratio=args.valid_ratio, seed=args.seed,
    )

    # Report
    print_report(train_df, valid_df, test_df, merged)

    if args.dry_run:
        print("\n[DRY RUN] No files saved.")
        return

    # Save
    output_prefix = DATA_DIR / "korean_standard_v3"

    train_df.to_csv(f"{output_prefix}_train.csv", index=False)
    valid_df.to_csv(f"{output_prefix}_valid.csv", index=False)
    test_df.to_csv(f"{output_prefix}_test.csv", index=False)

    # Metadata
    metadata = {
        "version": "v3",
        "created": pd.Timestamp.now().isoformat(),
        "seed": args.seed,
        "sources": {
            "AIHub_Ethics": {"count": len(ethics_df), "priority": 1, "description": "Government-verified ethics data (451K sentences)"},
            "KOLD": {"count": len(kold_df), "priority": 2, "description": "EMNLP 2022, Naver+YouTube"},
            "KPHC": {"count": len(kphc_df), "priority": 3, "description": "Political hate speech, 2024"},
            "KoMultiText": {"count": len(kmt_df), "priority": 4, "description": "NeurIPS 2023, DC Inside"},
            "AIHub_Harmful": {"count": len(aihub_df), "priority": 5, "description": "Government harmful query"},
            "v1_existing": {"count": len(v1_df), "priority": 6, "description": "Legacy data (decontaminated)"},
        },
        "splits": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
        },
        "decontamination": {
            "v1_pipe_suffix_fixed": True,
            "cross_dataset_deduped": True,
            "non_korean_filtered": True,
        },
        "label_distribution": {
            "train_toxic": int(train_df["label"].sum()),
            "train_clean": int((train_df["label"] == 0).sum()),
            "valid_toxic": int(valid_df["label"].sum()),
            "valid_clean": int((valid_df["label"] == 0).sum()),
            "test_toxic": int(test_df["label"].sum()),
            "test_clean": int((test_df["label"] == 0).sum()),
        },
    }

    with open(f"{output_prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:")
    print(f"  {output_prefix}_train.csv")
    print(f"  {output_prefix}_valid.csv")
    print(f"  {output_prefix}_test.csv")
    print(f"  {output_prefix}_metadata.json")


if __name__ == "__main__":
    main()
