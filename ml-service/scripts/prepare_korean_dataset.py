#!/usr/bin/env python3
"""Prepare Korean hate speech dataset by merging multiple sources.

Sources:
1. BEEP! (Korean HateSpeech) - 연예뉴스 댓글
2. UnSmile (Smilegate AI) - 다중 레이블 혐오표현
3. Curse-detection-data - 커뮤니티 댓글

Output: Unified CSV with (text, label) where label=1 is toxic
"""

import csv
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "korean"


def clean_text(text: str) -> str:
    """Clean text for training."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove very short texts
    if len(text) < 3:
        return ""
    return text


def load_beep():
    """Load BEEP! dataset.
    
    Columns: comments, contain_gender_bias, bias, hate
    hate values: hate, offensive, none
    """
    samples = []
    
    for filename in ["beep_train.tsv", "beep_dev.tsv"]:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
            
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                text = clean_text(row.get("comments", ""))
                if not text:
                    continue
                    
                hate = row.get("hate", "none")
                # hate, offensive -> 1 (toxic), none -> 0
                label = 1 if hate in ("hate", "offensive") else 0
                samples.append((text, label))
    
    print(f"BEEP!: {len(samples)} samples")
    return samples


def load_unsmile():
    """Load UnSmile dataset.
    
    Multiple binary columns for hate categories.
    clean=1 means non-toxic.
    """
    samples = []
    
    # 혐오 카테고리 컬럼들
    hate_columns = [
        "여성/가족", "남성", "성소수자", "인종/국적", 
        "연령", "지역", "종교", "기타 혐오", "악플/욕설"
    ]
    
    for filename in ["unsmile_train.tsv", "unsmile_valid.tsv"]:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
            
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                text = clean_text(row.get("문장", ""))
                if not text:
                    continue
                
                # Check if any hate category is 1
                is_toxic = any(row.get(col, "0") == "1" for col in hate_columns)
                label = 1 if is_toxic else 0
                samples.append((text, label))
    
    print(f"UnSmile: {len(samples)} samples")
    return samples


def load_curse():
    """Load Curse-detection dataset.
    
    Format: text|label (pipe separated)
    """
    samples = []
    filepath = DATA_DIR / "curse_dataset.txt"
    
    if not filepath.exists():
        return samples
        
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
                
            parts = line.rsplit("|", 1)
            if len(parts) != 2:
                continue
                
            text = clean_text(parts[0])
            if not text:
                continue
                
            try:
                label = int(parts[1])
            except ValueError:
                continue
                
            samples.append((text, label))
    
    print(f"Curse: {len(samples)} samples")
    return samples


def deduplicate(samples):
    """Remove duplicate texts."""
    seen = set()
    unique = []
    
    for text, label in samples:
        if text not in seen:
            seen.add(text)
            unique.append((text, label))
    
    return unique


def balance_dataset(samples, max_ratio=2.0):
    """Balance dataset so majority class is at most max_ratio times minority."""
    counter = Counter(label for _, label in samples)
    print(f"Before balance: {dict(counter)}")
    
    min_count = min(counter.values())
    max_count = int(min_count * max_ratio)
    
    balanced = []
    counts = {0: 0, 1: 0}
    
    # Shuffle for randomness
    import random
    random.seed(42)
    random.shuffle(samples)
    
    for text, label in samples:
        if counts[label] < max_count:
            balanced.append((text, label))
            counts[label] += 1
    
    print(f"After balance: {counts}")
    return balanced


def main():
    print("=" * 60)
    print("한국어 혐오 표현 데이터셋 준비")
    print("=" * 60)
    
    # Load all datasets
    all_samples = []
    all_samples.extend(load_beep())
    all_samples.extend(load_unsmile())
    all_samples.extend(load_curse())
    
    print(f"\nTotal raw: {len(all_samples)} samples")
    
    # Deduplicate
    unique_samples = deduplicate(all_samples)
    print(f"After dedup: {len(unique_samples)} samples")
    
    # Check distribution
    counter = Counter(label for _, label in unique_samples)
    print(f"Distribution: toxic={counter[1]}, non-toxic={counter[0]}")
    
    # Balance (optional, for training)
    balanced_samples = balance_dataset(unique_samples, max_ratio=1.5)
    
    # Save full dataset
    full_path = DATA_DIR / "korean_hate_speech_full.csv"
    with open(full_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in unique_samples:
            writer.writerow([text, label])
    print(f"\nSaved full dataset: {full_path}")
    
    # Save balanced dataset
    balanced_path = DATA_DIR / "korean_hate_speech_balanced.csv"
    with open(balanced_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in balanced_samples:
            writer.writerow([text, label])
    print(f"Saved balanced dataset: {balanced_path}")
    
    # Show examples
    print("\n" + "=" * 60)
    print("예시 데이터 (toxic)")
    print("=" * 60)
    toxic_examples = [(t, l) for t, l in balanced_samples if l == 1][:5]
    for text, label in toxic_examples:
        print(f"  [{label}] {text[:60]}...")
    
    print("\n" + "=" * 60)
    print("예시 데이터 (non-toxic)")
    print("=" * 60)
    clean_examples = [(t, l) for t, l in balanced_samples if l == 0][:5]
    for text, label in clean_examples:
        print(f"  [{label}] {text[:60]}...")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
