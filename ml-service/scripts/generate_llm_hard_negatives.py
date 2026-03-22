#!/usr/bin/env python3
"""Generate hard negative samples using LLM for implicit hate speech.

Takes FN (false negative) samples from model evaluation and generates
diverse variants using Claude API to improve coverage of subtle hate speech.

Variant styles:
    1. Sarcasm (비꼼)
    2. Metaphor (은유)
    3. Rhetorical question (수사의문)
    4. Coded language (코드어)
    5. Backhanded compliment (뒤통수칭찬)

Also generates hard benign counterparts for each FN to reduce FP.

Usage:
    # Generate from test set FN samples
    python scripts/generate_llm_hard_negatives.py

    # Custom model and output
    python scripts/generate_llm_hard_negatives.py \
      --production-model models/production \
      --variants-per-sample 5 \
      --output data/korean/hard_negatives.csv

    # From a custom FN list file
    python scripts/generate_llm_hard_negatives.py --fn-file data/korean/fn_samples.csv

    # Dry run (no API calls, just identify FN samples)
    python scripts/generate_llm_hard_negatives.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_service.training.standard_config import get_data_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HAS_ANTHROPIC = False
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass


# --- Prompts ---

IMPLICIT_VARIANT_SYSTEM = """\
당신은 한국어 혐오표현 탐지 시스템의 학습 데이터를 생성하는 연구 도구입니다.
주어진 독성 텍스트를 다양한 간접적/우회적 표현으로 변형하여, 모델이 은유적 혐오를 더 잘 탐지하도록 합니다.

중요: 생성된 텍스트는 여전히 독성(혐오)이어야 합니다. 의미를 보존하되 표현 방식만 변환하세요.
"""

IMPLICIT_VARIANT_USER = """\
다음 한국어 독성 텍스트의 의미를 보존하면서 {n}가지 간접적 변형을 생성하세요.

원본: "{text}"

각 변형은 다음 스타일 중 하나를 사용하세요:
1. 비꼼/풍자 (예: "정말 대단하시네요~")
2. 은유/비유 (예: 동물, 사물에 빗댐)
3. 수사의문문 (예: "이런 사람이 사람이라고?")
4. 코드어/완곡어법 (예: 커뮤니티 은어, 두문자어)
5. 뒤통수 칭찬 (예: "착하긴 한데...")

형식: 한 줄에 하나씩, 번호와 스타일 태그를 붙여 출력하세요.
예시:
[비꼼] 변형 텍스트
[은유] 변형 텍스트

변형만 출력하고 다른 설명은 하지 마세요.
"""

BENIGN_COUNTERPART_SYSTEM = """\
당신은 한국어 혐오표현 탐지 시스템의 False Positive를 줄이기 위한 연구 도구입니다.
주어진 독성 텍스트와 표면적으로 유사하지만 실제로는 무해한(정상) 텍스트를 생성합니다.
"""

BENIGN_COUNTERPART_USER = """\
다음 한국어 독성 텍스트와 비슷한 단어/구조를 사용하지만, 실제로는 무해한(비독성) 문장 {n}개를 생성하세요.

독성 원본: "{text}"

규칙:
- 비슷한 단어나 문장 구조를 사용하되, 혐오/차별 의미가 없어야 합니다
- 일상적 대화에서 자연스럽게 쓰일 수 있어야 합니다
- 다양한 맥락 (일상, 리뷰, 뉴스 댓글 등)

한 줄에 하나씩 출력하세요. 번호 없이 텍스트만 출력하세요.
"""


class SimpleDataset(Dataset):
    """Simple text dataset for inference."""

    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def find_fn_samples(
    texts: list[str],
    labels: list[int],
    model_path: str,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    """Find false negative samples from model predictions.

    FN = labeled toxic (1) but model predicts clean (0).

    Returns:
        DataFrame with FN samples, sorted by confidence (most confident FNs first).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = SimpleDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Finding FN samples"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    del model
    torch.cuda.empty_cache()

    all_probs = np.vstack(all_probs)

    # Find FN: label=1 but predicted=0
    fn_rows = []
    for i in range(len(texts)):
        pred = int(all_probs[i].argmax())
        if labels[i] == 1 and pred == 0:
            fn_rows.append({
                "index": i,
                "text": texts[i],
                "label": labels[i],
                "pred": pred,
                "toxic_prob": float(all_probs[i, 1]),
                "confidence": float(all_probs[i, 0]),  # confidence in wrong prediction
            })

    fn_df = pd.DataFrame(fn_rows)
    if not fn_df.empty:
        fn_df = fn_df.sort_values("confidence", ascending=False).reset_index(drop=True)

    logger.info(f"Found {len(fn_df)} FN samples out of {sum(1 for l in labels if l == 1)} toxic samples")
    return fn_df


def generate_variants_llm(
    text: str,
    client: "anthropic.Anthropic",
    n: int = 5,
    model: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.8,
) -> list[dict]:
    """Generate implicit hate speech variants using Claude API.

    Returns:
        List of dicts with 'text', 'style', 'label' keys.
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=IMPLICIT_VARIANT_SYSTEM,
            messages=[{"role": "user", "content": IMPLICIT_VARIANT_USER.format(text=text, n=n)}],
        )

        content = response.content[0].text.strip()
        variants = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse [style] text format
            style = "unknown"
            variant_text = line
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                style = line[1:bracket_end]
                variant_text = line[bracket_end + 1:].strip()

            # Strip numbering if present
            for prefix_len in range(1, 4):
                if len(variant_text) > prefix_len and variant_text[:prefix_len].rstrip(".)-:").isdigit():
                    variant_text = variant_text[prefix_len:].lstrip(".)-: ")
                    break

            if variant_text and variant_text != text:
                variants.append({
                    "text": variant_text,
                    "style": style,
                    "label": 1,  # toxic variants
                })

        return variants[:n]

    except Exception as e:
        logger.warning(f"API error generating variants: {e}")
        return []


def generate_benign_counterparts(
    text: str,
    client: "anthropic.Anthropic",
    n: int = 2,
    model: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.7,
) -> list[dict]:
    """Generate benign counterparts that look similar but are non-toxic.

    Returns:
        List of dicts with 'text', 'style', 'label' keys.
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=temperature,
            system=BENIGN_COUNTERPART_SYSTEM,
            messages=[{"role": "user", "content": BENIGN_COUNTERPART_USER.format(text=text, n=n)}],
        )

        content = response.content[0].text.strip()
        counterparts = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip numbering
            variant_text = line
            for prefix_len in range(1, 4):
                if len(variant_text) > prefix_len and variant_text[:prefix_len].rstrip(".)-:").isdigit():
                    variant_text = variant_text[prefix_len:].lstrip(".)-: ")
                    break

            if variant_text and variant_text != text:
                counterparts.append({
                    "text": variant_text,
                    "style": "benign_counterpart",
                    "label": 0,  # clean
                })

        return counterparts[:n]

    except Exception as e:
        logger.warning(f"API error generating counterparts: {e}")
        return []


def generate_all_variants(
    fn_df: pd.DataFrame,
    api_key: str,
    variants_per_sample: int = 5,
    benign_per_sample: int = 2,
    model: str = "claude-haiku-4-5-20251001",
    delay: float = 0.5,
) -> pd.DataFrame:
    """Generate variants for all FN samples.

    Args:
        fn_df: DataFrame of FN samples.
        api_key: Anthropic API key.
        variants_per_sample: Number of toxic variants per FN.
        benign_per_sample: Number of benign counterparts per FN.
        model: Claude model to use.
        delay: Delay between API calls (rate limiting).

    Returns:
        DataFrame with all generated samples.
    """
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package required. Install: pip install 'anthropic>=0.25.0'")

    client = anthropic.Anthropic(api_key=api_key)
    all_rows = []
    total_api_calls = 0

    for _, row in tqdm(fn_df.iterrows(), total=len(fn_df), desc="Generating variants"):
        text = row["text"]

        # Generate implicit toxic variants
        variants = generate_variants_llm(text, client, n=variants_per_sample, model=model)
        total_api_calls += 1

        for v in variants:
            all_rows.append({
                "text": v["text"],
                "label": v["label"],
                "style": v["style"],
                "source": "llm_implicit_variant",
                "original_text": text,
                "original_index": row.get("index", -1),
            })

        time.sleep(delay)

        # Generate benign counterparts
        if benign_per_sample > 0:
            counterparts = generate_benign_counterparts(text, client, n=benign_per_sample, model=model)
            total_api_calls += 1

            for c in counterparts:
                all_rows.append({
                    "text": c["text"],
                    "label": c["label"],
                    "style": c["style"],
                    "source": "llm_benign_counterpart",
                    "original_text": text,
                    "original_index": row.get("index", -1),
                })

            time.sleep(delay)

    logger.info(f"Total API calls: {total_api_calls}")
    logger.info(f"Generated {len(all_rows)} samples ({sum(1 for r in all_rows if r['label']==1)} toxic, "
                f"{sum(1 for r in all_rows if r['label']==0)} clean)")

    return pd.DataFrame(all_rows)


def validate_generated_samples(
    generated_df: pd.DataFrame,
    model_path: str,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    """Validate generated samples with the production model.

    Adds model prediction columns so we can check quality.

    Returns:
        DataFrame with added prediction columns.
    """
    if generated_df.empty:
        return generated_df

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    texts = generated_df["text"].tolist()
    dataset = SimpleDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating generated samples"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    del model
    torch.cuda.empty_cache()

    all_probs = np.vstack(all_probs)
    generated_df = generated_df.copy()
    generated_df["model_pred"] = all_probs.argmax(axis=1)
    generated_df["model_toxic_prob"] = all_probs[:, 1]
    generated_df["model_confidence"] = all_probs.max(axis=1)

    # Quality check
    toxic_variants = generated_df[generated_df["label"] == 1]
    benign_variants = generated_df[generated_df["label"] == 0]

    if len(toxic_variants) > 0:
        toxic_evasion = (toxic_variants["model_pred"] == 0).mean()
        logger.info(f"Toxic variants evasion rate: {toxic_evasion:.1%} "
                    f"({(toxic_variants['model_pred'] == 0).sum()}/{len(toxic_variants)} evade detection)")

    if len(benign_variants) > 0:
        benign_fp = (benign_variants["model_pred"] == 1).mean()
        logger.info(f"Benign counterparts FP rate: {benign_fp:.1%} "
                    f"({(benign_variants['model_pred'] == 1).sum()}/{len(benign_variants)} wrongly flagged)")

    return generated_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard negative samples using LLM"
    )
    parser.add_argument("--production-model", type=str, default="models/production",
                        help="Path to production model for FN detection")
    parser.add_argument("--fn-file", type=str, default=None,
                        help="CSV file with pre-identified FN samples (text, label columns)")
    parser.add_argument("--variants-per-sample", type=int, default=5,
                        help="Number of toxic variants per FN sample")
    parser.add_argument("--benign-per-sample", type=int, default=2,
                        help="Number of benign counterparts per FN sample")
    parser.add_argument("--output", type=str, default="data/korean/hard_negatives_llm.csv",
                        help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-model", type=str, default="claude-haiku-4-5-20251001",
                        help="Claude model for generation")
    parser.add_argument("--api-delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset-version", type=str, default="korean_standard_v1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only find FN samples, don't generate variants")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate generated samples with production model")
    parser.add_argument("--max-fn", type=int, default=None,
                        help="Max number of FN samples to process")
    args = parser.parse_args()

    # Resolve paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" / "korean"

    prod_path = Path(args.production_model)
    if not prod_path.exists():
        prod_path = Path(__file__).parent.parent / args.production_model
    if not prod_path.exists():
        logger.error(f"Production model not found: {args.production_model}")
        sys.exit(1)

    # Get FN samples
    if args.fn_file:
        logger.info(f"Loading FN samples from {args.fn_file}")
        fn_df = pd.read_csv(args.fn_file)
        if "text" not in fn_df.columns:
            logger.error("FN file must have 'text' column")
            sys.exit(1)
        if "label" not in fn_df.columns:
            fn_df["label"] = 1  # Assume all are toxic
    else:
        paths = get_data_paths(data_dir, args.dataset_version)
        test_path = paths["test"]
        if not test_path.exists():
            logger.error(f"Test data not found: {test_path}")
            sys.exit(1)

        logger.info(f"Loading test data from {test_path}")
        df = pd.read_csv(test_path)
        fn_df = find_fn_samples(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            model_path=str(prod_path),
            batch_size=args.batch_size,
        )

    if fn_df.empty:
        logger.info("No FN samples found. Model is performing well!")
        return

    if args.max_fn:
        fn_df = fn_df.head(args.max_fn)
        logger.info(f"Processing top {args.max_fn} FN samples")

    logger.info(f"FN samples to process: {len(fn_df)}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - FN SAMPLES FOUND")
        print("=" * 60)
        for _, row in fn_df.head(20).iterrows():
            text = str(row["text"])[:80]
            conf = row.get("confidence", 0)
            print(f"  [{conf:.3f}] {text}{'...' if len(str(row['text'])) > 80 else ''}")
        print(f"\nTotal: {len(fn_df)} FN samples")
        print(f"Expected output: ~{len(fn_df) * (args.variants_per_sample + args.benign_per_sample)} samples")

        # Save FN list
        fn_output = data_dir / "fn_samples.csv"
        fn_df.to_csv(fn_output, index=False, encoding="utf-8-sig")
        logger.info(f"FN samples saved to {fn_output}")
        return

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable required for generation")
        sys.exit(1)

    # Generate variants
    generated_df = generate_all_variants(
        fn_df=fn_df,
        api_key=api_key,
        variants_per_sample=args.variants_per_sample,
        benign_per_sample=args.benign_per_sample,
        model=args.api_model,
        delay=args.api_delay,
    )

    if generated_df.empty:
        logger.warning("No samples generated!")
        return

    # Validate with production model
    if args.validate:
        generated_df = validate_generated_samples(
            generated_df, str(prod_path), args.batch_size,
        )

    # Save output
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Generated samples saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"FN samples processed: {len(fn_df)}")
    print(f"Total generated: {len(generated_df)}")
    print(f"  Toxic variants: {(generated_df['label'] == 1).sum()}")
    print(f"  Benign counterparts: {(generated_df['label'] == 0).sum()}")
    if "style" in generated_df.columns:
        print("\nBy style:")
        for style, count in generated_df["style"].value_counts().items():
            print(f"  {style}: {count}")
    if "model_pred" in generated_df.columns:
        toxic_gen = generated_df[generated_df["label"] == 1]
        print(f"\nToxic variant evasion: {(toxic_gen['model_pred'] == 0).sum()}/{len(toxic_gen)}"
              f" ({(toxic_gen['model_pred'] == 0).mean():.1%})")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
