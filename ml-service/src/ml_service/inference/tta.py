"""Selective Test-Time Augmentation (STTA) for Korean Text Classification.

References:
- "Improved Text Classification via Test-Time Augmentation" (arxiv 2206.13607)
- "STTA: Enhanced text classification via selective test-time augmentation" (PMC 2024)

Korean-specific augmentations: spacing variants, Unicode normalization, jamo decomposition.
"""

import re
import unicodedata
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def _normalize_unicode(text: str, form: str = "NFC") -> str:
    """Unicode normalization (NFC, NFD, NFKC, NFKD)."""
    return unicodedata.normalize(form, text)


def _remove_extra_spaces(text: str) -> str:
    """Remove extra spaces between Korean characters."""
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove spaces between Korean characters
    text = re.sub(r"([\uAC00-\uD7AF])\s+([\uAC00-\uD7AF])", r"\1\2", text)
    return text


def _add_spaces_between_words(text: str) -> str:
    """Add spaces between Korean word boundaries (rough heuristic)."""
    # Add space before particles that commonly start new words
    particles = ["은", "는", "이", "가", "을", "를", "에", "의", "로", "와", "과", "도"]
    for p in particles:
        # Only add space if preceded by Korean char and no existing space
        text = re.sub(rf"([\uAC00-\uD7AF])({p})([\uAC00-\uD7AF])", rf"\1{p} \3", text)
    return text


def _strip_special_chars(text: str) -> str:
    """Remove common obfuscation characters."""
    # Remove zero-width chars, soft hyphens, invisible chars
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]", "", text)
    # Normalize fullwidth to halfwidth
    text = unicodedata.normalize("NFKC", text)
    return text


def _swap_similar_chars(text: str) -> str:
    """Normalize visually similar characters to standard Korean."""
    # Common homoglyph mappings
    replacements = {
        "ㅇㅏ": "아", "ㅅㅣ": "시", "ㄱㅏ": "가",
        "０": "0", "１": "1", "２": "2", "３": "3",
        "ａ": "a", "ｂ": "b", "ｃ": "c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# All available augmentation functions
AUGMENTATIONS = [
    ("nfc", lambda t: _normalize_unicode(t, "NFC")),
    ("nfkc", lambda t: _normalize_unicode(t, "NFKC")),
    ("strip_special", _strip_special_chars),
    ("remove_spaces", _remove_extra_spaces),
    ("add_spaces", _add_spaces_between_words),
    ("homoglyph", _swap_similar_chars),
]


def generate_augmented_texts(
    text: str,
    augmentations: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """Generate augmented versions of input text.

    Returns list of (aug_name, augmented_text) tuples.
    Always includes the original text.
    """
    results = [("original", text)]

    if augmentations is None:
        augmentations = [name for name, _ in AUGMENTATIONS]

    for name, fn in AUGMENTATIONS:
        if name in augmentations:
            augmented = fn(text)
            # Only include if different from original
            if augmented != text:
                results.append((name, augmented))

    return results


def selective_tta_predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: str = "cuda",
    augmentations: Optional[list[str]] = None,
    confidence_threshold: float = 0.8,
    max_length: int = 128,
    batch_size: int = 64,
) -> tuple[list[int], list[float]]:
    """Selective Test-Time Augmentation prediction.

    For each input text:
    1. Generate augmented variants
    2. Predict on all variants
    3. If original prediction is confident (>threshold), use it directly
    4. Otherwise, aggregate predictions from all variants (selective)

    Args:
        model: Trained classification model
        tokenizer: Associated tokenizer
        texts: Input texts to classify
        device: Compute device
        augmentations: List of augmentation names to apply
        confidence_threshold: Skip TTA if original confidence exceeds this
        max_length: Max token length
        batch_size: Batch size for inference

    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_preds = []
        batch_probs = []

        for text in batch_texts:
            # Get original prediction first
            inputs = tokenizer(
                [text], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                orig_probs = torch.softmax(outputs.logits, dim=-1)
                orig_conf = orig_probs.max().item()
                orig_pred = orig_probs.argmax(dim=-1).item()
                orig_toxic_prob = orig_probs[0, 1].item()

            # If confident enough, skip TTA
            if orig_conf >= confidence_threshold:
                batch_preds.append(orig_pred)
                batch_probs.append(orig_toxic_prob)
                continue

            # Generate augmented versions
            aug_texts = generate_augmented_texts(text, augmentations)

            # Predict on all variants
            variant_texts = [t for _, t in aug_texts]
            var_inputs = tokenizer(
                variant_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                var_outputs = model(**var_inputs)
                var_probs = torch.softmax(var_outputs.logits, dim=-1)

            # Selective aggregation: weight by distance from 0.5
            # More confident predictions get higher weight
            toxic_probs = var_probs[:, 1]
            weights = (toxic_probs - 0.5).abs()  # Higher weight for confident preds
            weights = weights / weights.sum()  # Normalize

            avg_toxic_prob = (toxic_probs * weights).sum().item()
            final_pred = 1 if avg_toxic_prob >= 0.5 else 0

            batch_preds.append(final_pred)
            batch_probs.append(avg_toxic_prob)

        all_preds.extend(batch_preds)
        all_probs.extend(batch_probs)

    return all_preds, all_probs


def evaluate_with_tta(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_path: str,
    device: str = "cuda",
    confidence_threshold: float = 0.8,
) -> dict:
    """Evaluate model with STTA on test set."""
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score

    test_df = pd.read_csv(test_path)
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    preds, probs = selective_tta_predict(
        model, tokenizer, texts, device,
        confidence_threshold=confidence_threshold,
    )

    f1 = f1_score(labels, preds, average="weighted")
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "f1_weighted": f1,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "confidence_threshold": confidence_threshold,
    }
