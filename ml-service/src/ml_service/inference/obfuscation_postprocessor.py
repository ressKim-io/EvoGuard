"""Obfuscation-aware post-processor for reducing false positives.

Detects obfuscated Korean text and applies confidence-based post-processing
to reduce FP caused by obfuscation artifacts.

Pipeline:
    1. Calculate obfuscation density of input text
    2. If borderline confidence (0.5-0.8) + high density → deobfuscate and re-classify
    3. Return adjusted prediction

Usage:
    from ml_service.inference.obfuscation_postprocessor import ObfuscationPostProcessor

    postprocessor = ObfuscationPostProcessor()
    adjusted = postprocessor.post_process(text, label=1, confidence=0.65, toxic_prob=0.65)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# Hangul jamo ranges
_JAMO_INITIAL = range(0x1100, 0x1113)   # ㄱ-ㅎ (Choseong)
_JAMO_MEDIAL = range(0x1161, 0x1176)    # ㅏ-ㅣ (Jungseong)
_JAMO_FINAL = range(0x11A8, 0x11C3)     # ㄱ-ㅎ (Jongseong)
_COMPAT_JAMO = range(0x3131, 0x3164)    # Compatibility jamo (ㄱ-ㅎ, ㅏ-ㅣ)
_HANGUL_SYLLABLE = range(0xAC00, 0xD7A4) # 가-힣


class ObfuscationPostProcessor:
    """Post-processor that handles obfuscated Korean text to reduce FP.

    Args:
        rules_dir: Path to KOTOX rules directory.
        confidence_low: Lower bound of borderline confidence zone.
        confidence_high: Upper bound of borderline confidence zone.
        density_threshold: Obfuscation density threshold to trigger processing.
        high_confidence_override: If confidence > this, trust the model regardless.
    """

    def __init__(
        self,
        rules_dir: str | Path | None = None,
        confidence_low: float = 0.5,
        confidence_high: float = 0.8,
        density_threshold: float = 0.15,
        high_confidence_override: float = 0.9,
    ) -> None:
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.density_threshold = density_threshold
        self.high_confidence_override = high_confidence_override

        # Load KOTOX deobfuscation rules
        self.consonant_map: dict[str, str] = {}
        self.vowel_map: dict[str, str] = {}
        self.trans_map: dict = {}

        if rules_dir is None:
            # Default path
            rules_dir = Path(__file__).parent.parent.parent.parent / "data" / "korean" / "KOTOX" / "rules"

        self.rules_dir = Path(rules_dir)
        if self.rules_dir.exists():
            self._load_rules()
        else:
            logger.warning(f"KOTOX rules not found at {self.rules_dir}. Using basic normalization only.")

    def _load_rules(self) -> None:
        """Load KOTOX deobfuscation rules."""
        iconic_path = self.rules_dir / "iconic_dictionary.json"
        if iconic_path.exists():
            with open(iconic_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for orig, variants in data.get("consonant_dict", {}).items():
                    for v in variants:
                        self.consonant_map[v] = orig
                for orig, variants in data.get("vowel_dict", {}).items():
                    for v in variants:
                        self.vowel_map[v] = orig

        trans_path = self.rules_dir / "transliterational_dictionary.json"
        if trans_path.exists():
            with open(trans_path, "r", encoding="utf-8") as f:
                self.trans_map = json.load(f)

        logger.debug(f"Loaded {len(self.consonant_map)} consonant + {len(self.vowel_map)} vowel mappings")

    def calculate_obfuscation_density(self, text: str) -> float:
        """Calculate what fraction of the text is obfuscated.

        Obfuscation indicators:
        - Zero-width characters
        - Standalone jamo (not part of syllables)
        - Special symbols between Korean characters
        - Non-standard Unicode in Korean context
        - Mixed script (Latin letters between Korean)

        Args:
            text: Input text.

        Returns:
            Density score between 0.0 and 1.0.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        total_chars = len(text)
        obfuscation_count = 0

        for i, ch in enumerate(text):
            code = ord(ch)

            # Zero-width characters
            if code in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x200E, 0x200F):
                obfuscation_count += 1
                continue

            # Standalone compatibility jamo (ㄱ, ㅏ, etc.) surrounded by non-jamo
            if code in _COMPAT_JAMO:
                # Check context: if isolated jamo between non-jamo chars, it's likely obfuscation
                prev_is_syllable = i > 0 and ord(text[i - 1]) in _HANGUL_SYLLABLE
                next_is_syllable = i < len(text) - 1 and ord(text[i + 1]) in _HANGUL_SYLLABLE
                if not prev_is_syllable and not next_is_syllable:
                    obfuscation_count += 1
                    continue

            # Special symbols between Korean characters (decorative obfuscation)
            if ch in "°♡♥★☆《》「」『』【】■□●○◆◇▶◀△▽←→↑↓":
                obfuscation_count += 1
                continue

            # Unusual punctuation used for obfuscation
            if ch in "._~*^" and i > 0 and i < len(text) - 1:
                prev_code = ord(text[i - 1])
                next_code = ord(text[i + 1])
                if (prev_code in _HANGUL_SYLLABLE or prev_code in _COMPAT_JAMO) and \
                   (next_code in _HANGUL_SYLLABLE or next_code in _COMPAT_JAMO):
                    obfuscation_count += 1
                    continue

            # Latin/CJK characters mixed into Korean context
            if ch.isascii() and ch.isalpha():
                # Check if surrounded by Korean
                prev_kr = i > 0 and (ord(text[i - 1]) in _HANGUL_SYLLABLE or ord(text[i - 1]) in _COMPAT_JAMO)
                next_kr = i < len(text) - 1 and (ord(text[i + 1]) in _HANGUL_SYLLABLE or ord(text[i + 1]) in _COMPAT_JAMO)
                if prev_kr or next_kr:
                    obfuscation_count += 1
                    continue

        return obfuscation_count / total_chars if total_chars > 0 else 0.0

    def normalize_text(self, text: str) -> str:
        """Normalize obfuscated Korean text.

        Steps:
        1. Unicode NFKC normalization
        2. Remove zero-width characters
        3. Apply KOTOX consonant/vowel reverse mappings
        4. Normalize whitespace
        5. Remove decorative symbols

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        result = text

        # NFKC normalization (fullwidth → halfwidth, compatibility chars)
        result = unicodedata.normalize("NFKC", result)

        # Remove zero-width characters
        result = re.sub(r"[\u200b\u200c\u200d\ufeff\u200e\u200f]", "", result)

        # Remove decorative symbols between Korean characters
        result = re.sub(r"(?<=[\uAC00-\uD7A3\u3131-\u3163])[°♡♥★☆《》「」『』【】■□●○◆◇▶◀△▽←→↑↓._~*^]+(?=[\uAC00-\uD7A3\u3131-\u3163])", "", result)

        # Apply KOTOX consonant mappings
        for obf, orig in self.consonant_map.items():
            result = result.replace(obf, orig)

        # Apply KOTOX vowel mappings
        for obf, orig in self.vowel_map.items():
            result = result.replace(obf, orig)

        # Normalize whitespace
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def should_apply(self, confidence: float, density: float) -> bool:
        """Determine if post-processing should be applied.

        Rules:
        - If confidence > high_confidence_override: skip (model is sure)
        - If confidence < confidence_low: skip (already classified as benign)
        - If borderline + high density: apply
        - If very high density (>2x threshold): always apply

        Args:
            confidence: Model confidence for the predicted label.
            density: Obfuscation density score.

        Returns:
            True if post-processing should be applied.
        """
        if confidence > self.high_confidence_override:
            return False
        if density >= self.density_threshold * 2:
            return True  # Very obfuscated, always process
        if self.confidence_low <= confidence <= self.confidence_high and density >= self.density_threshold:
            return True
        return False

    def post_process(
        self,
        text: str,
        label: int,
        confidence: float,
        toxic_prob: float,
        reclassify_fn=None,
    ) -> dict:
        """Post-process a prediction considering obfuscation.

        Args:
            text: Original input text.
            label: Predicted label (0=clean, 1=toxic).
            confidence: Model confidence.
            toxic_prob: Probability of toxic class.
            reclassify_fn: Optional function(text) -> {'label': int, 'toxic_prob': float, 'confidence': float}
                           to reclassify normalized text.

        Returns:
            Dict with adjusted prediction:
                - label: Adjusted label
                - confidence: Adjusted confidence
                - toxic_prob: Adjusted toxic probability
                - was_adjusted: Whether adjustment was made
                - obfuscation_density: Measured density
                - normalized_text: Text after normalization (if adjusted)
        """
        density = self.calculate_obfuscation_density(text)

        result = {
            "label": label,
            "confidence": confidence,
            "toxic_prob": toxic_prob,
            "was_adjusted": False,
            "obfuscation_density": density,
            "normalized_text": None,
        }

        # Only post-process toxic predictions (to reduce FP)
        if label != 1:
            return result

        if not self.should_apply(confidence, density):
            return result

        # Normalize the text
        normalized = self.normalize_text(text)

        # If text didn't change much after normalization, skip
        if normalized == text or len(normalized) < 2:
            return result

        result["normalized_text"] = normalized

        if reclassify_fn is not None:
            # Re-classify normalized text
            reclass = reclassify_fn(normalized)
            new_toxic_prob = reclass["toxic_prob"]
            new_confidence = reclass["confidence"]
            new_label = reclass["label"]

            # Conservative: take lower toxic probability (favoring non-toxic)
            if new_toxic_prob < toxic_prob:
                result["label"] = new_label
                result["confidence"] = new_confidence
                result["toxic_prob"] = new_toxic_prob
                result["was_adjusted"] = True
        else:
            # Without reclassification, apply confidence penalty
            # Obfuscated text with borderline confidence → reduce confidence
            penalty = density * 0.3  # Up to 30% penalty proportional to density
            adjusted_prob = toxic_prob * (1.0 - penalty)

            if adjusted_prob < 0.5 and toxic_prob >= 0.5:
                result["label"] = 0
                result["confidence"] = 1.0 - adjusted_prob
                result["toxic_prob"] = adjusted_prob
                result["was_adjusted"] = True
            elif adjusted_prob < toxic_prob:
                result["toxic_prob"] = adjusted_prob
                result["confidence"] = max(adjusted_prob, 1.0 - adjusted_prob)
                result["was_adjusted"] = True

        return result

    def post_process_batch(
        self,
        texts: list[str],
        labels: list[int],
        confidences: list[float],
        toxic_probs: list[float],
        reclassify_fn=None,
    ) -> list[dict]:
        """Post-process a batch of predictions.

        Args:
            texts: List of input texts.
            labels: List of predicted labels.
            confidences: List of model confidences.
            toxic_probs: List of toxic probabilities.
            reclassify_fn: Optional batch reclassification function.

        Returns:
            List of adjusted prediction dicts.
        """
        results = []
        n_adjusted = 0

        for text, label, conf, prob in zip(texts, labels, confidences, toxic_probs):
            result = self.post_process(text, label, conf, prob, reclassify_fn)
            results.append(result)
            if result["was_adjusted"]:
                n_adjusted += 1

        if n_adjusted > 0:
            logger.info(f"Post-processed {n_adjusted}/{len(texts)} predictions")

        return results
