"""Slang Evolver for Attacker Evolver.

슬랭 사전을 동적으로 확장합니다.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_service.attacker.slang_dictionary import (
    add_new_slang,
    get_all_slang,
    generate_variants,
    SUBSTITUTION_MAP,
    INSERTION_PATTERNS,
)


class SlangEvolver:
    """Dynamically expand slang dictionary.

    슬랭 사전을 동적으로 확장합니다.
    """

    def __init__(
        self,
        classifier: Any = None,
        storage_path: Path | None = None,
    ) -> None:
        """Initialize SlangEvolver.

        Args:
            classifier: 분류기 (테스트용)
            storage_path: 저장 경로
        """
        self.classifier = classifier
        self.storage_path = storage_path or Path("data/korean/evolved_slang.json")

        self._evolved_slang: list[dict] = []
        self._pending_slang: list[str] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load evolved slang state."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._evolved_slang = data.get("evolved", [])
                self._pending_slang = data.get("pending", [])
            except Exception:
                pass

    def _save_state(self) -> None:
        """Save evolved slang state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "evolved": self._evolved_slang,
            "pending": self._pending_slang,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_phonetic_variants(
        self,
        base_words: list[str] | None = None,
        max_variants: int = 20,
    ) -> list[str]:
        """Generate phonetic variants of base words.

        Args:
            base_words: 기본 단어 리스트 (없으면 슬랭 사전에서)
            max_variants: 최대 변형 수

        Returns:
            생성된 변형 리스트
        """
        if base_words is None:
            base_words = get_all_slang()[:20]

        variants = []

        for word in base_words:
            # 발음 변형
            for char, subs in SUBSTITUTION_MAP.items():
                if char in word:
                    for sub in subs:
                        variant = word.replace(char, sub)
                        if variant not in variants and variant != word:
                            variants.append(variant)

            # 구분자 삽입
            for pattern in INSERTION_PATTERNS[:5]:
                try:
                    variant = pattern(word)
                    if variant not in variants:
                        variants.append(variant)
                except Exception:
                    pass

        # 랜덤 샘플링
        if len(variants) > max_variants:
            variants = random.sample(variants, max_variants)

        return variants

    def discover_from_failures(
        self,
        successful_evasions: list[dict],
    ) -> list[str]:
        """Discover new slang from successful evasions.

        Args:
            successful_evasions: 성공한 우회 공격 리스트

        Returns:
            발견된 새 슬랭 리스트
        """
        existing = set(get_all_slang())
        existing.update(s["text"] for s in self._evolved_slang)

        discovered = []

        for evasion in successful_evasions:
            variant = evasion.get("variant_text", "")

            # 조건: 짧고, 새로운 것
            if 2 <= len(variant) <= 8 and variant not in existing:
                discovered.append(variant)
                existing.add(variant)

        return discovered

    def validate_and_add(
        self,
        candidates: list[str],
        category: str = "behavior",
    ) -> int:
        """Validate candidates and add to slang dictionary.

        Args:
            candidates: 후보 슬랭 리스트
            category: 카테고리

        Returns:
            추가된 슬랭 수
        """
        added = 0

        for candidate in candidates:
            # 분류기로 테스트 (있으면)
            if self.classifier:
                try:
                    pred = self.classifier.predict([candidate])[0]
                    # 정상으로 예측되면 유효한 우회 슬랭
                    if pred["label"] == 0:
                        if add_new_slang(candidate, category):
                            self._evolved_slang.append({
                                "text": candidate,
                                "category": category,
                                "added_at": datetime.now(UTC).isoformat(),
                            })
                            added += 1
                except Exception:
                    pass
            else:
                # 분류기 없으면 바로 추가
                if add_new_slang(candidate, category):
                    self._evolved_slang.append({
                        "text": candidate,
                        "category": category,
                        "added_at": datetime.now(UTC).isoformat(),
                    })
                    added += 1

        if added > 0:
            self._save_state()

        return added

    def get_evolved_count(self) -> int:
        """Get number of evolved slang."""
        return len(self._evolved_slang)


__all__ = ['SlangEvolver']
