"""Evolving LLM-based attacker that learns new evasion strategies.

This module implements a GAN-style adversarial system where:
1. The attacker generates evasion variants
2. The defender (classifier) tries to detect them
3. When attacks are blocked, the attacker evolves new strategies using LLM
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ClassifierProtocol(Protocol):
    """Protocol for classifier models."""

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict labels for texts."""
        ...


@dataclass
class EvolvedStrategy:
    """A newly evolved attack strategy."""

    name: str
    description: str
    transform_code: str  # Python code to transform text
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tested_count: int = 0
    evasion_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "transform_code": self.transform_code,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "tested_count": self.tested_count,
            "evasion_count": self.evasion_count,
        }


@dataclass
class EvolutionResult:
    """Result of attack evolution."""

    new_strategies: list[EvolvedStrategy]
    tested_count: int
    successful_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EvolvingAttacker:
    """LLM-based attacker that evolves new strategies when blocked.

    Uses an LLM to analyze blocked attacks and generate new evasion
    techniques that the defender hasn't seen before.

    Example:
        >>> attacker = EvolvingAttacker(classifier, llm_client)
        >>> result = await attacker.evolve(blocked_strategies, sample_texts)
        >>> print(f"Generated {len(result.new_strategies)} new strategies")
    """

    # Prompt template for generating new attack strategies (Korean-focused)
    EVOLUTION_PROMPT = '''You are a security researcher testing Korean text classification robustness.

The following Korean evasion strategies have been BLOCKED by the classifier:
{blocked_strategies}

Your task: Generate NEW Korean-specific evasion techniques that are DIFFERENT from the blocked ones.

IMPORTANT: Focus on KOREAN language transformations, not English!

Requirements:
1. The transformed text must preserve the original Korean meaning
2. Use creative Korean Unicode, Hangul Jamo, or linguistic tricks
3. Provide Python code that transforms Korean text

Respond with exactly 3 new strategies in this JSON format:
```json
[
  {{
    "name": "strategy_name",
    "description": "Brief description of the technique",
    "transform_code": "def transform(text: str) -> str:\\n    # transformation logic\\n    return transformed_text"
  }}
]
```

Korean-specific techniques to inspire you (but create NEW ones):
- 초성 추출: 시발 → ㅅㅂ (extract initial consonants)
- 자모 분리: 시발 → ㅅㅣㅂㅏㄹ (decompose into Jamo)
- 숫자 치환: 시발 → 시8 (number substitution for pronunciation)
- 영타 변환: 시발 → tlqkf (Korean to English keyboard)
- 유사 발음: 병신 → 뱅신, 벙신 (similar pronunciation)
- 신조어: 시발 → ㅅㅂ, 시부랄 (internet slang)
- 호환용 자모: ㅅㅂ → ᄉᄇ (compatibility Jamo U+1100-U+11FF)
- 방언: 병신 → 뱅신 (dialect variation)
- 음절 교환: 병신 → 신병 (syllable swap)
- 제로폭 문자: 시​발 (insert zero-width characters)
- 부분 마스킹: 시X, 병* (partial masking)
- 의도적 오타: 시발 → 씨발 (intentional typos)

Generate creative NEW Korean-specific techniques:'''

    def __init__(
        self,
        classifier: ClassifierProtocol,
        llm_endpoint: str = "http://localhost:11434",
        llm_model: str = "mistral",
        storage_path: Path | None = None,
    ) -> None:
        """Initialize evolving attacker.

        Args:
            classifier: Target classifier to attack.
            llm_endpoint: Ollama API endpoint.
            llm_model: LLM model to use.
            storage_path: Path to store evolved strategies.
        """
        self.classifier = classifier
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.storage_path = storage_path or Path("data/evolved_strategies.json")
        self.evolved_strategies: list[EvolvedStrategy] = []

        # Load existing strategies
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Load previously evolved strategies."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                for item in data:
                    self.evolved_strategies.append(EvolvedStrategy(
                        name=item["name"],
                        description=item["description"],
                        transform_code=item["transform_code"],
                        success_rate=item.get("success_rate", 0.0),
                        tested_count=item.get("tested_count", 0),
                        evasion_count=item.get("evasion_count", 0),
                    ))
                logger.info(f"Loaded {len(self.evolved_strategies)} evolved strategies")
            except Exception as e:
                logger.warning(f"Failed to load strategies: {e}")

    def _save_strategies(self) -> None:
        """Save evolved strategies to file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump([s.to_dict() for s in self.evolved_strategies], f, indent=2)

    async def evolve(
        self,
        blocked_strategies: list[str],
        sample_texts: list[str],
        min_success_rate: float = 0.3,
    ) -> EvolutionResult:
        """Evolve new attack strategies based on blocked ones.

        Args:
            blocked_strategies: List of strategy names that were blocked.
            sample_texts: Sample toxic texts to test new strategies on.
            min_success_rate: Minimum success rate to keep a strategy.

        Returns:
            EvolutionResult with new successful strategies.
        """
        logger.info(f"Evolving new strategies. Blocked: {blocked_strategies}")

        # Generate new strategies using LLM
        new_strategies = await self._generate_strategies(blocked_strategies)

        if not new_strategies:
            logger.warning("LLM failed to generate new strategies")
            return EvolutionResult(
                new_strategies=[],
                tested_count=0,
                successful_count=0,
            )

        # Test each strategy
        successful_strategies = []
        for strategy in new_strategies:
            success_rate = self._test_strategy(strategy, sample_texts)
            strategy.success_rate = success_rate

            if success_rate >= min_success_rate:
                successful_strategies.append(strategy)
                self.evolved_strategies.append(strategy)
                logger.info(f"New strategy '{strategy.name}' succeeded with {success_rate:.1%}")
            else:
                logger.info(f"Strategy '{strategy.name}' failed with {success_rate:.1%}")

        # Save successful strategies
        if successful_strategies:
            self._save_strategies()

        return EvolutionResult(
            new_strategies=successful_strategies,
            tested_count=len(new_strategies),
            successful_count=len(successful_strategies),
        )

    async def _generate_strategies(
        self,
        blocked_strategies: list[str],
    ) -> list[EvolvedStrategy]:
        """Use LLM to generate new attack strategies."""
        import httpx

        blocked_str = "\n".join(f"- {s}" for s in blocked_strategies)
        prompt = self.EVOLUTION_PROMPT.format(blocked_strategies=blocked_str)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.llm_endpoint}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.8,
                            "num_predict": 2048,
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()

            # Parse LLM response
            return self._parse_llm_response(result.get("response", ""))

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []

    def _parse_llm_response(self, response: str) -> list[EvolvedStrategy]:
        """Parse LLM response to extract strategies."""
        strategies = []

        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return []

        try:
            data = json.loads(json_match.group())
            for item in data:
                if all(k in item for k in ["name", "description", "transform_code"]):
                    strategies.append(EvolvedStrategy(
                        name=item["name"],
                        description=item["description"],
                        transform_code=item["transform_code"],
                    ))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")

        return strategies

    def _test_strategy(
        self,
        strategy: EvolvedStrategy,
        sample_texts: list[str],
    ) -> float:
        """Test a strategy against the classifier.

        Returns:
            Evasion success rate (0.0 - 1.0).
        """
        # Compile the transform function
        transform_fn = self._compile_transform(strategy.transform_code)
        if transform_fn is None:
            return 0.0

        # Transform texts
        transformed = []
        for text in sample_texts:
            try:
                result = transform_fn(text)
                transformed.append(result)
            except Exception:
                transformed.append(text)  # Use original on error

        # Test against classifier
        predictions = self.classifier.predict(transformed)

        # Count evasions (toxic classified as non-toxic)
        evasions = sum(1 for p in predictions if p["label"] == 0)

        strategy.tested_count = len(sample_texts)
        strategy.evasion_count = evasions

        return evasions / len(sample_texts) if sample_texts else 0.0

    def _compile_transform(self, code: str) -> callable | None:
        """Safely compile a transform function from code string."""
        # Clean the code
        code = code.strip()

        # Remove markdown code block markers if present
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            # Create a restricted namespace
            namespace: dict[str, Any] = {}

            # Execute the code to define the function
            exec(code, namespace)

            # Look for the transform function
            if "transform" in namespace:
                return namespace["transform"]

            # Try to find any callable
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    return obj

            return None

        except Exception as e:
            logger.warning(f"Failed to compile transform: {e}")
            return None

    def apply_evolved_strategies(
        self,
        text: str,
        num_variants: int = 3,
    ) -> list[dict[str, Any]]:
        """Apply evolved strategies to generate variants.

        Args:
            text: Text to transform.
            num_variants: Number of variants to generate.

        Returns:
            List of dicts with transformed text and strategy info.
        """
        results = []

        # Sort by success rate, use best strategies first
        sorted_strategies = sorted(
            self.evolved_strategies,
            key=lambda s: s.success_rate,
            reverse=True,
        )

        for strategy in sorted_strategies[:num_variants]:
            transform_fn = self._compile_transform(strategy.transform_code)
            if transform_fn is None:
                continue

            try:
                transformed = transform_fn(text)
                results.append({
                    "original": text,
                    "evasion": transformed,
                    "strategy": strategy.name,
                    "confidence": strategy.success_rate,
                })
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about evolved strategies."""
        if not self.evolved_strategies:
            return {"total_strategies": 0}

        success_rates = [s.success_rate for s in self.evolved_strategies]

        return {
            "total_strategies": len(self.evolved_strategies),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "best_strategy": max(self.evolved_strategies, key=lambda s: s.success_rate).name,
            "total_tested": sum(s.tested_count for s in self.evolved_strategies),
            "total_evasions": sum(s.evasion_count for s in self.evolved_strategies),
        }


class AdversarialCoevolution:
    """Manages the co-evolution of attacker and defender.

    Implements a GAN-style training loop where:
    1. Attacker generates evasions
    2. Defender tries to classify
    3. If defender wins (low evasion rate), attacker evolves
    4. If attacker wins (high evasion rate), defender retrains
    5. Repeat

    Example:
        >>> coevolution = AdversarialCoevolution(classifier, evolving_attacker)
        >>> result = await coevolution.run_cycle(sample_texts)
    """

    def __init__(
        self,
        classifier: ClassifierProtocol,
        evolving_attacker: EvolvingAttacker,
        attack_evolution_threshold: float = 0.2,  # Evolve attacker if evasion < 20%
        defender_retrain_threshold: float = 0.5,  # Retrain defender if evasion > 50%
    ) -> None:
        self.classifier = classifier
        self.attacker = evolving_attacker
        self.attack_evolution_threshold = attack_evolution_threshold
        self.defender_retrain_threshold = defender_retrain_threshold
        self.history: list[dict[str, Any]] = []

    async def run_cycle(
        self,
        sample_texts: list[str],
        current_evasion_rate: float,
        blocked_strategies: list[str],
    ) -> dict[str, Any]:
        """Run a co-evolution cycle.

        Args:
            sample_texts: Toxic texts to use for testing.
            current_evasion_rate: Current evasion rate from pipeline.
            blocked_strategies: Strategies that are currently blocked.

        Returns:
            Cycle result with actions taken.
        """
        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "current_evasion_rate": current_evasion_rate,
            "action": None,
            "details": {},
        }

        # Decide action based on evasion rate
        if current_evasion_rate < self.attack_evolution_threshold:
            # Defender is winning - evolve attacker
            logger.info(f"Evasion rate {current_evasion_rate:.1%} < {self.attack_evolution_threshold:.1%}, evolving attacker")

            evolution_result = await self.attacker.evolve(
                blocked_strategies=blocked_strategies,
                sample_texts=sample_texts,
            )

            result["action"] = "evolve_attacker"
            result["details"] = {
                "new_strategies": len(evolution_result.new_strategies),
                "successful": evolution_result.successful_count,
                "strategy_names": [s.name for s in evolution_result.new_strategies],
            }

        elif current_evasion_rate > self.defender_retrain_threshold:
            # Attacker is winning - signal to retrain defender
            logger.info(f"Evasion rate {current_evasion_rate:.1%} > {self.defender_retrain_threshold:.1%}, defender needs retraining")

            result["action"] = "retrain_defender"
            result["details"] = {
                "evasion_rate": current_evasion_rate,
                "recommendation": "Collect failed samples and retrain",
            }

        else:
            # Balanced state
            result["action"] = "none"
            result["details"] = {
                "status": "balanced",
                "evasion_rate": current_evasion_rate,
            }

        self.history.append(result)
        return result


# Built-in evolved strategies (Korean-specific creative techniques)
BUILTIN_EVOLVED_STRATEGIES = [
    EvolvedStrategy(
        name="korean_zwc_injection",
        description="한글 사이에 제로폭 문자 삽입",
        transform_code='''
def transform(text: str) -> str:
    import random
    zwc = ["\\u200b", "\\u200c", "\\u200d", "\\ufeff"]
    result = []
    for i, char in enumerate(text):
        result.append(char)
        if i < len(text) - 1 and 0xAC00 <= ord(char) <= 0xD7A3:
            if random.random() < 0.4:
                result.append(random.choice(zwc))
    return "".join(result)
''',
        success_rate=0.6,
    ),
    EvolvedStrategy(
        name="korean_compat_jamo",
        description="호환용 자모로 변환 (U+3131-U+318E → U+1100-U+11FF)",
        transform_code='''
def transform(text: str) -> str:
    compat_map = {
        "ㄱ": "ᄀ", "ㄴ": "ᄂ", "ㄷ": "ᄃ", "ㄹ": "ᄅ", "ㅁ": "ᄆ",
        "ㅂ": "ᄇ", "ㅅ": "ᄉ", "ㅇ": "ᄋ", "ㅈ": "ᄌ", "ㅊ": "ᄎ",
        "ㅋ": "ᄏ", "ㅌ": "ᄐ", "ㅍ": "ᄑ", "ㅎ": "ᄒ",
        "ㅏ": "ᅡ", "ㅓ": "ᅥ", "ㅗ": "ᅩ", "ㅜ": "ᅮ", "ㅡ": "ᅳ", "ㅣ": "ᅵ",
    }
    return "".join(compat_map.get(c, c) for c in text)
''',
        success_rate=0.7,
    ),
    EvolvedStrategy(
        name="korean_fullwidth_mix",
        description="전각/반각 문자 혼용",
        transform_code='''
def transform(text: str) -> str:
    import random
    result = []
    for char in text:
        if char.isascii() and char.isalpha() and random.random() < 0.5:
            result.append(chr(ord(char) + 0xFEE0))
        else:
            result.append(char)
    return "".join(result)
''',
        success_rate=0.5,
    ),
    EvolvedStrategy(
        name="korean_slang_abbrev",
        description="신조어/축약어 변환",
        transform_code='''
def transform(text: str) -> str:
    import random
    slang = {
        "시발": ["ㅅㅂ", "시부랄", "시팔"],
        "병신": ["ㅂㅅ", "ㅄ", "벵신"],
        "새끼": ["ㅅㄲ", "색히"],
        "미친": ["ㅁㅊ", "믻친"],
        "지랄": ["ㅈㄹ", "지랄"],
    }
    result = text
    for orig, slangs in slang.items():
        if orig in result:
            result = result.replace(orig, random.choice(slangs), 1)
    return result
''',
        success_rate=0.65,
    ),
    EvolvedStrategy(
        name="korean_phonetic_variant",
        description="유사 발음 변형",
        transform_code='''
def transform(text: str) -> str:
    import random
    variants = {
        "시": ["씨", "쉬", "싀"],
        "발": ["빨", "8", "ㅂㅏㄹ"],
        "병": ["뱅", "벙", "빙"],
        "신": ["싄", "$ㅣㄴ"],
        "새": ["쌔", "섀"],
        "끼": ["키", "띠"],
    }
    result = list(text)
    for i, char in enumerate(result):
        if char in variants and random.random() < 0.4:
            result[i] = random.choice(variants[char])
    return "".join(result)
''',
        success_rate=0.55,
    ),
]


def create_evolving_attacker(
    classifier: ClassifierProtocol,
    llm_endpoint: str = "http://localhost:11434",
    include_builtin: bool = True,
) -> EvolvingAttacker:
    """Create an evolving attacker with optional built-in strategies.

    Args:
        classifier: Target classifier.
        llm_endpoint: Ollama endpoint.
        include_builtin: Whether to include built-in evolved strategies.

    Returns:
        Configured EvolvingAttacker.
    """
    attacker = EvolvingAttacker(
        classifier=classifier,
        llm_endpoint=llm_endpoint,
    )

    if include_builtin:
        for strategy in BUILTIN_EVOLVED_STRATEGIES:
            attacker.evolved_strategies.append(strategy)
        attacker._save_strategies()

    return attacker
