#!/usr/bin/env python3
"""Run adversarial co-evolution pipeline.

This script runs the GAN-style co-evolution where:
1. Attacker attacks the defender
2. If attacker wins (high evasion) → Defender retrains
3. If defender wins (low evasion) → Attacker evolves new strategies
4. Repeat until equilibrium

Usage:
    python scripts/run_coevolution.py --cycles 5
    python scripts/run_coevolution.py --cycles 10 --use-llm
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from ml_service.pipeline.classifier_adapter import TrainedClassifierAdapter
from ml_service.pipeline.attack_runner import AttackRunner
from ml_service.pipeline.config import AttackConfig
from ml_service.pipeline.evolving_attacker import (
    EvolvingAttacker,
    AdversarialCoevolution,
    create_evolving_attacker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Sample toxic texts for testing
SAMPLE_TOXIC_TEXTS = [
    "You are an idiot and I hate you",
    "This is the worst thing ever, die in a fire",
    "I will find you and hurt you badly",
    "You're so stupid, kill yourself",
    "Everyone hates you, you worthless person",
    "I hope something terrible happens to you",
    "You deserve to suffer for what you did",
    "Nobody likes you, go away forever",
    "You're disgusting and should be ashamed",
    "I can't believe how pathetic you are",
]


async def run_coevolution_cycle(
    classifier: TrainedClassifierAdapter,
    coevolution: AdversarialCoevolution,
    cycle_num: int,
) -> dict:
    """Run a single co-evolution cycle."""
    logger.info(f"\n{'='*60}")
    logger.info(f"CO-EVOLUTION CYCLE {cycle_num}")
    logger.info(f"{'='*60}")

    # Step 1: Run standard attacks
    logger.info("\n[1] Running standard attacks...")
    config = AttackConfig(batch_size=10, num_variants=10)
    runner = AttackRunner(config, classifier)
    attack_result = runner.run_batch()

    standard_evasion_rate = attack_result.evasion_rate
    logger.info(f"Standard attack evasion rate: {standard_evasion_rate:.1%}")

    # Get blocked strategies (those with low evasion)
    breakdown = attack_result.get_strategy_breakdown()
    blocked = [
        name for name, stats in breakdown.items()
        if stats["total"] > 0 and stats["evasions"] / stats["total"] < 0.3
    ]
    logger.info(f"Blocked strategies: {blocked}")

    # Step 2: Run evolved attacks
    logger.info("\n[2] Running evolved attacks...")
    evolved_results = coevolution.attacker.apply_evolved_strategies(
        SAMPLE_TOXIC_TEXTS[0],
        num_variants=5,
    )

    evolved_evasions = 0
    if evolved_results:
        evolved_texts = [r["evasion"] for r in evolved_results]
        predictions = classifier.predict(evolved_texts)
        evolved_evasions = sum(1 for p in predictions if p["label"] == 0)
        evolved_evasion_rate = evolved_evasions / len(evolved_results)
        logger.info(f"Evolved attack evasion rate: {evolved_evasion_rate:.1%}")
    else:
        evolved_evasion_rate = 0.0
        logger.info("No evolved strategies available yet")

    # Combined evasion rate
    total_attacks = attack_result.total_variants + len(evolved_results)
    total_evasions = attack_result.total_evasions + evolved_evasions
    combined_rate = total_evasions / total_attacks if total_attacks > 0 else 0.0

    logger.info(f"\nCombined evasion rate: {combined_rate:.1%}")

    # Step 3: Run co-evolution decision
    logger.info("\n[3] Co-evolution decision...")
    cycle_result = await coevolution.run_cycle(
        sample_texts=SAMPLE_TOXIC_TEXTS,
        current_evasion_rate=combined_rate,
        blocked_strategies=blocked,
    )

    logger.info(f"Action taken: {cycle_result['action']}")
    logger.info(f"Details: {cycle_result['details']}")

    return {
        "cycle": cycle_num,
        "standard_evasion_rate": standard_evasion_rate,
        "evolved_evasion_rate": evolved_evasion_rate,
        "combined_evasion_rate": combined_rate,
        "action": cycle_result["action"],
        "new_strategies": cycle_result["details"].get("new_strategies", 0),
    }


async def main():
    parser = argparse.ArgumentParser(description="Run adversarial co-evolution")
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of co-evolution cycles",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/adversarial-retrained"),
        help="Path to classifier model",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to generate new strategies (requires Ollama)",
    )
    parser.add_argument(
        "--llm-endpoint",
        type=str,
        default="http://localhost:11434",
        help="Ollama API endpoint",
    )

    args = parser.parse_args()

    # Load classifier
    logger.info(f"Loading classifier from {args.model_path}...")
    if not args.model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        logger.info("Run retraining first: python scripts/retrain_from_samples.py")
        sys.exit(1)

    classifier = TrainedClassifierAdapter(args.model_path)
    classifier.load()

    # Create evolving attacker
    logger.info("Creating evolving attacker...")
    attacker = create_evolving_attacker(
        classifier=classifier,
        llm_endpoint=args.llm_endpoint,
        include_builtin=True,
    )

    logger.info(f"Loaded {len(attacker.evolved_strategies)} evolved strategies")
    for s in attacker.evolved_strategies:
        logger.info(f"  - {s.name}: {s.success_rate:.1%} success rate")

    # Create co-evolution manager
    coevolution = AdversarialCoevolution(
        classifier=classifier,
        evolving_attacker=attacker,
        attack_evolution_threshold=0.2,
        defender_retrain_threshold=0.5,
    )

    # Run cycles
    results = []
    for i in range(1, args.cycles + 1):
        try:
            result = await run_coevolution_cycle(classifier, coevolution, i)
            results.append(result)
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
            break
        except Exception as e:
            logger.error(f"Cycle {i} failed: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("CO-EVOLUTION SUMMARY")
    print("=" * 60)
    print(f"{'Cycle':<8} {'Standard':<12} {'Evolved':<12} {'Combined':<12} {'Action':<20}")
    print("-" * 60)

    for r in results:
        std_rate = f"{r['standard_evasion_rate']:.1%}"
        evo_rate = f"{r['evolved_evasion_rate']:.1%}"
        comb_rate = f"{r['combined_evasion_rate']:.1%}"
        print(
            f"{r['cycle']:<8} "
            f"{std_rate:<12} "
            f"{evo_rate:<12} "
            f"{comb_rate:<12} "
            f"{r['action']:<20}"
        )

    print("=" * 60)

    # Final statistics
    stats = attacker.get_statistics()
    print(f"\nEvolved Strategies: {stats['total_strategies']}")
    if stats['total_strategies'] > 0:
        print(f"Average Success Rate: {stats['avg_success_rate']:.1%}")
        print(f"Best Strategy: {stats['best_strategy']}")


if __name__ == "__main__":
    asyncio.run(main())
