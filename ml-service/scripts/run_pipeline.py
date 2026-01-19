#!/usr/bin/env python3
"""CLI script for running the Adversarial MLOps Pipeline.

This script provides a command-line interface for:
- Running pipeline cycles
- Checking pipeline status
- Viewing execution history
- Configuring pipeline parameters

Usage:
    python scripts/run_pipeline.py run [--cycle-id ID]
    python scripts/run_pipeline.py status
    python scripts/run_pipeline.py history [--limit N]
    python scripts/run_pipeline.py config [--show | --update KEY=VALUE]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_service.pipeline.config import PipelineConfig
from ml_service.pipeline.orchestrator import create_pipeline, AdversarialPipelineOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | None) -> PipelineConfig:
    """Load configuration from file or use defaults."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            return PipelineConfig.from_yaml(path)
        else:
            print(f"Warning: Config file not found: {config_path}")

    return PipelineConfig()


def print_json(data: dict) -> None:
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, default=str))


async def run_cycle(
    pipeline: AdversarialPipelineOrchestrator,
    cycle_id: str | None = None,
) -> int:
    """Run a pipeline cycle.

    Args:
        pipeline: Pipeline orchestrator.
        cycle_id: Optional cycle identifier.

    Returns:
        Exit code (0 for success).
    """
    print("=" * 60)
    print("ADVERSARIAL MLOPS PIPELINE")
    print("=" * 60)

    result = await pipeline.run_cycle(cycle_id)

    print("\n" + "=" * 60)
    print("CYCLE RESULT")
    print("=" * 60)

    print(f"Cycle ID:        {result.cycle_id}")
    print(f"State:           {result.state.value}")
    print(f"Duration:        {result.duration_seconds:.1f}s")

    if result.attack_result:
        ar = result.attack_result
        print(f"\nAttack Results:")
        print(f"  Total Variants:   {ar.total_variants}")
        print(f"  Total Evasions:   {ar.total_evasions}")
        print(f"  Evasion Rate:     {ar.evasion_rate:.1%}")
        print(f"  Strategies:       {', '.join(ar.strategies_used)}")

    if result.quality_decision:
        qd = result.quality_decision
        print(f"\nQuality Decision:")
        print(f"  Decision:         {qd.decision.value.upper()}")
        print(f"  Needs Retraining: {qd.needs_retraining}")
        if qd.failure_reasons:
            print(f"  Failure Reasons:  {', '.join(r.value for r in qd.failure_reasons)}")

    print(f"\nSamples Collected: {result.samples_collected}")
    print(f"Retraining:        {result.retraining_triggered}")

    if result.retraining_metrics:
        print(f"\nRetraining Metrics:")
        for k, v in result.retraining_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if result.promotion_decision:
        pd = result.promotion_decision
        print(f"\nPromotion Decision:")
        print(f"  Status:      {pd.status.value}")
        print(f"  Message:     {pd.message}")
        print(f"  Improvement: {pd.improvement:.3f}")

    if result.error:
        print(f"\nError: {result.error}")
        return 1

    print("\n" + "=" * 60)
    return 0


def show_status(pipeline: AdversarialPipelineOrchestrator) -> int:
    """Show pipeline status.

    Args:
        pipeline: Pipeline orchestrator.

    Returns:
        Exit code.
    """
    status = pipeline.get_status()
    print_json(status)
    return 0


def show_history(pipeline: AdversarialPipelineOrchestrator, limit: int = 10) -> int:
    """Show pipeline history.

    Args:
        pipeline: Pipeline orchestrator.
        limit: Number of entries to show.

    Returns:
        Exit code.
    """
    history = pipeline.load_history()

    if not history:
        print("No pipeline history found.")
        return 0

    print(f"Pipeline History (showing last {min(limit, len(history))} of {len(history)} cycles):\n")

    for entry in history[-limit:]:
        print(f"Cycle: {entry.get('cycle_id', 'unknown')}")
        print(f"  Started:    {entry.get('started_at', 'N/A')}")
        print(f"  State:      {entry.get('state', 'N/A')}")

        attack = entry.get("attack_result", {})
        if attack:
            print(f"  Evasion:    {attack.get('evasion_rate', 0):.1%}")

        quality = entry.get("quality_decision", {})
        if quality:
            print(f"  Decision:   {quality.get('decision', 'N/A')}")

        print(f"  Retrained:  {entry.get('retraining_triggered', False)}")
        print()

    return 0


def show_config(config: PipelineConfig) -> int:
    """Show current configuration.

    Args:
        config: Pipeline configuration.

    Returns:
        Exit code.
    """
    print("Current Pipeline Configuration:\n")
    print(config.model_dump_yaml())
    return 0


def show_metrics(pipeline: AdversarialPipelineOrchestrator) -> int:
    """Show pipeline metrics.

    Args:
        pipeline: Pipeline orchestrator.

    Returns:
        Exit code.
    """
    metrics = pipeline.get_metrics()

    if not metrics:
        print("No metrics available (no pipeline cycles run yet).")
        return 0

    print("Pipeline Metrics:\n")
    print(f"  Total Cycles:      {metrics.get('total_cycles', 0)}")
    print(f"  Avg Evasion Rate:  {metrics.get('avg_evasion_rate', 0):.1%}")
    print(f"  Max Evasion Rate:  {metrics.get('max_evasion_rate', 0):.1%}")
    print(f"  Min Evasion Rate:  {metrics.get('min_evasion_rate', 0):.1%}")
    print(f"  Retraining Count:  {metrics.get('retraining_count', 0)}")
    print(f"  Success Rate:      {metrics.get('success_rate', 0):.1%}")
    print(f"  Samples Collected: {metrics.get('total_samples_collected', 0)}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adversarial MLOps Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py run                  # Run a single pipeline cycle
  python run_pipeline.py run --cycle-id test  # Run with custom cycle ID
  python run_pipeline.py status               # Show pipeline status
  python run_pipeline.py history --limit 5    # Show last 5 cycles
  python run_pipeline.py config               # Show configuration
  python run_pipeline.py metrics              # Show aggregated metrics
        """,
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a pipeline cycle")
    run_parser.add_argument(
        "--cycle-id",
        type=str,
        help="Custom cycle identifier",
    )
    run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock classifier (for testing)",
    )

    # Status command
    subparsers.add_parser("status", help="Show pipeline status")

    # History command
    history_parser = subparsers.add_parser("history", help="Show execution history")
    history_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of entries to show",
    )

    # Config command
    subparsers.add_parser("config", help="Show configuration")

    # Metrics command
    subparsers.add_parser("metrics", help="Show pipeline metrics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup
    setup_logging(args.verbose)
    config = load_config(args.config)

    # Create pipeline
    pipeline = create_pipeline(config)

    # Execute command
    if args.command == "run":
        return asyncio.run(run_cycle(pipeline, args.cycle_id))
    elif args.command == "status":
        return show_status(pipeline)
    elif args.command == "history":
        return show_history(pipeline, args.limit)
    elif args.command == "config":
        return show_config(config)
    elif args.command == "metrics":
        return show_metrics(pipeline)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
