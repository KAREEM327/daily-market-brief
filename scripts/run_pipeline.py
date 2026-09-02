#!/usr/bin/env python3
"""
Daily Market Brief — Master Pipeline Runner
Runs all engines in sequence, then generates the brief.
Designed for GitHub Actions (free) or any cron environment.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, Any

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from data_ingest import run_full_ingestion
from regime_engine import run_regime_analysis
from correlation_engine import run_correlation_analysis
from analogue_engine import run_analogue_analysis
from options_engine import run_options_analysis
from smart_money import run_smart_money_analysis
from futures_engine import analyze_futures  # <-- NEW
from brief_generator import run_brief_generation

# ─── Config ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dmb.pipeline")

# ─── Pipeline Steps ──────────────────────────────────────────────────────
STEPS = [
    ("Data Ingestion", run_full_ingestion),
    ("Regime Engine", run_regime_analysis),
    ("Correlation Engine", run_correlation_analysis),
    ("Analogue Engine", run_analogue_analysis),
    ("Futures Engine", analyze_futures),   # <-- INSERTED HERE
    ("Options Engine", run_options_analysis),
    ("Smart Money Engine", run_smart_money_analysis),
    ("Brief Generator", run_brief_generation),
]


def run_pipeline(force: bool = False) -> Dict[str, Any]:
    """Run the complete pipeline."""
    log.info("=" * 60)
    log.info("DAILY MARKET BRIEF — MASTER PIPELINE")
    log.info(f"Date: {date.today().isoformat()}")
    log.info("=" * 60)

    start_time = time.time()
    results = {"date": date.today().isoformat(), "steps": {}}

    for step_name, step_func in STEPS:
        step_start = time.time()
        log.info(f"\n{'─' * 40}")
        log.info(f"STEP: {step_name}")
        log.info(f"{'─' * 40}")

        try:
            result = step_func()
            elapsed = time.time() - step_start
            results["steps"][step_name] = {
                "status": "success",
                "elapsed_seconds": round(elapsed, 1),
                "output_keys": list(result.keys()) if isinstance(result, dict) else [],
            }
            log.info(f"✓ {step_name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - step_start
            results["steps"][step_name] = {
                "status": "failed",
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e),
            }
            log.error(f"✗ {step_name} failed after {elapsed:.1f}s: {e}")
            if step_name != "Brief Generator":  # Continue on non-final step failures
                continue
            else:
                break

    total_elapsed = time.time() - start_time
    results["total_elapsed_seconds"] = round(total_elapsed, 1)
    results["overall_status"] = "success" if all(
        s["status"] == "success" for s in results["steps"].values()
    ) else "partial"

    log.info(f"\n{'=' * 60}")
    log.info(f"PIPELINE COMPLETE — {total_elapsed:.1f}s total")
    log.info(f"{'=' * 60}")

    for name, info in results["steps"].items():
        status = "✓" if info["status"] == "success" else "✗"
        log.info(f"  {status} {name}: {info['elapsed_seconds']:.1f}s")

    return results


def main():
    """Entry point."""
    force = "--force" in sys.argv
    results = run_pipeline(force=force)

    # Save pipeline log
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{date.today().isoformat()}.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Exit code for GitHub Actions
    if results["overall_status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()