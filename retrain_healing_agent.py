# -*- coding: utf-8 -*-
"""
Retrain healing agent — regenerates chaos_scenarios.parquet and fits model.

Steps:
  1. Generates 3000 chaos scenarios using ScenarioEngine + real snapshots
  2. Saves fresh parquet with all 7 action columns including a_ramp_nuclear
  3. Trains HistGradientBoosting model on new data
  4. Saves model to data/models/healing_agent.pkl

Usage:
    python retrain_healing_agent.py
    python retrain_healing_agent.py --scenarios 1000  # faster, less accurate
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')  # Windows cp1252 fix

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=3000)
parser.add_argument("--dates", type=int, default=50, help="Number of snapshot dates to sample from")
parser.add_argument("--skip-data-gen", action="store_true", help="Skip generation, just retrain from existing parquet")
args = parser.parse_args()

print("=" * 60)
print("  HEALING AGENT RETRAIN PIPELINE")
print("=" * 60)

# ── Step 1: Load snapshot builder ───────────────────────────────────────────
if not args.skip_data_gen:
    print("\n[1/3] Loading SnapshotBuilder...")
    from digital_twin.snapshot_builder import SnapshotBuilder
    from digital_twin.scenario_engine import ScenarioEngine, results_to_dataframe
    import pandas as pd
    import numpy as np
    import random

    sb = SnapshotBuilder()

    # Sample diverse dates spanning 2019-2024 (varied seasons / load conditions)
    import datetime
    random.seed(42)
    all_dates = []
    for year in range(2019, 2025):
        for _ in range(args.dates // 6):
            month = random.randint(1, 12)
            day   = random.randint(1, 28)
            hour  = random.randint(0, 23)
            all_dates.append(f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00")

    scenarios_per_date = max(1, args.scenarios // len(all_dates))
    print(f"[1/3] Sampling {len(all_dates)} dates × {scenarios_per_date} scenarios = target {args.scenarios}")

    all_results = []
    built_dates = 0

    # ScenarioEngine is constructed ONCE with an int seed — state is passed per-call
    engine = ScenarioEngine(seed=42)

    for idx, dt_str in enumerate(all_dates):
        try:
            state = sb.build_snapshot(dt_str)
            results = engine.generate_random_scenarios(state, n_scenarios=scenarios_per_date)
            all_results.extend(results)
            built_dates += 1
            done = len(all_results)
            print(f"  {dt_str}: +{len(results)} scenarios (total: {done})", flush=True)
            if done >= args.scenarios:
                break
        except Exception as e:
            print(f"  SKIP {dt_str}: {e}")

    if not all_results:
        print("ERROR: No scenarios generated. Check data pipeline.")
        sys.exit(1)

    print(f"\n[2/3] Generated {len(all_results)} total scenarios from {built_dates} dates")
    df = results_to_dataframe(all_results)

    out_path = "data/processed/chaos_scenarios.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[2/3] Saved {len(df)} rows to {out_path}")
    print(f"      Columns: {[c for c in df.columns if c.startswith('a_')]}")

else:
    print("\n[1-2/3] Skipping data generation (--skip-data-gen flag set)")
    import pandas as pd
    out_path = "data/processed/chaos_scenarios.parquet"

# ── Step 3: Train ─────────────────────────────────────────────────────────────
print("\n[3/3] Training HealingAgent model...")
from digital_twin.healing_agent import HealingAgent, ACTION_NAMES, FEATURE_NAMES

df = pd.read_parquet(out_path)
print(f"  Dataset size: {len(df)} rows")
print(f"  Expected action cols: {ACTION_NAMES}")
print(f"  Available action cols: {[c for c in df.columns if c.startswith('a_')]}")

# Check all expected columns exist; add zeroed nuclear column if missing
import numpy as np
for col in ACTION_NAMES:
    if col not in df.columns:
        print(f"  WARNING: '{col}' missing from dataset — adding zero column")
        df[col] = 0.0

ha = HealingAgent()
metrics = ha.train(data_path=out_path)

print("\n  Training metrics:")
for k, v in metrics.items():
    print(f"    {k}: {v}")

ha.save("data/models/healing_agent.pkl")

print("\n" + "=" * 60)
print("  RETRAIN COMPLETE")
print(f"  Model saved: data/models/healing_agent.pkl")
print(f"  Action outputs: {len(ACTION_NAMES)} ({', '.join(ACTION_NAMES)})")
print("=" * 60)
