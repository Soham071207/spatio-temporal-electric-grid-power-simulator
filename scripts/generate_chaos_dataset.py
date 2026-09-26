"""
Generate the chaos scenario training dataset.

Loads real grid snapshots from multiple dates, runs hundreds of
chaos scenarios on each, and saves the results as a parquet file
for training the ML Healing Agent.

Output: data/processed/chaos_scenarios.parquet
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from digital_twin.snapshot_builder import SnapshotBuilder
from digital_twin.scenario_engine import ScenarioEngine, results_to_dataframe


def main():
    print("=" * 60)
    print("CHAOS SCENARIO DATASET GENERATOR")
    print("=" * 60)

    sb = SnapshotBuilder()

    # Pick diverse timestamps across seasons, times of day, weekdays/weekends
    target_dates = [
        # Winter weekday peak
        "2023-01-18 19:00:00",
        "2024-02-14 18:00:00",
        # Winter weekend
        "2023-02-12 15:00:00",
        # Spring
        "2023-04-15 12:00:00",
        "2024-04-10 14:00:00",
        # Summer peak (high solar, high demand)
        "2023-07-20 14:00:00",
        "2024-08-05 15:00:00",
        "2025-06-15 14:00:00",
        # Summer evening (solar gone, still hot)
        "2023-08-10 21:00:00",
        # Autumn
        "2023-10-15 12:00:00",
        "2024-11-20 18:00:00",
        # Night (low demand, nuclear base)
        "2023-03-15 03:00:00",
        "2024-06-01 04:00:00",
        # High wind day
        "2023-11-28 14:00:00",
        # Low wind + low solar
        "2024-01-05 07:00:00",
    ]

    scenarios_per_snapshot = 200  # How many scenarios per grid snapshot
    all_results = []
    engine = ScenarioEngine(seed=42)

    for i, dt in enumerate(target_dates):
        print(f"\n{'-' * 60}")
        print(f"[{i+1}/{len(target_dates)}] Building snapshot for {dt}...")

        try:
            state = sb.build_snapshot(dt)
        except Exception as e:
            print(f"  [WARN] Failed to build snapshot: {e}")
            continue

        if len(state.generators) < 5:
            print(f"  [WARN] Too few generators ({len(state.generators)}), skipping.")
            continue

        print(f"  Grid: {state.total_generation_mw:.0f} MW gen, "
              f"{state.total_demand_mw:.0f} MW demand, "
              f"{len(state.generators)} plants")

        t0 = time.time()
        results = engine.generate_random_scenarios(state, n_scenarios=scenarios_per_snapshot)
        elapsed = time.time() - t0

        all_results.extend(results)
        print(f"  [OK] Generated {len(results)} scenarios in {elapsed:.1f}s")

        # Print stats
        if results:
            scores = [r.resilience_score_after_fault for r in results]
            costs = [r.healing_cost_eur for r in results]
            print(f"  Resilience range: {min(scores):.0f} - {max(scores):.0f}")
            print(f"  Healing cost range: EUR {min(costs):.0f} - EUR {max(costs):.0f}")

    # Convert to DataFrame and save
    print(f"\n{'=' * 60}")
    print(f"TOTAL SCENARIOS: {len(all_results)}")

    if not all_results:
        print("No results generated! Check data availability.")
        return

    df = results_to_dataframe(all_results)

    out_path = os.path.join("data", "processed", "chaos_scenarios.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Shape: {df.shape}")

    # Also save as CSV for inspection
    csv_path = os.path.join("data", "processed", "chaos_scenarios_sample.csv")
    df.head(500).to_csv(csv_path, index=False)
    print(f"Sample CSV saved to {csv_path}")

    # Summary statistics
    print(f"\n{'-' * 40}")
    print("DATASET SUMMARY")
    print(f"{'-' * 40}")
    print(f"Scenario types:")
    print(df["scenario_type"].value_counts().to_string())
    print(f"\nResilience after fault (mean): {df['resilience_after_fault'].mean():.1f}")
    print(f"Resilience after healing (mean): {df['resilience_after_healing'].mean():.1f}")
    print(f"Healing improvement (mean): {(df['resilience_after_healing'] - df['resilience_after_fault']).mean():.1f}")
    print(f"Average cost: EUR {df['healing_cost_eur'].mean():.0f}")


if __name__ == "__main__":
    main()
