"""
Test suite for the Grid Chaos Engine.

Exercises all 7 chaos modes, cascade simulation, resilience scoring,
and the redispatch engine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin.simulator import ChaosEngine, build_demo_grid_state, ChaosMode
from digital_twin.cascade import CascadeEngine
from digital_twin.metrics import calculate_resilience_score, get_risk_label
from digital_twin.redispatch import RedispatchEngine


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_initial_state():
    separator("TEST: Initial Grid State")
    state = build_demo_grid_state()
    summary = state.summary()

    print(f"  Timestamp:         {summary['timestamp']}")
    print(f"  Total Demand:      {summary['total_demand_mw']:,.0f} MW")
    print(f"  Total Generation:  {summary['total_generation_mw']:,.0f} MW")
    print(f"  Imports:           {summary['imports_mw']:,.0f} MW")
    print(f"  Deficit:           {summary['deficit_mw']:,.0f} MW")
    print(f"  Reserve Margin:    {summary['reserve_margin_pct']:.1f}%")
    print(f"  Frequency:         {summary['frequency_hz']:.3f} Hz")
    print(f"  Online Generators: {summary['online_generators']}")
    print(f"  Active Lines:      {summary['active_lines']}")
    print(f"  Generation Mix:")
    for tech, mw in summary['generation_mix'].items():
        print(f"    {tech:10s}: {mw:>8,.0f} MW")

    assert summary['deficit_mw'] == 0.0, "Initial state should have no deficit"
    assert summary['frequency_hz'] == 50.0, "Initial frequency should be 50 Hz"
    print("\n  ✅ PASSED")


def test_generator_failure():
    separator("TEST: 🐒 Generator Failure (Chaos Monkey)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    # Trip the biggest nuclear plant
    event = engine.inject_generator_failure("NUC_Cofrentes")
    print(f"  Event: {event.description}")
    print(f"  Impact: {event.impact_mw:.0f} MW lost")
    print(f"  New Frequency: {engine.state.frequency_hz:.3f} Hz")
    print(f"  New Deficit: {engine.state.deficit_mw:.0f} MW")

    assert event.impact_mw > 0, "Should have lost generation"
    assert engine.state.frequency_hz < 50.0, "Frequency should drop"
    print("\n  ✅ PASSED")


def test_line_outage():
    separator("TEST: 🔌 Transmission Line Outage")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    event = engine.inject_line_outage("L_Madrid_CMancha")
    print(f"  Event: {event.description}")
    print(f"  Impact: {event.impact_mw:.0f} MW flow disrupted")

    # Check that the line is actually down
    assert not engine.state.lines["L_Madrid_CMancha"].is_active
    print("\n  ✅ PASSED")


def test_renewable_shock():
    separator("TEST: ☀️ Renewable Shock (Solar -60%)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    before_gen = state.total_generation_mw
    event = engine.inject_renewable_shock("Solar", 60.0)
    after_gen = engine.state.total_generation_mw

    print(f"  Event: {event.description}")
    print(f"  Generation Before: {before_gen:,.0f} MW")
    print(f"  Generation After:  {after_gen:,.0f} MW")
    print(f"  Lost: {event.impact_mw:,.0f} MW")
    print(f"  Frequency: {engine.state.frequency_hz:.3f} Hz")

    assert event.impact_mw > 0, "Should have lost solar generation"
    print("\n  ✅ PASSED")


def test_demand_surge():
    separator("TEST: 🔥 Demand Surge (Madrid +30%)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    before_demand = state.regional_demand_mw["Madrid"]
    event = engine.inject_demand_surge("Madrid", 30.0)
    after_demand = engine.state.regional_demand_mw["Madrid"]

    print(f"  Event: {event.description}")
    print(f"  Madrid Demand Before: {before_demand:,.0f} MW")
    print(f"  Madrid Demand After:  {after_demand:,.0f} MW")
    print(f"  Surge: +{event.impact_mw:,.0f} MW")

    assert after_demand > before_demand
    print("\n  ✅ PASSED")


def test_interconnection_failure():
    separator("TEST: 🔌 Island Mode (France Disconnected)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    event = engine.inject_interconnection_failure("France")
    print(f"  Event: {event.description}")
    print(f"  France capacity now: {engine.state.interconnections['France']:.0f} MW")

    assert engine.state.interconnections["France"] == 0.0
    print("\n  ✅ PASSED")


def test_random_chaos():
    separator("TEST: 🐒 Random Chaos (3 random failures)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    events = engine.run_random_chaos(n_failures=3)
    print(f"  Injected {len(events)} random failures:")
    for e in events:
        print(f"    - {e.description}")

    assert len(events) == 3
    print("\n  ✅ PASSED")


def test_targeted_chaos():
    separator("TEST: Level 2 — Targeted Chaos (highest-loaded)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    events = engine.run_targeted_chaos()
    print(f"  Targeted {len(events)} components:")
    for e in events:
        print(f"    - {e.description}")

    assert len(events) >= 1
    print("\n  ✅ PASSED")


def test_storm_mode():
    separator("TEST: 🌪️ Storm Mode (Andalucía)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    events = engine.run_storm_mode("Andalucía", n_failures=3)
    print(f"  Storm hit Andalucía with {len(events)} failures:")
    for e in events:
        print(f"    - {e.description}")

    assert len(events) >= 1
    print("\n  ✅ PASSED")


def test_cascade_simulation():
    separator("TEST: 💥 Cascade Simulation")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    # Create an overload situation by tripping multiple lines
    engine.inject_line_outage("L_Madrid_CMancha")
    engine.inject_line_outage("L_Madrid_Castilla")

    # Force an overload on the remaining Madrid line
    engine.state.lines["L_Madrid_Valencia"].current_flow_mw = 3000  # Over its 2500 MW capacity

    cascade_engine = CascadeEngine(engine)
    result = cascade_engine.run_cascade(initial_trigger="Madrid corridor collapse")

    print(f"  Cascade Depth:       {result.cascade_depth}")
    print(f"  Cascade Size:        {result.cascade_size} components lost")
    print(f"  Geographic Spread:   {result.geographic_spread} regions")
    print(f"  Affected Regions:    {', '.join(result.affected_regions)}")
    print(f"  Final Deficit:       {result.final_deficit_mw:,.0f} MW")
    print(f"  Final Frequency:     {result.final_frequency_hz:.3f} Hz")
    print(f"  System Stable:       {result.system_stable}")
    print(f"  Blackout:            {result.blackout}")
    if result.recovery_actions:
        print(f"  Recovery Actions:")
        for action in result.recovery_actions[:5]:
            print(f"    → {action}")

    print("\n  ✅ PASSED")


def test_resilience_scoring():
    separator("TEST: Resilience Scoring")
    initial = build_demo_grid_state()
    engine = ChaosEngine(initial)

    # Scenario 1: Minor disruption
    engine.inject_generator_failure("GAS_Arcos")
    score_minor = calculate_resilience_score(initial, engine.state)
    print(f"  Minor disruption (1 gas plant):")
    print(f"    Score: {score_minor['score']}/100 — {score_minor['label']}")

    # Scenario 2: Major disruption
    engine.reset()
    engine.inject_renewable_shock("Solar", 70.0)
    engine.inject_generator_failure("NUC_Cofrentes")
    engine.inject_generator_failure("NUC_Trillo")
    score_major = calculate_resilience_score(initial, engine.state)
    print(f"\n  Major disruption (70% solar loss + 2 nuclear plants):")
    print(f"    Score: {score_major['score']}/100 — {score_major['label']}")
    print(f"    Breakdown:")
    for k, v in score_major['breakdown'].items():
        print(f"      {k}: {v}")

    assert score_minor['score'] > score_major['score'], "Minor should score higher than major"
    print("\n  ✅ PASSED")


def test_redispatch():
    separator("TEST: Redispatch Engine")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    # Create a deficit
    engine.inject_renewable_shock("Solar", 50.0)
    print(f"  Deficit after solar shock: {engine.state.deficit_mw:,.0f} MW")

    # Run redispatch
    redispatch = RedispatchEngine()
    result = redispatch.execute_redispatch(engine.state)

    print(f"  Status: {result['status']}")
    print(f"  Deficit Before: {result['deficit_before_mw']:,.0f} MW")
    print(f"  Deficit After:  {result['deficit_after_mw']:,.0f} MW")
    print(f"  Total Cost:     €{result['total_cost_eur']:,.0f}")
    print(f"  Actions Taken:  {len(result['actions'])}")
    for action in result['actions'][:5]:
        if action['type'] == 'ramp_up':
            print(f"    → Ramped {action['generator_id']} ({action['technology']}): +{action['ramped_mw']} MW — €{action['cost_eur']:,.0f}")
        elif action['type'] == 'emergency_import':
            print(f"    → Emergency imports: +{action['imported_mw']} MW — €{action['cost_eur']:,.0f}")

    print("\n  ✅ PASSED")


def test_what_if():
    separator("TEST: What-If Simulator")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)

    def wind_drop_scenario(eng):
        eng.inject_renewable_shock("Wind", 50.0)

    comparison = engine.what_if(wind_drop_scenario)

    print(f"  Scenario: 50% Wind Generation Drop")
    print(f"  {'Metric':<25s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
    print(f"  {'-'*55}")
    for key in ['total_demand_mw', 'total_generation_mw', 'deficit_mw', 'reserve_margin_pct', 'frequency_hz']:
        before = comparison['before'].get(key, 'N/A')
        after = comparison['after'].get(key, 'N/A')
        delta = comparison['deltas'].get(key, 'N/A')
        if isinstance(before, float):
            print(f"  {key:<25s} {before:>10.1f} {after:>10.1f} {delta:>+10.1f}")

    print("\n  ✅ PASSED")


def test_full_pipeline():
    separator("TEST: Full Pipeline (Chaos → Cascade → Redispatch → Score)")
    initial = build_demo_grid_state()
    engine = ChaosEngine(initial)

    # 1. Inject chaos
    print("  1. Injecting chaos...")
    engine.inject_generator_failure("NUC_Cofrentes")
    engine.inject_renewable_shock("Wind", 40.0)
    engine.inject_line_outage("L_Cataluña_Valencia")

    # 2. Run cascade
    print("  2. Running cascade...")
    cascade = CascadeEngine(engine)
    cascade_result = cascade.run_cascade(initial_trigger="Multi-fault scenario")
    print(f"     Cascade depth: {cascade_result.cascade_depth}, size: {cascade_result.cascade_size}")

    # 3. Redispatch
    print("  3. Running redispatch...")
    redispatch = RedispatchEngine()
    redispatch_result = redispatch.execute_redispatch(engine.state)
    print(f"     Status: {redispatch_result['status']}, Cost: €{redispatch_result['total_cost_eur']:,.0f}")

    # 4. Score
    print("  4. Calculating resilience score...")
    score = calculate_resilience_score(
        initial,
        engine.state,
        cascade_depth=cascade_result.cascade_depth,
        cascade_size=cascade_result.cascade_size,
        geographic_spread=cascade_result.geographic_spread,
    )
    print(f"     Resilience: {score['score']}/100 — {score['label']}")

    # 5. Summary
    print(f"\n  Final Grid State:")
    summary = engine.state.summary()
    print(f"    Demand:     {summary['total_demand_mw']:>10,.0f} MW")
    print(f"    Generation: {summary['total_generation_mw']:>10,.0f} MW")
    print(f"    Deficit:    {summary['deficit_mw']:>10,.0f} MW")
    print(f"    Frequency:  {summary['frequency_hz']:>10.3f} Hz")

    print("\n  ✅ PASSED")


if __name__ == "__main__":
    print("=" * 70)
    print("  GRID CHAOS ENGINE — Full Test Suite")
    print("=" * 70)

    test_initial_state()
    test_generator_failure()
    test_line_outage()
    test_renewable_shock()
    test_demand_surge()
    test_interconnection_failure()
    test_random_chaos()
    test_targeted_chaos()
    test_storm_mode()
    test_cascade_simulation()
    test_resilience_scoring()
    test_redispatch()
    test_what_if()
    test_full_pipeline()

    separator("ALL TESTS PASSED ✅")
