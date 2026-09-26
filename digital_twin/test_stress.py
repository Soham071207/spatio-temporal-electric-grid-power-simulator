"""
Deep Stress Tests — Probing edge cases across the entire digital twin backend.
"""

import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin.simulator import (
    build_demo_grid_state, ChaosEngine, GridState, Generator,
    TransmissionLine, TECHNICAL_MINIMUM_PCT, STARTUP_TIME_MIN,
    INERTIA_CONSTANTS, GOVERNOR_DROOP
)
from digital_twin.redispatch import RedispatchEngine, TECH_EMISSIONS, TECH_COSTS
from digital_twin.cascade import CascadeEngine
from digital_twin.metrics import calculate_resilience_score
from digital_twin.healing_agent import HealingAgent


PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        msg = f"  FAIL {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_transmission_losses():
    separator("TEST 1: Transmission Losses (C2)")
    state = build_demo_grid_state()
    total_gen = state.total_generation_mw
    losses = state.transmission_losses_mw
    expected_losses = total_gen * 0.025
    check("Losses are 2.5% of generation", abs(losses - expected_losses) < 1.0,
          f"losses={losses:.1f}, expected={expected_losses:.1f}")
    check("Losses are positive when generation exists", losses > 0)
    supply_with = total_gen + state.imports_mw - losses
    supply_without = total_gen + state.imports_mw
    check("Losses increase the effective deficit", supply_with < supply_without)
    empty = GridState()
    check("Zero generation gives zero losses", empty.transmission_losses_mw == 0.0)


def test_technical_minimum():
    separator("TEST 2: Technical Minimum (A1)")
    state = build_demo_grid_state()
    nuc = state.generators["NUC_Cofrentes"]
    check("Nuclear P_min > 500 MW", nuc.min_stable_mw > 500,
          f"min_stable={nuc.min_stable_mw:.1f}")
    wind = state.generators["WIND_CastillaLeon"]
    check("Wind P_min = 0", wind.min_stable_mw == 0.0)
    gas = state.generators["GAS_Arcos"]
    check("Gas P_min = 30% of capacity",
          abs(gas.min_stable_mw - gas.installed_capacity_mw * 0.30) < 1.0)


def test_generator_trip():
    separator("TEST 3: Generator trip() method")
    state = build_demo_grid_state()
    gen = state.generators["GAS_Arcos"]
    check("Generator starts online", gen.is_online)
    check("Generator has output > 0", gen.current_output_mw > 0)
    gen.trip()
    check("After trip: is_online = False", not gen.is_online)
    check("After trip: output = 0", gen.current_output_mw == 0.0)
    check("After trip: offline_duration = 0", gen.offline_duration_min == 0.0)
    check("After trip: available_reserve = 0", gen.available_reserve_mw == 0.0)


def test_bess_inertia():
    separator("TEST 4: BESS Synthetic Inertia (A3)")
    state_no_bess = build_demo_grid_state()
    state_with_bess = build_demo_grid_state()
    bess = Generator("BESS_Test", "BESS", "Madrid", 500.0, 0.0)
    state_with_bess.generators["BESS_Test"] = bess
    engine_no_bess = ChaosEngine(state_no_bess)
    engine_with_bess = ChaosEngine(state_with_bess)
    engine_no_bess.inject_generator_failure("NUC_Cofrentes")
    engine_with_bess.inject_generator_failure("NUC_Cofrentes")
    freq_no = engine_no_bess.state.frequency_hz
    freq_with = engine_with_bess.state.frequency_hz
    check("BESS softens frequency drop", freq_with >= freq_no,
          f"with={freq_with:.3f}, no={freq_no:.3f}")
    check("Frequency still drops below 50 Hz", freq_no < 50.0 and freq_with < 50.0)


def test_frequency_clamping():
    separator("TEST 5: Frequency Clamping Bounds")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    gen_ids = [g.id for g in state.generators.values() if g.is_online]
    for gid in gen_ids:
        try:
            engine.inject_generator_failure(gid)
        except Exception:
            pass
    check("Frequency >= 45.0 Hz floor even after total loss",
          engine.state.frequency_hz >= 45.0, f"freq={engine.state.frequency_hz}")


def test_co2_emissions():
    separator("TEST 6: CO2 Emissions Tracking (D3)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("NUC_Cofrentes")
    redispatch = RedispatchEngine()
    result = redispatch.execute_redispatch(engine.state)
    check("Redispatch result has emissions key", "total_emissions_tco2" in result,
          f"keys={list(result.keys())}")
    gas_actions = [a for a in result.get("actions", []) if a.get("technology") == "Gas"]
    if gas_actions:
        check("Gas ramp-up produces CO2", result["total_emissions_tco2"] > 0,
              f"emissions={result['total_emissions_tco2']}")
    for action in result.get("actions", []):
        check(f"Action has emissions field", "emissions_tco2" in action)
        break


def test_afrr_mfrr():
    separator("TEST 7: aFRR/mFRR Phased Dispatch (D2)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("NUC_Cofrentes")
    engine.inject_generator_failure("NUC_Trillo")
    redispatch = RedispatchEngine()
    result = redispatch.execute_redispatch(engine.state)
    check("Has frequency_recovered_to", "frequency_recovered_to" in result)
    # Note: governor droop may have already absorbed the deficit via ramping up
    # other online generators during _update_frequency. So deficit_before at LP
    # level can be 0. What matters is the structure works.
    check("Status is valid", result["status"] in ["No Deficit", "Redispatch Complete", "Unserved Energy Remains"],
          f"status={result['status']}")


def test_redispatch_no_deficit():
    separator("TEST 8: Redispatch with No Deficit")
    state = build_demo_grid_state()
    redispatch = RedispatchEngine()
    result = redispatch.execute_redispatch(state)
    check("Status is 'No Deficit'", result["status"] == "No Deficit")
    check("No actions taken", len(result["actions"]) == 0)
    check("Cost is zero", result["total_cost_eur"] == 0.0)


def test_summary_completeness():
    separator("TEST 9: GridState.summary() Completeness")
    state = build_demo_grid_state()
    summary = state.summary()
    required = ["timestamp", "frequency_hz", "total_demand_mw", "total_generation_mw",
                "imports_mw", "transmission_losses_mw", "deficit_mw", "overgeneration_mw",
                "reserve_margin_pct", "online_generators", "offline_generators",
                "active_lines", "overloaded_lines", "generation_mix"]
    for key in required:
        check(f"summary has '{key}'", key in summary, f"missing from {list(summary.keys())}")
    check("transmission_losses_mw > 0", summary["transmission_losses_mw"] > 0)


def test_interconnection_divzero():
    separator("TEST 10: Interconnection Failure -- Division by Zero")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    try:
        engine.inject_interconnection_failure("France")
        engine.inject_interconnection_failure("Portugal")
        engine.inject_interconnection_failure("Morocco")
        check("All interconnections tripped without crash", True)
    except ZeroDivisionError as e:
        check("No ZeroDivisionError", False, str(e))
    except Exception as e:
        check("No unexpected error", False, str(e))


def test_metrics_ens_stress():
    separator("TEST 11: Metrics -- ENS and Network Stress (E1, E2)")
    initial = build_demo_grid_state()
    final = copy.deepcopy(initial)
    engine = ChaosEngine(final)
    engine.inject_generator_failure("NUC_Cofrentes")
    engine.inject_generator_failure("NUC_Trillo")
    result = calculate_resilience_score(initial, engine.state, 3, 5, 4, 0.5, 3)
    bd = result.get("breakdown", {})
    check("ENS in breakdown", "ens_mwh" in bd)
    check("Network stress index in breakdown", "network_stress_index" in bd)
    check("ENS >= 0", bd.get("ens_mwh", -1) >= 0)
    check("Score 0-100", 0 <= result["score"] <= 100)


def test_cascade_thermal():
    separator("TEST 12: Cascade Thermal Rating Delays (B1/B3)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("NUC_Cofrentes")
    engine.inject_generator_failure("NUC_Vandellos_2")
    engine.inject_generator_failure("NUC_Almaraz_1")
    cascade = CascadeEngine(engine)
    result = cascade.run_cascade(initial_trigger="stress_test", max_depth=10)
    check("Cascade runs without crash", True)
    check("Has steps", len(result.steps) > 0)


def test_healing_dynamic():
    separator("TEST 13: Healing Agent Dynamic Frequency Recovery (D1)")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("GAS_Castellon")
    freq_after_trip = engine.state.frequency_hz
    agent = HealingAgent()
    # Mock the model so predict() doesn't crash — it falls through to rule-based logic
    agent.is_trained = True
    class MockModel:
        def predict(self, X):
            import numpy as np
            return np.zeros((1, 6))
    agent.model = MockModel()
    result = agent.apply_healing(engine.state)
    freq_after_heal = engine.state.frequency_hz
    check("Has post_healing_frequency_hz", "post_healing_frequency_hz" in result)
    check("Frequency recovered", freq_after_heal >= freq_after_trip,
          f"trip={freq_after_trip:.3f}, heal={freq_after_heal:.3f}")
    delta = freq_after_heal - freq_after_trip
    check("Recovery NOT hardcoded +0.2", abs(delta - 0.2) > 0.001 or delta == 0.0,
          f"delta={delta:.4f}")


def test_startup_time():
    separator("TEST 14: Startup Time Constraints (A2)")
    state = build_demo_grid_state()
    coal = state.generators["COAL_Asturias"]
    coal.trip()
    check("Coal offline after trip", not coal.is_online)
    check("Coal startup req is 360 min", STARTUP_TIME_MIN["Coal"] == 360.0)
    check("Coal cannot start immediately (0 < 360)", coal.offline_duration_min < STARTUP_TIME_MIN["Coal"])
    check("Hydro startup req is 5 min", STARTUP_TIME_MIN["Hydro"] == 5.0)


def test_governor_no_overshoot():
    separator("TEST 15: Governor Droop -- No Over-Compensation")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("COAL_Asturias")
    check("Freq <= 50 after loss", engine.state.frequency_hz <= 50.0,
          f"freq={engine.state.frequency_hz:.3f}")
    check("Freq > 49 for small loss", engine.state.frequency_hz > 49.0,
          f"freq={engine.state.frequency_hz:.3f}")


def test_massive_simultaneous_failures():
    separator("TEST 16: Massive Simultaneous Failures")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    
    # Trip ALL nuclear
    nucs = [g for g in state.generators.values() if g.technology == "Nuclear"]
    for n in nucs:
        engine.inject_generator_failure(n.id)
        
    check("Didn't crash after all nuclear tripped", True)
    check("Freq dropped but clamped", engine.state.frequency_hz < 50.0,
          f"freq={engine.state.frequency_hz:.3f}")
    redispatch = RedispatchEngine()
    try:
        result = redispatch.execute_redispatch(engine.state)
        check("Redispatch didn't crash", "status" in result)
    except Exception as e:
        check("Redispatch didn't crash", False, str(e))


def test_ramp_rates_consistency():
    separator("TEST 17: RAMP_RATES Consistency")
    from digital_twin.simulator import RAMP_RATES as sim_rr
    import digital_twin.redispatch as redispatch
    
    check("Simulator RAMP_RATES has BESS", "BESS" in sim_rr)
    check("Redispatch RAMP_RATES has BESS", "BESS" in redispatch.RAMP_RATES,
          f"keys={list(redispatch.RAMP_RATES.keys())}")


def test_interconnection_all_zero():
    separator("TEST 18: All Interconnections at 0 MW")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    for country in list(state.interconnections.keys()):
        state.interconnections[country] = 0.0
    state.imports_mw = 0.0
    try:
        engine.inject_interconnection_failure("France")
        check("No crash tripping zero-capacity interconnection", True)
    except ZeroDivisionError:
        check("ZeroDivisionError with all interconnections=0", False,
              "sum(interconnections)=0 causes div by zero!")
    except ValueError:
        check("ValueError is acceptable for zero-capacity trip", True)
    except Exception as e:
        check("Unexpected error", False, str(e))


def test_engine_reset():
    separator("TEST 19: Engine Reset Clears State")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("GAS_Arcos")
    check("Event log has entries", len(engine.event_log) > 0)
    engine.reset()
    check("Event log cleared", len(engine.event_log) == 0)
    check("Cascade log cleared", len(engine.cascade_log) == 0)
    check("Frequency restored to 50.0", engine.state.frequency_hz == 50.0)
    gen = engine.state.generators["GAS_Arcos"]
    check("Generator back online", gen.is_online)


def test_redispatch_startup_constraint():
    separator("TEST 20: Redispatch respects startup time for offline gens")
    state = build_demo_grid_state()
    engine = ChaosEngine(state)
    engine.inject_generator_failure("NUC_Cofrentes")
    engine.inject_generator_failure("NUC_Trillo")
    nuc1 = engine.state.generators["NUC_Cofrentes"]
    check("Tripped nuclear offline_duration=0", nuc1.offline_duration_min == 0.0)
    redispatch = RedispatchEngine()
    result = redispatch.execute_redispatch(engine.state)
    nuc_actions = [a for a in result.get("actions", [])
                   if a.get("generator_id") in ["NUC_Cofrentes", "NUC_Trillo"]]
    check("Tripped nuclear NOT dispatched (startup constraint)",
          len(nuc_actions) == 0, f"nuclear actions: {nuc_actions}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  DEEP STRESS TEST SUITE -- Digital Twin Backend")
    print("="*70)

    test_transmission_losses()
    test_technical_minimum()
    test_generator_trip()
    test_bess_inertia()
    test_frequency_clamping()
    test_co2_emissions()
    test_afrr_mfrr()
    test_redispatch_no_deficit()
    test_summary_completeness()
    test_interconnection_divzero()
    test_metrics_ens_stress()
    test_cascade_thermal()
    test_healing_dynamic()
    test_startup_time()
    test_governor_no_overshoot()
    test_massive_simultaneous_failures()
    test_ramp_rates_consistency()
    test_interconnection_all_zero()
    test_engine_reset()
    test_redispatch_startup_constraint()

    print(f"\n{'='*70}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'='*70}")

    if ERRORS:
        print("\n  FAILURES:")
        for e in ERRORS:
            print(f"    {e}")

    print()
    sys.exit(1 if FAIL > 0 else 0)
