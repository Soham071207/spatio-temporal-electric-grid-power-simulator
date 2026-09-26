"""
Scenario Engine — Autonomously generates realistic failure scenarios
and runs them through the Chaos Engine to produce training data.

Scenario types:
  - single_gen_failure: Trip one plant
  - n_of_k_failure: Trip N random plants simultaneously
  - renewable_collapse: Wind or Solar drops 30-80%
  - demand_spike: One region demand surges 10-40%
  - storm_regional: All plants in one region fail + transmission cuts
  - compound: Combination of 2+ above
  - heatwave: Demand +15% everywhere + thermal derate 10%

Each scenario produces a ScenarioResult with before/after state,
resilience scores, and optimal redispatch actions.
"""

import copy
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .simulator import GridState, ChaosEngine, Generator
from .redispatch import RedispatchEngine
from .metrics import calculate_resilience_score
from .cascade import CascadeEngine


@dataclass
class ScenarioResult:
    """Complete result of one chaos scenario simulation."""
    scenario_id: str
    scenario_type: str
    description: str
    parameters: dict
    # State summaries
    initial_state_summary: dict = field(default_factory=dict)
    post_fault_summary: dict = field(default_factory=dict)
    post_healing_summary: dict = field(default_factory=dict)
    # Scores
    resilience_score_before: float = 100.0
    resilience_score_after_fault: float = 100.0
    resilience_score_after_healing: float = 100.0
    # Healing details
    healing_actions: list = field(default_factory=list)
    healing_cost_eur: float = 0.0
    # Cascade details
    cascade_depth: int = 0
    cascade_size: int = 0
    geographic_spread: int = 0
    # Impact metrics
    generation_lost_mw: float = 0.0
    unserved_energy_mw: float = 0.0
    frequency_deviation_hz: float = 0.0
    # Training features (flattened for ML)
    # Training features (flattened for ML)
    feature_vector: list = field(default_factory=list)
    optimal_action_vector: list = field(default_factory=list)
    # Timeline for simulation
    cascade_history: list = field(default_factory=list)


class ScenarioEngine:
    """
    Generates and executes chaos scenarios on a GridState.
    
    Uses the existing ChaosEngine for fault injection,
    CascadeEngine for propagation, and RedispatchEngine for healing.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.redispatch = RedispatchEngine()

    def run_scenario(
        self,
        initial_state: GridState,
        scenario_type: str,
        params: Optional[dict] = None,
    ) -> ScenarioResult:
        """
        Run a single scenario on the given grid state.
        
        Args:
            initial_state: The pre-fault grid state (will not be modified).
            scenario_type: One of the supported scenario types.
            params: Optional parameters for the scenario.
            
        Returns:
            ScenarioResult with full before/after analysis.
        """
        params = params or {}
        state_copy = copy.deepcopy(initial_state)
        engine = ChaosEngine(state_copy)

        # Record initial state
        initial_summary = engine.state.summary()
        initial_gen = engine.state.total_generation_mw

        # Generate a unique scenario ID
        scenario_id = hashlib.md5(
            f"{scenario_type}_{params}_{initial_state.timestamp}".encode()
        ).hexdigest()[:12]

        # ── Inject the fault ──
        description = self._inject_scenario(engine, scenario_type, params)

        # ── Run cascade propagation ──
        cascade_engine = CascadeEngine(engine)
        cascade_result = cascade_engine.run_cascade(
            initial_trigger=description,
            max_depth=8,
        )

        # Record post-fault state
        post_fault_summary = engine.state.summary()
        generation_lost = max(0, initial_gen - engine.state.total_generation_mw)

        # ── Calculate POST-FAULT resilience (before any healing) ──
        resilience_after_fault = calculate_resilience_score(
            initial_state, engine.state,
            cascade_depth=cascade_result.cascade_depth,
            cascade_size=cascade_result.cascade_size,
            geographic_spread=cascade_result.geographic_spread,
        )

        # ── Apply redispatch healing ──
        healing_result = self.redispatch.execute_redispatch(engine.state)

        # Record post-healing state
        post_healing_summary = engine.state.summary()

        # ── Calculate POST-HEALING resilience ──
        resilience_after_healing = calculate_resilience_score(
            initial_state, engine.state,
            cascade_depth=cascade_result.cascade_depth,
            cascade_size=cascade_result.cascade_size,
            geographic_spread=cascade_result.geographic_spread,
        )

        # ── Calculate INITIAL resilience (pre-fault baseline) ──
        resilience_before = calculate_resilience_score(initial_state, initial_state)

        # ── Build training vectors ──
        feature_vec = self._build_feature_vector(initial_state, post_fault_summary)
        action_vec = self._build_action_vector(healing_result)

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            description=description,
            parameters=params,
            initial_state_summary=initial_summary,
            post_fault_summary=post_fault_summary,
            post_healing_summary=post_healing_summary,
            resilience_score_before=resilience_before["score"],
            resilience_score_after_fault=resilience_after_fault["score"],
            resilience_score_after_healing=resilience_after_healing["score"],
            healing_actions=healing_result.get("actions", []),
            healing_cost_eur=healing_result.get("total_cost_eur", 0),
            cascade_depth=cascade_result.cascade_depth,
            cascade_size=cascade_result.cascade_size,
            geographic_spread=cascade_result.geographic_spread,
            generation_lost_mw=round(generation_lost, 1),
            unserved_energy_mw=round(healing_result.get("deficit_after_mw", 0), 1),
            frequency_deviation_hz=round(abs(50.0 - engine.state.frequency_hz), 3),
            feature_vector=feature_vec,
            optimal_action_vector=action_vec,
            cascade_history=cascade_engine.to_dict(cascade_result).get("steps", []),
        )

    def generate_random_scenarios(
        self,
        initial_state: GridState,
        n_scenarios: int = 100,
    ) -> List[ScenarioResult]:
        """Generate N random scenarios of mixed types."""
        results = []
        scenario_types = [
            "single_gen_failure",
            "n_of_k_failure",
            "renewable_collapse",
            "demand_spike",
            "storm_regional",
            "compound",
            "heatwave",
        ]
        weights = [30, 20, 15, 10, 10, 10, 5]  # Probability weights

        for i in range(n_scenarios):
            stype = self.rng.choices(scenario_types, weights=weights, k=1)[0]
            try:
                result = self.run_scenario(initial_state, stype)
                results.append(result)
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{n_scenarios}] Generated {stype} → "
                          f"Resilience: {result.resilience_score_after_fault} → "
                          f"{result.resilience_score_after_healing}")
            except Exception as e:
                # Skip scenarios that fail (e.g., no valid targets)
                continue

        return results

    # ────────────────────────────────────────────────
    #  SCENARIO INJECTION METHODS
    # ────────────────────────────────────────────────

    def _inject_scenario(self, engine: ChaosEngine, stype: str, params: dict) -> str:
        """Dispatch to the appropriate injection method."""
        if stype == "single_gen_failure":
            return self._inject_single_gen(engine, params)
        elif stype == "n_of_k_failure":
            return self._inject_n_of_k(engine, params)
        elif stype == "renewable_collapse":
            return self._inject_renewable_collapse(engine, params)
        elif stype == "demand_spike":
            return self._inject_demand_spike(engine, params)
        elif stype == "storm_regional":
            return self._inject_storm(engine, params)
        elif stype == "compound":
            return self._inject_compound(engine, params)
        elif stype == "heatwave":
            return self._inject_heatwave(engine, params)
        else:
            raise ValueError(f"Unknown scenario type: {stype}")

    def _inject_single_gen(self, engine: ChaosEngine, params: dict) -> str:
        online = [g for g in engine.state.generators.values() if g.is_online]
        if not online:
            raise ValueError("No online generators to fail")
        target = params.get("generator_id")
        if target:
            gen = engine.state.generators.get(target)
            if not gen or not gen.is_online:
                target = None
        if not target:
            gen = self.rng.choice(online)
            target = gen.id
        event = engine.inject_generator_failure(target)
        return event.description

    def _inject_n_of_k(self, engine: ChaosEngine, params: dict) -> str:
        n = params.get("n", self.rng.randint(2, 5))
        online = [g for g in engine.state.generators.values() if g.is_online]
        n = min(n, len(online))
        targets = self.rng.sample(online, n)
        descs = []
        for gen in targets:
            event = engine.inject_generator_failure(gen.id)
            descs.append(f"{gen.id} ({gen.technology})")
        return f"Multi-failure: {n} plants tripped — {', '.join(descs)}"

    def _inject_renewable_collapse(self, engine: ChaosEngine, params: dict) -> str:
        fuel = params.get("fuel_type", self.rng.choice(["Wind", "Solar"]))
        pct = params.get("reduction_pct", self.rng.uniform(30, 80))
        event = engine.inject_renewable_shock(fuel, pct)
        return event.description

    def _inject_demand_spike(self, engine: ChaosEngine, params: dict) -> str:
        regions = list(engine.state.regional_demand_mw.keys())
        if not regions:
            raise ValueError("No regions with demand data")
        region = params.get("region", self.rng.choice(regions))
        pct = params.get("increase_pct", self.rng.uniform(10, 40))
        event = engine.inject_demand_surge(region, pct)
        return event.description

    def _inject_storm(self, engine: ChaosEngine, params: dict) -> str:
        # Find regions that have generators
        gen_regions = set(g.region for g in engine.state.generators.values() if g.is_online)
        if not gen_regions:
            raise ValueError("No regions with online generators")
        region = params.get("region", self.rng.choice(list(gen_regions)))
        events = engine.run_storm_mode(region, n_failures=self.rng.randint(2, 5))
        total_impact = sum(e.impact_mw for e in events)
        return f"Storm in {region}: {len(events)} components failed, {total_impact:.0f} MW lost"

    def _inject_compound(self, engine: ChaosEngine, params: dict) -> str:
        """Combine 2-3 random scenario types."""
        sub_types = self.rng.sample(
            ["single_gen_failure", "renewable_collapse", "demand_spike"],
            k=self.rng.randint(2, 3),
        )
        descs = []
        for st in sub_types:
            try:
                desc = self._inject_scenario(engine, st, {})
                descs.append(desc)
            except (ValueError, Exception):
                continue
        return "COMPOUND: " + " + ".join(descs) if descs else "Compound (no valid targets)"

    def _inject_heatwave(self, engine: ChaosEngine, params: dict) -> str:
        # Demand surge across all regions
        surge_pct = params.get("demand_increase_pct", 15.0)
        for region in list(engine.state.regional_demand_mw.keys()):
            engine.inject_demand_surge(region, surge_pct)
        # Thermal derate — reduce gas/coal output by 10%
        derate_pct = params.get("thermal_derate_pct", 10.0)
        for gen in engine.state.generators.values():
            if gen.is_online and gen.technology in ("Gas", "Coal"):
                reduction = gen.current_output_mw * (derate_pct / 100.0)
                gen.current_output_mw = max(0, gen.current_output_mw - reduction)
        # Fix 5A: Heatwave also reduces hydro (low water) and solar (thermal derate)
        hydro_derate = params.get("hydro_derate_pct", 20.0)
        solar_derate = params.get("solar_derate_pct", 5.0)
        for gen in engine.state.generators.values():
            if gen.is_online and gen.technology == "Hydro":
                reduction = gen.current_output_mw * (hydro_derate / 100.0)
                gen.current_output_mw = max(0, gen.current_output_mw - reduction)
            elif gen.is_online and gen.technology == "Solar":
                reduction = gen.current_output_mw * (solar_derate / 100.0)
                gen.current_output_mw = max(0, gen.current_output_mw - reduction)
        return (f"Heatwave: demand +{surge_pct}% all regions, thermal derated {derate_pct}%, "
                f"hydro derated {hydro_derate}%, solar derated {solar_derate}%")

    # ────────────────────────────────────────────────
    #  FEATURE EXTRACTION (for ML Healing Agent)
    # ────────────────────────────────────────────────

    def _build_feature_vector(self, initial_state: GridState, post_fault: dict) -> list:
        """
        Extract a flat feature vector from the post-fault state
        that the Healing Agent can use as input.
        """
        total_demand = post_fault.get("total_demand_mw", 0)
        total_gen = post_fault.get("total_generation_mw", 0)
        deficit = post_fault.get("deficit_mw", 0)
        freq = post_fault.get("frequency_hz", 50.0)
        n_offline = post_fault.get("offline_generators", 0)
        n_overloaded = post_fault.get("overloaded_lines", 0)
        reserve = post_fault.get("reserve_margin_pct", 0)
        imports = post_fault.get("imports_mw", 0)

        # Generation mix
        mix = post_fault.get("generation_mix", {})
        gen_hydro = mix.get("Hydro", 0)
        gen_gas = mix.get("Gas", 0)
        gen_coal = mix.get("Coal", 0)
        gen_nuclear = mix.get("Nuclear", 0)
        gen_solar = mix.get("Solar", 0)
        gen_wind = mix.get("Wind", 0)

        # Deficit ratio
        deficit_ratio = deficit / max(total_demand, 1.0)
        freq_dev = abs(50.0 - freq)

        return [
            total_demand, total_gen, deficit, deficit_ratio,
            freq, freq_dev,
            n_offline, n_overloaded, reserve, imports,
            gen_hydro, gen_gas, gen_coal, gen_nuclear, gen_solar, gen_wind,
        ]

    def _build_action_vector(self, healing_result: dict) -> list:
        """
        Extract the optimal action vector from the redispatch result.
        This is the 'ground truth' for training the Healing Agent.
        """
        actions = healing_result.get("actions", [])

        ramp_hydro = 0.0
        ramp_gas = 0.0
        ramp_coal = 0.0
        ramp_nuclear = 0.0  # Fix 5B: Include nuclear
        pull_imports = 0.0

        for action in actions:
            if isinstance(action, dict):
                tech = action.get("technology", "")
                ramped = action.get("ramped_mw", 0)
                if tech == "Hydro":
                    ramp_hydro += ramped
                elif tech == "Gas":
                    ramp_gas += ramped
                elif tech == "Coal":
                    ramp_coal += ramped
                elif tech == "Nuclear":
                    ramp_nuclear += ramped
                elif action.get("type") == "emergency_import":
                    pull_imports += action.get("imported_mw", 0)

        total_cost = healing_result.get("total_cost_eur", 0)
        remaining_deficit = healing_result.get("deficit_after_mw", 0)

        return [ramp_hydro, ramp_gas, ramp_coal, ramp_nuclear, pull_imports, remaining_deficit, total_cost]


def results_to_dataframe(results: List[ScenarioResult]):
    """Convert a list of ScenarioResults to a pandas DataFrame for analysis/training."""
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "scenario_id": r.scenario_id,
            "scenario_type": r.scenario_type,
            "description": r.description,
            "resilience_before": r.resilience_score_before,
            "resilience_after_fault": r.resilience_score_after_fault,
            "resilience_after_healing": r.resilience_score_after_healing,
            "generation_lost_mw": r.generation_lost_mw,
            "unserved_energy_mw": r.unserved_energy_mw,
            "healing_cost_eur": r.healing_cost_eur,
            "cascade_depth": r.cascade_depth,
            "cascade_size": r.cascade_size,
            "geographic_spread": r.geographic_spread,
            "frequency_deviation_hz": r.frequency_deviation_hz,
        }
        # Flatten feature vector
        feat_names = [
            "f_demand", "f_gen", "f_deficit", "f_deficit_ratio",
            "f_freq", "f_freq_dev",
            "f_offline_gens", "f_overloaded_lines", "f_reserve", "f_imports",
            "f_hydro", "f_gas", "f_coal", "f_nuclear", "f_solar", "f_wind",
        ]
        for i, name in enumerate(feat_names):
            row[name] = r.feature_vector[i] if i < len(r.feature_vector) else 0

        # Flatten action vector
        act_names = [
            "a_ramp_hydro", "a_ramp_gas", "a_ramp_coal", "a_ramp_nuclear",
            "a_pull_imports", "a_remaining_deficit", "a_cost",
        ]
        for i, name in enumerate(act_names):
            row[name] = r.optimal_action_vector[i] if i < len(r.optimal_action_vector) else 0

        rows.append(row)

    return pd.DataFrame(rows)
