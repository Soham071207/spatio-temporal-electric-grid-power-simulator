"""
Redispatch Engine — Acts as the grid operator.

When a deficit occurs after chaos injection, the engine logically brings
reserve capacity online to re-balance the grid, following a merit order:
  1. Hydro (fastest ramp)
  2. Gas / CCGT
  3. Coal
  4. Emergency imports (France, Portugal, Morocco)

Tracks redispatch costs and interconnection import limits.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linprog
from .simulator import GridState, Generator, HEAT_RATE_BANDS, COST_OF_UNSERVED_ENERGY


# Approximate marginal cost per MWh by technology (EUR/MWh)
TECH_COSTS = {
    "Hydro": 5.0,
    "Nuclear": 12.0,
    "Wind": 0.0,
    "Solar": 0.0,
    "Gas": 65.0,
    "Coal": 85.0,
    "Import": 90.0,
    "BESS": 10.0,  # Degradation cost mostly
}

# Fix D3: CO2 emissions tracking (tons CO2 / MWh)
TECH_EMISSIONS = {
    "Hydro": 0.0,
    "Nuclear": 0.0,
    "Wind": 0.0,
    "Solar": 0.0,
    "Gas": 0.37,
    "Coal": 0.90,
    "Import": 0.30,  # Average European mix
    "BESS": 0.0,
}

# Fix 2B: Per-country import costs (EUR/MWh)
IMPORT_COSTS = {
    "France": 55.0,      # Cheap nuclear baseload
    "Portugal": 45.0,    # Hydro-heavy
    "Morocco": 110.0,    # More expensive
    "Andorra": 70.0,
}

# Interconnection import limits (MW) per border
INTERCONNECTION_LIMITS = {
    "France": 2800.0,
    "Portugal": 3100.0,
    "Morocco": 900.0,
    "Andorra": 50.0,
}

# Fix 2A: Ramp rates (MW/min) by technology
RAMP_RATES = {
    "Nuclear": 5.0,
    "Coal": 8.0,
    "Gas": 20.0,
    "Hydro": 50.0,
    "Wind": 0.0,
    "Solar": 0.0,
    "Import": 100.0,
    "BESS": 1000.0,
}

# Dispatch window in minutes (how much time we have to ramp)
DISPATCH_WINDOW_MIN = 15.0


class RedispatchEngine:
    """
    Acts as the grid operator. When a deficit occurs, it logically brings
    reserve capacity online to re-balance the grid.
    """

    def __init__(self, dispatch_order: list = None):
        pass

    def execute_redispatch(self, state: GridState) -> dict:
        """
        Executes a two-phase Redispatch to heal the grid:
        Phase 1: aFRR (Automatic Frequency Restoration Reserve) - ultra fast (BESS, Hydro, partial Gas)
        Phase 2: mFRR (Manual Frequency Restoration Reserve) - full fleet optimization
        """
        if state.deficit_mw <= 0.1:
            return {
                "status": "No Deficit",
                "total_cost_eur": 0.0,
                "total_emissions_tco2": 0.0,
                "deficit_before_mw": 0.0,
                "deficit_after_mw": 0.0,
                "frequency_recovered_to": state.frequency_hz,
                "actions": []
            }

        initial_deficit = state.deficit_mw
        all_actions = []
        total_cost = 0.0
        total_emissions = 0.0

        # Phase 1: aFRR (Fast reserves)
        # BESS, Hydro (up to 10% of capacity), Gas (up to 5% of capacity)
        afrr_limits = {"BESS": 1.0, "Hydro": 0.10, "Gas": 0.05}
        res_afrr = self._run_lp_phase(state, state.deficit_mw, afrr_limits, allow_imports=False)
        all_actions.extend(res_afrr["actions"])
        total_cost += res_afrr["cost"]
        total_emissions += res_afrr["emissions"]

        # Phase 2: mFRR (Slow reserves)
        # All remaining capacities + imports
        current_deficit = res_afrr["remaining_deficit"]
        if current_deficit > 0.1:
            mfrr_limits = {k: 1.0 for k in TECH_COSTS.keys()} # 100% of remaining available
            res_mfrr = self._run_lp_phase(state, current_deficit, mfrr_limits, allow_imports=True)
            all_actions.extend(res_mfrr["actions"])
            total_cost += res_mfrr["cost"]
            total_emissions += res_mfrr["emissions"]
            final_deficit = res_mfrr["remaining_deficit"]
        else:
            final_deficit = 0.0

        # Frequency recovery logic
        total_ramped = sum(a.get("ramped_mw", 0) for a in all_actions if a.get("type") == "ramp_up") + \
                       sum(a.get("imported_mw", 0) for a in all_actions if a.get("type") == "emergency_import")
        
        if total_ramped > 0 and state.frequency_hz < 50.0:
            from .simulator import INERTIA_CONSTANTS
            H_sys_total = 0.0
            S_sys_total = 0.0
            for g in state.generators.values():
                if g.is_online and g.technology in INERTIA_CONSTANTS:
                    h = INERTIA_CONSTANTS[g.technology]
                    if h > 0:
                        H_sys_total += h * g.installed_capacity_mw
                        S_sys_total += g.installed_capacity_mw
            # Add synthetic inertia
            bess_cap = sum(g.installed_capacity_mw for g in state.generators.values() if g.is_online and g.technology == "BESS")
            if bess_cap > 0:
                H_sys_total += 3.0 * bess_cap
                S_sys_total += bess_cap
                
            H_sys = H_sys_total / max(S_sys_total, 1.0)
            if S_sys_total > 0 and H_sys > 0:
                recovery_f = (total_ramped / (2 * H_sys * S_sys_total)) * 50.0
                state.frequency_hz = min(50.0, state.frequency_hz + recovery_f)

        return {
            "status": "Redispatch Complete" if final_deficit <= 0.1 else "Unserved Energy Remains",
            "total_cost_eur": round(total_cost, 2),
            "total_emissions_tco2": round(total_emissions, 2),
            "deficit_before_mw": round(initial_deficit, 1),
            "deficit_after_mw": round(max(0, final_deficit), 1),
            "frequency_recovered_to": round(state.frequency_hz, 3),
            "actions": all_actions
        }

    def _run_lp_phase(self, state: GridState, target_deficit: float, tech_limits: dict, allow_imports: bool) -> dict:
        """Runs a single phase of LP optimization."""
        available_gens = []
        for g in state.generators.values():
            if g.technology in tech_limits:
                limit_pct = tech_limits[g.technology]
                if limit_pct > 0:
                    available_gens.append((g, limit_pct))

        available_imports = []
        if allow_imports:
            for country, capacity in state.interconnections.items():
                available_capacity = max(0.0, capacity * 0.9) 
                if available_capacity > 0:
                    cost = IMPORT_COSTS.get(country, TECH_COSTS.get("Import", 90.0))
                    available_imports.append({
                        "country": country,
                        "capacity": available_capacity,
                        "cost": cost
                    })

        n_gens = len(available_gens)
        n_imports = len(available_imports)
        n_vars = n_gens + n_imports + 1
        
        c = np.zeros(n_vars)
        bounds = []

        for i, (gen, limit_pct) in enumerate(available_gens):
            base_cost = TECH_COSTS.get(gen.technology, 50.0)
            size_penalty = 50.0 / (gen.installed_capacity_mw + 1.0)
            ramp_penalty = 10.0 * (1.0 - gen.load_factor)
            
            # Fix #2: Heat-rate penalty based on current load factor
            # Thermal plants are less efficient at very low or very high load
            lf = gen.load_factor
            heat_rate_mult = 1.15  # default: high-load or unknown band
            for max_lf, mult in HEAT_RATE_BANDS:
                if lf <= max_lf:
                    heat_rate_mult = mult
                    break
            
            c[i] = (base_cost * heat_rate_mult) + size_penalty + ramp_penalty
            
            from .simulator import STARTUP_TIME_MIN
            ramp_rate = RAMP_RATES.get(gen.technology, 10.0)
            max_ramp_mw = ramp_rate * DISPATCH_WINDOW_MIN
            
            # Phase-specific capacity cap
            phase_cap = gen.installed_capacity_mw * limit_pct
            
            if gen.is_online:
                available_for_phase = min(gen.available_reserve_mw, phase_cap)
                upper_bound = min(available_for_phase, max_ramp_mw)
                bounds.append((0.0, upper_bound))
            else:
                startup_req = STARTUP_TIME_MIN.get(gen.technology, 9999.0)
                if gen.offline_duration_min >= startup_req:
                    upper_bound = min(phase_cap, max_ramp_mw)
                    bounds.append((0.0, upper_bound))
                else:
                    bounds.append((0.0, 0.0))

        for i, imp in enumerate(available_imports):
            idx = n_gens + i
            c[idx] = imp["cost"]
            bounds.append((0.0, imp["capacity"]))

        slack_idx = n_gens + n_imports
        # Fix #8: Load shedding is the last resort.
        # The LP slack variable now represents deliberately unserved load (load-shed),
        # costed at the regulatory Value of Lost Load (VoLL = €50,000/MWh).
        # This ensures load shedding only occurs when all other options are exhausted.
        c[slack_idx] = COST_OF_UNSERVED_ENERGY
        bounds.append((0.0, None))

        A_eq = np.ones((1, n_vars))
        b_eq = np.array([target_deficit])

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        actions_taken = []
        phase_cost = 0.0
        phase_emissions = 0.0
        remaining_deficit = target_deficit

        if res.success:
            x = res.x
            for i, (gen, _) in enumerate(available_gens):
                ramp = float(x[i])
                if ramp > 0.1:
                    if not gen.is_online:
                        gen.is_online = True
                        gen.current_output_mw = 0.0
                    gen.current_output_mw += ramp
                    
                    cost = float(c[i] * ramp)
                    emissions = float(TECH_EMISSIONS.get(gen.technology, 0.0) * ramp)
                    phase_cost += cost
                    phase_emissions += emissions
                    
                    actions_taken.append({
                        "type": "ramp_up",
                        "generator_id": gen.id,
                        "technology": gen.technology,
                        "region": gen.region,
                        "ramped_mw": round(ramp, 1),
                        "new_output_mw": round(gen.current_output_mw, 1),
                        "cost_eur": round(cost, 2),
                        "emissions_tco2": round(emissions, 2)
                    })

            for i, imp in enumerate(available_imports):
                idx = n_gens + i
                pull = float(x[idx])
                if pull > 0.1:
                    state.imports_mw += pull
                    cost = float(c[idx] * pull)
                    emissions = float(TECH_EMISSIONS.get("Import", 0.3) * pull)
                    phase_cost += cost
                    phase_emissions += emissions
                    
                    actions_taken.append({
                        "type": "emergency_import",
                        "country": imp["country"],
                        "imported_mw": round(pull, 1),
                        "cost_eur": round(cost, 2),
                        "emissions_tco2": round(emissions, 2)
                    })
                    
            remaining_deficit = float(x[slack_idx])

        return {
            "actions": actions_taken,
            "cost": phase_cost,
            "emissions": phase_emissions,
            "remaining_deficit": remaining_deficit
        }

    def estimate_redispatch_cost(self, state: GridState) -> dict:
        """Estimate the cost to fully redispatch without modifying state."""
        import copy
        test_state = copy.deepcopy(state)
        result = self.execute_redispatch(test_state)
        return {
            "estimated_cost_eur": result["total_cost_eur"],
            "can_fully_serve": result["deficit_after_mw"] <= 0,
            "remaining_deficit_mw": result["deficit_after_mw"],
        }
