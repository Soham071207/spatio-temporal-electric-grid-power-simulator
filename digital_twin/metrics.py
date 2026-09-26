"""
Grid Resilience Scoring Engine.

Calculates a unified resilience score (0-100) based on:
  - Load served percentage
  - Line overloads
  - Frequency deviation
  - Cascade depth & size
  - Geographic impact
  - Reserve margin

Matches Section 6 of the chaos engineering design doc.
"""

from typing import Dict, Optional
from .simulator import GridState


def calculate_resilience_score(
    initial_state: GridState,
    final_state: GridState,
    cascade_depth: int = 0,
    cascade_size: int = 0,
    geographic_spread: int = 0,
    max_rocof_hz: float = 0.0,
    cascade_steps_to_stable: int = 0,
) -> Dict:
    """
    Calculates a resilience score from 0 to 100 with a detailed breakdown.

    100 = Perfect resilience (no impact whatsoever)
    0   = Total collapse (blackout)

    Returns a dict with the overall score and per-category breakdown.
    """
    score = 100.0
    breakdown = {}

    # ─── 1. Load Served Penalty (max -40 points) ───
    # If demand is unserved, this is the heaviest penalty
    if final_state.total_demand_mw > 0:
        served_mw = final_state.total_generation_mw + final_state.imports_mw
        pct_served = min(1.0, served_mw / final_state.total_demand_mw)
        pct_unserved = 1.0 - pct_served
        load_penalty = pct_unserved * 400  # 10% unserved = -40 points
        load_penalty = min(40.0, load_penalty)
    else:
        pct_served = 1.0
        pct_unserved = 0.0
        load_penalty = 0.0

    score -= load_penalty
    breakdown["load_served_pct"] = round(pct_served * 100, 1)
    breakdown["load_penalty"] = round(load_penalty, 1)

    # ─── 2. Line Overload Penalty (max -15 points) ───
    n_overloaded = len(final_state.overloaded_lines)
    total_active = sum(1 for l in final_state.lines.values() if l.is_active)
    if total_active > 0:
        overload_ratio = n_overloaded / total_active
        overload_penalty = min(15.0, overload_ratio * 150)
    else:
        overload_penalty = 15.0  # No active lines = maximum penalty

    score -= overload_penalty
    breakdown["overloaded_lines"] = n_overloaded
    breakdown["overload_penalty"] = round(overload_penalty, 1)

    # Fix E2: Network Stress Index (average loading of active lines)
    if total_active > 0:
        avg_loading = sum(l.loading_pct for l in final_state.lines.values() if l.is_active) / total_active
        network_stress_index = avg_loading / 100.0
        
        # Penalize if the overall network is highly stressed (>70% average loading)
        if network_stress_index > 0.7:
            stress_penalty = min(5.0, (network_stress_index - 0.7) * 16.6) # max 5 pts
            score -= stress_penalty
            breakdown["stress_penalty"] = round(stress_penalty, 1)
    else:
        network_stress_index = 1.0
        
    breakdown["network_stress_index"] = round(network_stress_index, 3)

    # ─── 3. Frequency Deviation Penalty (max -15 points) ───
    # Fix 4A: Continuous frequency scoring instead of step-function
    freq_deviation = abs(50.0 - final_state.frequency_hz)
    if freq_deviation <= 0.05:
        freq_penalty = 0.0       # Dead band — normal operating range
    else:
        freq_penalty = min(15.0, (freq_deviation / 1.5) * 15.0)

    score -= freq_penalty
    breakdown["frequency_hz"] = round(final_state.frequency_hz, 3)
    breakdown["frequency_deviation"] = round(freq_deviation, 3)
    breakdown["frequency_penalty"] = round(freq_penalty, 1)

    # ─── 3B. RoCoF Penalty (max -5 points) ───
    # Fix 4B: Rate of Change of Frequency — how fast frequency fell
    rocof_penalty = 0.0
    if max_rocof_hz > 0:
        rocof_penalty = min(5.0, (max_rocof_hz / 1.0) * 5.0)  # 1 Hz/step = max penalty
    score -= rocof_penalty
    breakdown["max_rocof_hz"] = round(max_rocof_hz, 3)
    breakdown["rocof_penalty"] = round(rocof_penalty, 1)

    # ─── 4. Cascade Penalty (max -15 points) ───
    cascade_penalty = 0.0
    if cascade_depth > 0:
        cascade_penalty += min(8.0, cascade_depth * 2.0)   # -2 per depth level
    if cascade_size > 0:
        cascade_penalty += min(7.0, cascade_size * 1.0)     # -1 per component lost

    score -= cascade_penalty
    breakdown["cascade_depth"] = cascade_depth
    breakdown["cascade_size"] = cascade_size
    breakdown["cascade_penalty"] = round(cascade_penalty, 1)

    # ─── 5. Geographic Impact Penalty (max -5 points) ───
    geo_penalty = min(5.0, geographic_spread * 1.0)  # -1 per region affected
    score -= geo_penalty
    breakdown["geographic_spread"] = geographic_spread
    breakdown["geographic_penalty"] = round(geo_penalty, 1)

    # ─── 6. Reserve Margin Penalty (max -10 points) ───
    reserve_margin = final_state.reserve_margin_pct
    if reserve_margin >= 15.0:
        reserve_penalty = 0.0
    elif reserve_margin >= 10.0:
        reserve_penalty = 3.0
    elif reserve_margin >= 5.0:
        reserve_penalty = 6.0
    else:
        reserve_penalty = 10.0

    score -= reserve_penalty
    breakdown["reserve_margin_pct"] = round(reserve_margin, 1)
    breakdown["reserve_penalty"] = round(reserve_penalty, 1)

    # ─── 7. Emergency Import Reliance Penalty (max -5 points) ───
    if final_state.imports_mw > initial_state.imports_mw:
        extra_imports = final_state.imports_mw - initial_state.imports_mw
        if final_state.total_demand_mw > 0:
            import_ratio = extra_imports / final_state.total_demand_mw
            import_penalty = min(5.0, import_ratio * 50)
        else:
            import_penalty = 0.0
    else:
        import_penalty = 0.0

    score -= import_penalty
    breakdown["extra_imports_mw"] = round(
        max(0, final_state.imports_mw - initial_state.imports_mw), 1
    )
    breakdown["import_penalty"] = round(import_penalty, 1)

    # ─── 8. Recovery Speed Bonus (up to +3 points) ───
    # Fix 4C: Reward fast stabilization
    recovery_bonus = 0.0
    if cascade_steps_to_stable > 0:
        if cascade_steps_to_stable <= 2:
            recovery_bonus = 3.0
        elif cascade_steps_to_stable <= 4:
            recovery_bonus = 1.5
    score += recovery_bonus
    breakdown["cascade_steps_to_stable"] = cascade_steps_to_stable
    breakdown["recovery_bonus"] = round(recovery_bonus, 1)

    # ─── Final Score ───
    final_score = int(max(0, min(100, score)))

    # Fix E1: Calculate Energy Not Served (ENS) in MWh
    # Assuming each cascade step or interval represents ~5 minutes (1/12th of an hour)
    ens_duration_hours = max(1, cascade_steps_to_stable) * (5.0 / 60.0)
    ens_mwh = final_state.deficit_mw * ens_duration_hours
    breakdown["ens_mwh"] = round(ens_mwh, 1)

    return {
        "score": final_score,
        "label": get_risk_label(final_score),
        "breakdown": breakdown,
    }


def get_risk_label(score: int) -> str:
    """Maps a resilience score to a human-readable risk label."""
    if score >= 80:
        return "LOW RISK (Normal)"
    elif score >= 60:
        return "MODERATE RISK"
    elif score >= 40:
        return "HIGH RISK"
    elif score >= 20:
        return "VERY HIGH RISK"
    else:
        return "CRITICAL (Blackout Imminent)"


def get_risk_color(score: int) -> str:
    """Maps a resilience score to a color for the UI."""
    if score >= 80:
        return "#22c55e"   # Green
    elif score >= 60:
        return "#eab308"   # Yellow
    elif score >= 40:
        return "#f97316"   # Orange
    elif score >= 20:
        return "#ef4444"   # Red
    else:
        return "#7f1d1d"   # Dark red
