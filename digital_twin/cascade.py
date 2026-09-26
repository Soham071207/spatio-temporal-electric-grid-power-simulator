"""
Cascading Failure Engine — Dedicated cascade propagation logic.

Tracks overloaded lines after each fault, auto-trips lines exceeding capacity,
redistributes power flows, and logs cascade depth/size/geographic spread.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .simulator import GridState, ChaosEngine, TransmissionLine

# Fix 3A: UFLS region priority weights
# Lower weight = load shed first (industrial). Higher weight = protected (critical).
REGION_UFLS_PRIORITY = {
    "País Vasco": 0.5,              # Heavy industry
    "Asturias": 0.5,                # Industrial
    "Cantabria": 0.6,
    "Aragón": 0.6,
    "Castilla y León": 0.7,
    "Castilla-La Mancha": 0.7,
    "Extremadura": 0.7,
    "La Rioja": 0.7,
    "Navarra": 0.7,
    "Galicia": 0.8,
    "Cataluña": 0.8,
    "Comunidad Valenciana": 0.8,
    "Murcia": 0.8,
    "Andalucía": 0.9,
    "Madrid": 1.0,                  # Capital, heavily residential/critical
    "Baleares": 1.0,                # Island, critical
    "Canarias": 1.0,                # Island, critical
    "Ceuta": 1.0,
    "Melilla": 1.0,
}


@dataclass
class CascadeStep:
    """Records one step in a cascading failure sequence."""
    depth: int
    tripped_lines: List[str]
    tripped_generators: List[str]
    overloaded_lines_before: int
    overloaded_lines_after: int
    total_generation_mw: float
    total_demand_mw: float
    deficit_mw: float
    frequency_hz: float
    affected_regions: List[str]


@dataclass
class CascadeResult:
    """Full result of a cascade simulation."""
    initial_trigger: str
    cascade_depth: int = 0
    cascade_size: int = 0           # Total components lost
    steps: List[CascadeStep] = field(default_factory=list)
    affected_regions: List[str] = field(default_factory=list)
    geographic_spread: int = 0      # Number of unique regions affected
    final_deficit_mw: float = 0.0
    final_frequency_hz: float = 50.0
    total_load_lost_mw: float = 0.0
    total_generation_lost_mw: float = 0.0
    system_stable: bool = True
    blackout: bool = False
    recovery_actions: List[str] = field(default_factory=list)


class CascadeEngine:
    """
    Dedicated cascade propagation engine.

    Section 5 of the chaos engineering doc:
      Line A fails → Power transfers to Line B → Line B overloaded →
      Line B trips → Power transfers to C/D → Voltage instability →
      Generator trips → Regional imbalance → CASCADE
    """

    def __init__(self, engine: ChaosEngine):
        self.engine = engine
        self.state = engine.state

    def run_cascade(
        self,
        initial_trigger: str = "manual",
        max_depth: int = 10,
        overload_threshold_pct: float = 100.0,
        generator_trip_frequency_hz: float = 49.0,
    ) -> CascadeResult:
        """
        Run a full cascading failure simulation.

        Args:
            initial_trigger: Description of what started the cascade.
            max_depth: Maximum number of cascade iterations.
            overload_threshold_pct: Lines above this loading % auto-trip.
            generator_trip_frequency_hz: Generators trip below this frequency.

        Returns:
            CascadeResult with full cascade timeline.
        """
        result = CascadeResult(initial_trigger=initial_trigger)
        all_affected_regions = set()

        initial_gen = self.state.total_generation_mw
        initial_demand = self.state.total_demand_mw
        
        # Fix B1/B3: Track consecutive iterations a line has been overloaded
        line_overload_iterations: Dict[str, int] = {}

        # --- Add step 0 for initial faults ---
        step0 = CascadeStep(
            depth=0,
            tripped_lines=[],
            tripped_generators=[],
            overloaded_lines_before=0,
            overloaded_lines_after=len(self.state.overloaded_lines),
            total_generation_mw=self.state.total_generation_mw,
            total_demand_mw=self.state.total_demand_mw,
            deficit_mw=self.state.deficit_mw,
            frequency_hz=self.state.frequency_hz,
            affected_regions=[],
        )
        
        for event in self.engine.event_log:
            if event.target_type == "generator":
                step0.tripped_generators.append(event.target_id)
                if event.target_id in self.engine._initial_state.generators:
                    step0.affected_regions.append(self.engine._initial_state.generators[event.target_id].region)
            elif event.target_type == "line":
                step0.tripped_lines.append(event.target_id)

        step0.affected_regions = list(set(step0.affected_regions))
        all_affected_regions.update(step0.affected_regions)
        result.steps.append(step0)

        for depth in range(1, max_depth + 1):
            # 1. Find overloaded lines & evaluate thermal ratings (Fix B1/B3)
            # Normal: <100%, LTE: 100-115%, STE: 115-130%, Instant: >130%
            overloaded = []
            current_overloads = [l for l in self.state.lines.values() if l.is_active and l.loading_pct > overload_threshold_pct]
            
            for line in current_overloads:
                # Increment counter
                line_overload_iterations[line.id] = line_overload_iterations.get(line.id, 0) + 1
                iterations = line_overload_iterations[line.id]
                
                # Check thermal rating tripping logic
                if line.loading_pct >= 130.0:
                    overloaded.append(line) # Instant trip
                elif line.loading_pct >= 115.0 and iterations >= 2:
                    overloaded.append(line) # STE trip after 2 cascade iterations
                elif line.loading_pct >= 100.0 and iterations >= 4:
                    overloaded.append(line) # LTE trip after 4 cascade iterations

            # Clear tracking for lines that are no longer overloaded
            current_overload_ids = {l.id for l in current_overloads}
            for line_id in list(line_overload_iterations.keys()):
                if line_id not in current_overload_ids:
                    del line_overload_iterations[line_id]

            # 2. Check for UFLS (Under Frequency Load Shedding) and Gen trips
            freq_trips = []
            load_shed_total = 0.0
            
            # Fix 3A: Prioritized UFLS (shed industrial regions first, protect critical)
            if self.state.frequency_hz < 49.2:
                # Determine stage based on frequency
                if self.state.frequency_hz < 48.8:
                    base_shed_pct = 0.15 # Stage 3
                elif self.state.frequency_hz < 49.0:
                    base_shed_pct = 0.10 # Stage 2
                else:
                    base_shed_pct = 0.05 # Stage 1
                    
                for region, demand in self.state.regional_demand_mw.items():
                    # Lower-priority regions (industry) shed more
                    priority = REGION_UFLS_PRIORITY.get(region, 0.8)
                    region_shed_pct = base_shed_pct * (1.5 - priority)  # 0.5 priority → 1.0x base, 1.0 priority → 0.5x base
                    shed = demand * region_shed_pct
                    if shed > 0:
                        self.state.regional_demand_mw[region] = max(0.0, demand - shed)
                        load_shed_total += shed
                        
                # Dropping load improves frequency
                if load_shed_total > 0:
                    self.engine._update_frequency(power_imbalance_mw=-load_shed_total)

            # Fix 3B: Voltage collapse detection
            # If multiple long/heavily-loaded lines are stressed, trigger voltage collapse
            voltage_stressed_lines = []
            for line in self.state.lines.values():
                if line.is_active and line.loading_pct > 80.0:
                    # Check if it's a "long" line (high reactance = long distance)
                    reactance = self.engine._line_reactance(line)
                    if reactance > 0.05:  # Long corridor
                        voltage_stressed_lines.append(line)
            
            if len(voltage_stressed_lines) >= 3:
                # Voltage collapse event: trip the most stressed line
                worst = max(voltage_stressed_lines, key=lambda l: l.loading_pct)
                worst.is_active = False
                self.engine._redistribute_flow(worst)
                all_affected_regions.update({worst.from_node, worst.to_node})
                overloaded.append(worst)  # Include in this round's tripped lines

            # Generator protection trip (under-frequency relays)
            if self.state.frequency_hz < 47.5:
                online_gens = [g for g in self.state.generators.values() if g.is_online]
                # Trip up to 20% of remaining generators to protect them
                n_to_trip = max(1, len(online_gens) // 5)
                freq_trips = sorted(online_gens, key=lambda g: g.current_output_mw)[:n_to_trip]

            if not overloaded and not freq_trips and load_shed_total == 0.0:
                break  # System stabilized

            # 3. Record pre-trip state
            overloaded_before = len(overloaded)

            # 4. Trip overloaded lines (with Relay Hidden Failures - Fix B2)
            import random
            tripped_lines = []
            for line in overloaded:
                # Fix B2: 3% chance of relay hidden failure (misoperation)
                # Misoperation trips an adjacent healthy line instead of or alongside the faulted line.
                if random.random() < 0.03:
                    # Find active adjacent lines
                    adj_lines = [
                        l for l in self.state.lines.values()
                        if l.is_active and l.id != line.id and (
                            l.from_node in {line.from_node, line.to_node} or
                            l.to_node in {line.from_node, line.to_node}
                        )
                    ]
                    if adj_lines:
                        hidden_failure_line = random.choice(adj_lines)
                        hidden_failure_line.is_active = False
                        self.engine._redistribute_flow(hidden_failure_line)
                        tripped_lines.append(f"{hidden_failure_line.id} (Relay Misoperation)")
                        hidden_failure_line.current_flow_mw = 0.0
                        all_affected_regions.update({hidden_failure_line.from_node, hidden_failure_line.to_node})
                
                # Normal intentional trip
                regions = {line.from_node, line.to_node}
                all_affected_regions.update(regions)

                line.is_active = False
                self.engine._redistribute_flow(line)
                tripped_lines.append(line.id)
                line.current_flow_mw = 0.0

            # 5. Trip under-frequency generators (protection)
            tripped_gens = []
            for gen in freq_trips:
                all_affected_regions.add(gen.region)
                lost = gen.current_output_mw
                gen.is_online = False
                gen.current_output_mw = 0.0
                self.engine._update_frequency(power_imbalance_mw=lost)
                tripped_gens.append(gen.id)

            # 6. Check for generator isolation
            self.engine._check_generator_isolation()

            # 8. Record this cascade step
            overloaded_after = len(self.state.overloaded_lines)
            step = CascadeStep(
                depth=depth,
                tripped_lines=tripped_lines,
                tripped_generators=tripped_gens,
                overloaded_lines_before=overloaded_before,
                overloaded_lines_after=overloaded_after,
                total_generation_mw=self.state.total_generation_mw,
                total_demand_mw=self.state.total_demand_mw,
                deficit_mw=self.state.deficit_mw,
                frequency_hz=self.state.frequency_hz,
                affected_regions=list(all_affected_regions),
            )
            result.steps.append(step)

            # 9. Check for total blackout
            if self.state.total_generation_mw <= 0:
                result.blackout = True
                break

        # Compile final result
        result.cascade_depth = len(result.steps)
        result.cascade_size = sum(
            len(s.tripped_lines) + len(s.tripped_generators) for s in result.steps
        )
        result.affected_regions = list(all_affected_regions)
        result.geographic_spread = len(all_affected_regions)
        result.final_deficit_mw = self.state.deficit_mw
        result.final_frequency_hz = self.state.frequency_hz
        result.total_generation_lost_mw = max(0, initial_gen - self.state.total_generation_mw)
        result.total_load_lost_mw = result.final_deficit_mw
        result.system_stable = len(self.state.overloaded_lines) == 0 and not result.blackout

        # Generate recovery recommendations
        result.recovery_actions = self._recommend_recovery(result)

        return result

    def _recommend_recovery(self, result: CascadeResult) -> List[str]:
        """Generate recovery action recommendations based on cascade result."""
        actions = []

        if result.blackout:
            actions.append("CRITICAL: Initiate black start procedure")
            actions.append("Prioritize nuclear and hydro units for restart")
            actions.append("Request maximum emergency imports from all interconnections")
            return actions

        if result.final_deficit_mw > 0:
            actions.append(f"Redispatch {result.final_deficit_mw:.0f} MW from available reserves")

            # Check which technologies have spare capacity
            for gen in self.state.generators.values():
                if gen.is_online and gen.available_reserve_mw > 100:
                    actions.append(
                        f"Ramp up {gen.id} ({gen.technology}): "
                        f"+{gen.available_reserve_mw:.0f} MW available"
                    )

        if result.final_frequency_hz < 49.5:
            actions.append(f"URGENT: Frequency at {result.final_frequency_hz:.2f} Hz — activate frequency reserves")

        if result.geographic_spread >= 3:
            actions.append(f"Wide-area event ({result.geographic_spread} regions) — coordinate with TSO")

        total_import_capacity = sum(self.state.interconnections.values())
        if total_import_capacity > 0 and result.final_deficit_mw > 0:
            actions.append(f"Emergency import capacity available: {total_import_capacity:.0f} MW")

        return actions

    def to_dict(self, result: CascadeResult) -> dict:
        """Serialize a CascadeResult to a JSON-friendly dict."""
        return {
            "initial_trigger": result.initial_trigger,
            "cascade_depth": result.cascade_depth,
            "cascade_size": result.cascade_size,
            "geographic_spread": result.geographic_spread,
            "affected_regions": result.affected_regions,
            "final_deficit_mw": round(result.final_deficit_mw, 1),
            "final_frequency_hz": round(result.final_frequency_hz, 3),
            "total_generation_lost_mw": round(result.total_generation_lost_mw, 1),
            "total_load_lost_mw": round(result.total_load_lost_mw, 1),
            "system_stable": result.system_stable,
            "blackout": result.blackout,
            "recovery_actions": result.recovery_actions,
            "steps": [
                {
                    "depth": s.depth,
                    "tripped_lines": s.tripped_lines,
                    "tripped_generators": s.tripped_generators,
                    "overloaded_lines_before": s.overloaded_lines_before,
                    "overloaded_lines_after": s.overloaded_lines_after,
                    "generation_mw": round(s.total_generation_mw, 1),
                    "demand_mw": round(s.total_demand_mw, 1),
                    "deficit_mw": round(s.deficit_mw, 1),
                    "frequency_hz": round(s.frequency_hz, 3),
                    "affected_regions": s.affected_regions,
                }
                for s in result.steps
            ],
        }
