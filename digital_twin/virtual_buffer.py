"""
VirtualBuffer — Imaginary energy container for the Digital Twin.

The concept:
  Throughout the day, some regions generate MORE than they need.
  Instead of wasting this surplus, it accumulates in a per-region
  virtual buffer (physically: pumped hydro, BESS, smart grid storage).

  When the Chaos Engine trips a plant and creates a deficit:
    1. Buffer discharge is checked FIRST (cheapest, cleanest source)
    2. Then spatial excess routing (instantaneous overproduction)
    3. Then conventional ramp-up (Hydro -> Gas -> Coal)
    4. Finally emergency imports (most expensive)

Economic ordering:
  Buffer (EUR 2/MWh) < Hydro (EUR 5) < Gas (EUR 65) < Coal (EUR 85) < Imports (EUR 90)

Distance physics:
  - Transmission losses scale with distance: ~1.5% per 100 km on 400kV
  - Island regions (Canarias, Baleares) are electrically isolated from mainland
    (no HVDC link exists in the model scope)

Usage:
    buffer = VirtualBuffer()
    buffer.charge("Extremadura", excess_mw=1200, duration_hours=6.0)
    actions = buffer.dispatch_to_cover_deficit({"Madrid": 800})
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List


# Per-region storage capacity estimates (MWh)
# Based on pumped hydro + BESS infrastructure
REGION_BUFFER_CAPACITY_MWH: Dict[str, float] = {
    "Galicia":              4800.0,   # Large hydro / pumped hydro
    "Castilla y León":      3500.0,   # Significant hydro + wind curtailment
    "Aragón":               2800.0,   # Hydro + growing BESS
    "Extremadura":          2500.0,   # Solar + pumped hydro
    "Andalucía":            2200.0,   # Solar surplus
    "Cataluña":             2000.0,   # Mixed + BESS
    "Castilla-La Mancha":   2000.0,   # Wind surplus
    "Comunidad Valenciana": 1500.0,
    "Madrid":               1200.0,   # Mostly demand node
    "País Vasco":           1000.0,
    "Cantabria":             800.0,
    "Asturias":              800.0,
    "Navarra":               600.0,
    "Murcia":                500.0,
    "La Rioja":              400.0,
    "Baleares":              200.0,   # Island — limited storage
    "Canarias":              200.0,   # Island — limited storage
    "Ceuta":                  50.0,
    "Melilla":                50.0,
}

DEFAULT_CAPACITY_MWH = 1000.0

# Regions that are electrically isolated from the mainland grid
# (no HVDC interconnection in this model scope)
ISLAND_REGIONS = {"Canarias", "Islas Canarias", "Baleares", "Islas Baleares", "Ceuta", "Melilla",
                  "Comunidad de Ceuta", "Comunidad de Melilla"}

# Canonical mapping: official CCAA names (from REE / ENTSO-E data) → internal short keys
# These official names appear in demand_regional_disaggregated.csv and node_ids.csv.
# The short keys are used in REGION_CENTROIDS and REGION_BUFFER_CAPACITY_MWH.
REGION_NAME_MAP: dict = {
    # Official name                        : internal key
    "Comunidad de Madrid":                   "Madrid",
    "Comunidad Foral de Navarra":            "Navarra",
    "Comunidad de Navarra":                  "Navarra",
    "Principado de Asturias":               "Asturias",
    "Region de Murcia":                     "Murcia",
    "Regi\u00f3n de Murcia":                     "Murcia",
    "Islas Baleares":                       "Baleares",
    "Islas Canarias":                       "Canarias",
    "Comunidad de Ceuta":                   "Ceuta",
    "Comunidad de Melilla":                 "Melilla",
    "Castilla la Mancha":                   "Castilla-La Mancha",
    "Castilla-La Mancha":                   "Castilla-La Mancha",
    "Castilla y Leon":                      "Castilla y Leon",
    "Castilla y Le\u00f3n":                      "Castilla y Leon",
    "Aragon":                               "Aragon",
    "Arag\u00f3n":                               "Aragon",
    "Catalua":                              "Cataluna",
    "Catalu\u00f1a":                             "Cataluna",
    "Andalucia":                            "Andalucia",
    "Andaluc\u00eda":                            "Andalucia",
    "Pas Vasco":                            "Pais Vasco",
    "Pa\u00eds Vasco":                           "Pais Vasco",
    "Comunidad Valenciana":                  "Comunidad Valenciana",
    "Extremadura":                           "Extremadura",
    "Galicia":                              "Galicia",
    "Cantabria":                            "Cantabria",
    "La Rioja":                             "La Rioja",
    "Madrid":                               "Madrid",
    "Navarra":                              "Navarra",
    "Asturias":                             "Asturias",
    "Murcia":                               "Murcia",
    "Baleares":                             "Baleares",
    "Canarias":                             "Canarias",
    "Ceuta":                                "Ceuta",
    "Melilla":                              "Melilla",
}


def normalize_region(name: str) -> str:
    """
    Normalize any region name variant to the internal short key used in
    REGION_CENTROIDS and REGION_BUFFER_CAPACITY_MWH.

    Handles:
      - Official CCAA names with accents (Andalucía, Aragón, ...)
      - Mojibake variants from encoding issues (Castilla la Mancha, Cataluña, ...)
      - Already-normalized short keys (pass-through)
    """
    return REGION_NAME_MAP.get(name, name)


REGION_CENTROIDS: Dict[str, tuple] = {
    "Galicia":              (42.78, -7.87),
    "Asturias":             (43.36, -5.86),
    "Cantabria":            (43.18, -3.99),
    "Pais Vasco":           (43.02, -2.68),
    "Navarra":              (42.69, -1.64),
    "La Rioja":             (42.29, -2.34),
    "Aragon":               (41.60, -0.91),
    "Cataluna":             (41.83,  1.52),
    "Castilla y Leon":      (41.65, -4.72),
    "Madrid":               (40.42, -3.70),
    "Comunidad Valenciana": (39.48, -0.38),
    "Extremadura":          (39.17, -6.01),
    "Castilla-La Mancha":   (39.86, -3.25),
    "Murcia":               (37.99, -1.12),
    "Andalucia":            (37.45, -4.73),
    "Canarias":             (28.29,-15.63),
    "Islas Canarias":       (28.29,-15.63),
    "Baleares":             (39.57,  2.65),
    "Islas Baleares":       (39.57,  2.65),
    "Ceuta":                (35.89, -5.32),
    "Melilla":              (35.29, -2.94),
}

# Max transmission distance (km) for buffer dispatch
# Beyond this, losses make the transfer uneconomical
MAX_BUFFER_DISPATCH_KM = 1200.0



@dataclass
class BufferDischargeRecord:
    """Records a single buffer discharge event."""
    from_region: str
    to_region: str
    discharged_mw: float
    transmission_efficiency: float = 1.0
    cost_eur_mwh: float = 2.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _region_distance_km(r1: str, r2: str) -> float:
    """Distance in km between two region centroids. Returns 9999 if unknown."""
    c1 = REGION_CENTROIDS.get(r1)
    c2 = REGION_CENTROIDS.get(r2)
    if not c1 or not c2:
        return 9999.0
    return _haversine_km(c1[0], c1[1], c2[0], c2[1])


def _transmission_efficiency(distance_km: float) -> float:
    """
    Efficiency of power transmission over a given distance.
    Assumes 400kV overhead line: ~1.5% loss per 100 km.
    Returns fraction of sent power that arrives (e.g. 0.88 for 800km).
    """
    loss_pct_per_100km = 1.5
    loss = min(0.99, (distance_km / 100.0) * loss_pct_per_100km / 100.0)
    return max(0.01, 1.0 - loss)




class VirtualBuffer:
    """
    Per-region virtual energy storage buffer.

    Accumulates surplus generation throughout the simulation day,
    then discharges it preferentially during fault recovery.

    Dispatch priority:
      - Nearest mainland regions first (distance-weighted)
      - Island regions cannot supply mainland (no HVDC link)
      - Transmission efficiency applied (1.5% loss per 100 km)
    """

    # Buffer discharge cost per MWh — strictly below Hydro (€5/MWh)
    # Justification: energy was already generated; cost is only scheduling overhead
    DISCHARGE_COST_EUR_MWH = 2.0

    # Round-trip efficiency (L10 fix)
    # Pumped hydro: ~80%, BESS: ~90%. We use 80% (conservative / mixed storage).
    # Charging 1000 MWh only stores 800 MWh — physically unavoidable loss.
    ROUND_TRIP_EFFICIENCY = 0.80

    # Max discharge rate expressed as C-rate fraction (L4 fix)
    # A C-rate of 0.25 means a full discharge takes 4 hours (realistic for pumped hydro).
    # This prevents the buffer from discharging 4800 MWh in a single 1-hour window.
    MAX_DISCHARGE_C_RATE = 0.25  # max MW = stored_mwh * C_rate / window_hours

    def __init__(self):
        # Current stored energy per region (MWh)
        self.stored_mwh: Dict[str, float] = {}

        # Maximum capacity per region
        self.capacity_mwh: Dict[str, float] = dict(REGION_BUFFER_CAPACITY_MWH)

        # Lifetime stats
        self.total_charged_mwh: float = 0.0
        self.total_discharged_mwh: float = 0.0

        # Action log (returned as healing actions)
        self.discharge_log: List[BufferDischargeRecord] = []

    # ─── Charging ───────────────────────────────────────────────────────

    def charge(self, region: str, excess_mw: float, duration_hours: float = 1.0) -> float:
        """
        Add excess generation to the region's buffer.

        Region name is normalized automatically, so both official CCAA names
        (e.g. "Comunidad de Madrid") and short keys ("Madrid") work.

        Args:
            region: Region name (official or short form)
            excess_mw: Surplus generation rate (MW)
            duration_hours: How long this surplus lasted (default 1 hour)

        Returns:
            MWh actually added (may be capped by capacity)
        """
        if excess_mw <= 0.0:
            return 0.0

        region = normalize_region(region)  # always work with canonical short key

        # L10 fix: Apply round-trip efficiency at charge time.
        # A 80% RTE means only 80% of incoming energy is recoverable on discharge.
        mwh_to_add = excess_mw * duration_hours * self.ROUND_TRIP_EFFICIENCY
        current = self.stored_mwh.get(region, 0.0)
        cap = self.capacity_mwh.get(region, DEFAULT_CAPACITY_MWH)
        space = max(0.0, cap - current)
        added = min(mwh_to_add, space)

        self.stored_mwh[region] = current + added
        self.total_charged_mwh += added
        return added

    def charge_from_state(self, regional_excess_mw: Dict[str, float], duration_hours: float = 1.0):
        """
        Bulk-charge the buffer from a regional_excess_mw snapshot.

        Called by SnapshotBuilder after computing daily overproduction.
        """
        for region, excess in regional_excess_mw.items():
            if excess > 0.1:
                self.charge(region, excess, duration_hours)

    # ─── Discharging ────────────────────────────────────────────────────

    def get_available_mw(self, region: str, window_hours: float = 1.0) -> float:
        """
        How much power (MW) can this region discharge right now?
        Computed as stored_mwh / window_hours.
        """
        stored = self.stored_mwh.get(region, 0.0)
        return stored / max(window_hours, 0.001)

    def discharge(self, region: str, needed_mw: float, window_hours: float = 1.0) -> float:
        """
        Withdraw energy from a region's buffer, respecting ramp-rate limits.

        L4 fix: Max discharge rate = stored_mwh * MAX_DISCHARGE_C_RATE / window_hours.
        Prevents physically impossible instantaneous full discharge.

        Args:
            region: The source region to draw from
            needed_mw: Demand rate (MW)
            window_hours: Assumed dispatch window (default 1 hour)

        Returns:
            Actual MW dispatched (≤ needed_mw, ≤ available, ≤ ramp limit)
        """
        stored = self.stored_mwh.get(region, 0.0)
        if stored <= 0.0:
            return 0.0

        # Physical ramp-rate cap: can't discharge faster than C-rate allows
        max_mw_from_c_rate = (stored * self.MAX_DISCHARGE_C_RATE) / max(window_hours, 0.001)
        max_mw = min(needed_mw, max_mw_from_c_rate)

        needed_mwh = max_mw * window_hours
        actual_mwh = min(needed_mwh, stored)
        self.stored_mwh[region] = stored - actual_mwh
        self.total_discharged_mwh += actual_mwh
        return actual_mwh / max(window_hours, 0.001)  # Convert back to MW

    def dispatch_to_cover_deficit(
        self,
        deficit_regions: Dict[str, float],
        window_hours: float = 1.0,
    ) -> List[dict]:
        """
        Try to cover deficits in affected regions using buffer stored in
        ANY eligible region.

        Donor priority order:
          1. Same region as deficit (self-discharge, zero transmission loss)
          2. Nearest mainland region with stored energy
          3. Farther mainland regions
          4. Island regions are EXCLUDED unless the deficit is also on the same island
             (no physical HVDC link to mainland in this model)

        Transmission efficiency:
          ~1.5% loss per 100 km on 400kV lines.
          Donors beyond MAX_BUFFER_DISPATCH_KM are skipped (uneconomical).

        Args:
            deficit_regions: {region: deficit_mw} — regions needing power
            window_hours: Dispatch window assumed

        Returns:
            List of buffer_discharge action dicts for the healing log
        """
        actions = []
        self.discharge_log.clear()

        remaining_deficits = dict(deficit_regions)  # copy

        for deficit_region, deficit_mw in list(remaining_deficits.items()):
            if deficit_mw <= 0.1:
                continue

            deficit_is_island = normalize_region(deficit_region) in ISLAND_REGIONS

            # Build sorted donor list for this specific deficit region:
            # Score = stored_mwh * efficiency / distance_km  (higher = better)
            donors = []
            for donor_region, stored in self.stored_mwh.items():
                if stored < 1.0:
                    continue

                donor_is_island = normalize_region(donor_region) in ISLAND_REGIONS

                # Island isolation constraint:
                # - Island cannot supply mainland
                # - Mainland cannot supply a different island
                if donor_is_island and not deficit_is_island:
                    continue
                if deficit_is_island and not donor_is_island:
                    continue
                if donor_is_island and deficit_is_island and donor_region != deficit_region:
                    # Different islands — no cross-island cable
                    continue

                dist_km = _region_distance_km(donor_region, deficit_region)

                # Skip if too far (uneconomical — losses > ~18%)
                if dist_km > MAX_BUFFER_DISPATCH_KM:
                    continue

                efficiency = _transmission_efficiency(dist_km)
                # Score: prefer close, high-storage donors
                score = (stored * efficiency) / max(dist_km, 1.0)

                donors.append((score, donor_region, dist_km, efficiency))

            # Sort by score descending (best donor first)
            donors.sort(key=lambda x: -x[0])

            remaining = deficit_mw
            for score, donor_region, dist_km, efficiency in donors:
                if remaining <= 0.1:
                    break

                available_mw = self.get_available_mw(donor_region, window_hours)
                if available_mw < 1.0:
                    continue

                # How much to send, accounting for transmission losses
                # We need `remaining` MW to arrive, so we send `remaining / efficiency`
                mw_to_send = min(remaining / efficiency, available_mw)
                actually_withdrawn = self.discharge(donor_region, mw_to_send, window_hours)
                mw_arrives = actually_withdrawn * efficiency

                if mw_arrives >= 0.5:
                    remaining -= mw_arrives
                    transport_cost = self.DISCHARGE_COST_EUR_MWH + (dist_km * 0.02)  # €/MWh per km penalty
                    cost = round(mw_arrives * transport_cost, 0)

                    action = {
                        "type": "buffer_discharge",
                        "from_region": donor_region,
                        "to_region": deficit_region,
                        "discharged_mw": round(mw_arrives, 1),
                        "sent_mw": round(actually_withdrawn, 1),
                        "distance_km": round(dist_km, 0),
                        "transmission_efficiency": round(efficiency, 3),
                        "cost_eur": cost,
                    }
                    actions.append(action)
                    self.discharge_log.append(
                        BufferDischargeRecord(
                            from_region=donor_region,
                            to_region=deficit_region,
                            discharged_mw=mw_arrives,
                            transmission_efficiency=efficiency,
                            cost_eur_mwh=transport_cost,
                        )
                    )

            remaining_deficits[deficit_region] = max(0.0, remaining)

        return actions


    # ─── State & Reporting ──────────────────────────────────────────────

    def total_stored_mwh(self) -> float:
        return sum(self.stored_mwh.values())

    def summary(self) -> dict:
        """Return a JSON-serializable summary of the buffer state."""
        return {
            "total_stored_mwh": round(self.total_stored_mwh(), 1),
            "total_charged_mwh": round(self.total_charged_mwh, 1),
            "total_discharged_mwh": round(self.total_discharged_mwh, 1),
            "per_region": {
                r: round(v, 1)
                for r, v in sorted(self.stored_mwh.items(), key=lambda x: -x[1])
                if v >= 1.0
            },
        }

    def __repr__(self) -> str:
        return (
            f"VirtualBuffer(stored={self.total_stored_mwh():.0f} MWh, "
            f"charged={self.total_charged_mwh:.0f} MWh, "
            f"discharged={self.total_discharged_mwh:.0f} MWh)"
        )
