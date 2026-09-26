"""
SnapshotBuilder — Constructs a fully calibrated GridState from real data
for any historical hour (2015-2025).

Reads:
  - plant_metadata.csv (155 plants with region, capacity, technology)
  - generation_per_plant_11_years.csv (hourly output per plant)
  - demand_regional_disaggregated.csv (hourly regional demand)
  - interchanges.csv (hourly net imports/exports)
  - gen_*.csv (technology-level aggregates for validation)
  - A_elec.npy (electrical adjacency matrix for transmission topology)
  - node_ids.csv (region names and coordinates)

Returns:
  A fully populated GridState dataclass ready for chaos injection.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional
from .simulator import GridState, Generator, TransmissionLine
from .virtual_buffer import VirtualBuffer, normalize_region



# ENTSO-E tech code → simplified category
TECH_CATEGORY = {
    "B02": "Coal", "B04": "Gas", "B05": "Coal",
    "B10": "Hydro", "B12": "Hydro", "B14": "Nuclear",
    "B16": "Solar", "B19": "Wind",
}


class SnapshotBuilder:
    """
    Builds a calibrated GridState from real ENTSO-E + REE data.
    
    Caches heavy data on first load so subsequent snapshots are fast.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._metadata: Optional[pd.DataFrame] = None
        self._gen_data: Optional[pd.DataFrame] = None
        self._demand_data: Optional[pd.DataFrame] = None
        self._interchange_data: Optional[pd.DataFrame] = None
        self._adjacency: Optional[np.ndarray] = None
        self._node_ids: Optional[pd.DataFrame] = None
        self._gen_by_tech: dict = {}   # tech_file → DataFrame
        self._loaded = False

    def _load_data(self):
        """Load and cache all data sources on first call."""
        if self._loaded:
            return

        proc = os.path.join(self.data_dir, "processed")

        # 1. Plant metadata (small, fast)
        meta_path = os.path.join(proc, "plant_metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"plant_metadata.csv not found at {meta_path}. "
                "Run scripts/build_plant_metadata.py first."
            )
        self._metadata = pd.read_csv(meta_path)
        print(f"[SnapshotBuilder] Loaded {len(self._metadata)} plant metadata entries.")

        # 2. Plant generation (large — 338MB, keep datetime as string for fast matching)
        gen_path = os.path.join(proc, "entsoe_generation", "generation_per_plant_11_years.csv")
        print(f"[SnapshotBuilder] Loading plant generation data (this takes a few seconds)...")
        self._gen_data = pd.read_csv(gen_path)
        print(f"[SnapshotBuilder] Loaded {len(self._gen_data)} generation records.")

        # 3. Regional demand
        demand_path = os.path.join(proc, "grid_telemetry", "demand_regional_disaggregated.csv")
        self._demand_data = pd.read_csv(demand_path)
        print(f"[SnapshotBuilder] Loaded demand data: {len(self._demand_data)} rows.")

        # 4. Interchange (imports/exports)
        inter_path = os.path.join(proc, "grid_telemetry", "interchanges.csv")
        self._interchange_data = pd.read_csv(inter_path)
        print(f"[SnapshotBuilder] Loaded interchange data.")

        # 5. Adjacency matrix and node IDs
        graph_dir = os.path.join(proc, "graphs")
        a_elec_path = os.path.join(graph_dir, "A_elec.npy")
        node_ids_path = os.path.join(graph_dir, "node_ids.csv")

        if os.path.exists(a_elec_path):
            self._adjacency = np.load(a_elec_path)
        if os.path.exists(node_ids_path):
            self._node_ids = pd.read_csv(node_ids_path)

        self._loaded = True
        print("[SnapshotBuilder] All data loaded and cached.")

    def build_snapshot(self, target_datetime: str) -> GridState:
        """
        Build a fully calibrated GridState for a specific datetime.
        
        Args:
            target_datetime: ISO format string, e.g. "2025-06-15 14:00:00"
            
        Returns:
            GridState populated with real generator outputs, regional demand,
            and transmission lines from the adjacency matrix.
        """
        self._load_data()

        # Normalize target to string format matching the CSV
        target_str = str(pd.Timestamp(target_datetime))
        state = GridState(timestamp=target_str)

        # ── 1. Build generators from real plant data ──
        # Use string matching (reliable with arrow-backed strings)
        hour_gen = self._gen_data[self._gen_data["datetime"] == target_str].copy()

        if hour_gen.empty:
            # Try nearest hour from unique timestamps
            unique_times = sorted(self._gen_data["datetime"].unique())
            # Binary search for nearest
            import bisect
            idx = bisect.bisect_left(unique_times, target_str)
            if idx >= len(unique_times):
                idx = len(unique_times) - 1
            nearest_str = unique_times[idx]
            hour_gen = self._gen_data[self._gen_data["datetime"] == nearest_str].copy()
            print(f"[SnapshotBuilder] Using nearest hour: {nearest_str}")

        for _, row in hour_gen.iterrows():
            uid = row["unit_id"]
            if uid == "UNKNOWN":
                continue

            # Look up metadata
            meta_row = self._metadata[self._metadata["unit_id"] == uid]
            if meta_row.empty:
                continue

            meta = meta_row.iloc[0]
            output_mw = max(0.0, float(row["generation_mw"]))
            capacity = float(meta["estimated_capacity_mw"])
            tech = str(meta["tech_category"])
            region = str(meta["region"])

            gen = Generator(
                id=uid,
                technology=tech,
                region=region,
                installed_capacity_mw=capacity,
                current_output_mw=min(output_mw, capacity),
                is_online=output_mw > 0.1,
            )
            state.generators[uid] = gen

        # ── 2. Build regional demand ──
        demand_hour = self._demand_data[
            self._demand_data["datetime"] == target_str
        ]
        if demand_hour.empty:
            # Try nearest — grab the 19 rows closest to target (one per region)
            unique_demand_times = sorted(self._demand_data["datetime"].unique())
            import bisect
            d_idx = bisect.bisect_left(unique_demand_times, target_str)
            if d_idx >= len(unique_demand_times):
                d_idx = len(unique_demand_times) - 1
            nearest_d = unique_demand_times[d_idx]
            demand_hour = self._demand_data[self._demand_data["datetime"] == nearest_d]

        # The demand data has columns: datetime, region, demand_mwh
        # (or might be pivoted — handle both formats)
        if "region" in demand_hour.columns:
            for _, row in demand_hour.iterrows():
                region = str(row.get("region", ""))
                demand = float(row.get("demand_mwh", row.get("value", 0)))
                if region and demand > 0:
                    state.regional_demand_mw[region] = demand
        else:
            # Pivoted format: columns are region names
            for col in demand_hour.columns:
                if col != "datetime":
                    val = demand_hour[col].iloc[0] if len(demand_hour) > 0 else 0
                    if pd.notna(val) and float(val) > 0:
                        state.regional_demand_mw[col] = float(val)

        # ── 3. Net imports/exports ──
        inter_hour = self._interchange_data[
            self._interchange_data["datetime"] == target_str
        ]
        if not inter_hour.empty:
            # Positive = importing, Negative = exporting
            net_flow = float(inter_hour["value"].iloc[0])
            state.imports_mw = max(0.0, net_flow)  # Only count net imports

        # ── 4. Build transmission lines from adjacency matrix ──
        if self._adjacency is not None and self._node_ids is not None:
            self._build_transmission_lines(state)

        print(f"[SnapshotBuilder] Snapshot built for {target_str}:")
        print(f"  Generators: {len(state.generators)} "
              f"(online: {sum(1 for g in state.generators.values() if g.is_online)})")
        print(f"  Total generation: {state.total_generation_mw:.0f} MW")
        print(f"  Total demand: {state.total_demand_mw:.0f} MW")
        print(f"  Net imports: {state.imports_mw:.0f} MW")
        print(f"  Transmission lines: {len(state.lines)}")

        # ── 5. Compute regional_excess_mw from actual gen vs demand ──────────
        # This is needed by HealingAgent for spatial routing and buffer dispatch.
        # DCPF also sets this, but only for nodes that appear in the adjacency
        # matrix — computing it here from raw data ensures complete coverage.
        self._compute_regional_excess(state, target_str)

        # ── 6. Charge the virtual buffer from today's accumulated surplus ──
        self._charge_virtual_buffer(state, target_str)
        excess_regions = sum(1 for v in state.regional_excess_mw.values() if v > 0)
        deficit_regions = sum(1 for v in state.regional_excess_mw.values() if v < 0)
        print(f"  Regional balance: {excess_regions} surplus, {deficit_regions} deficit regions")
        print(f"  Virtual buffer: {state.virtual_buffer.total_stored_mwh():.0f} MWh stored")

        return state


    def _compute_regional_excess(self, state: GridState, target_str: str):
        """
        Compute regional_excess_mw = regional_generation - regional_demand
        for the snapshot hour.

        This uses the same data sources as the snapshot itself, normalized
        to canonical short region keys so HealingAgent can route power correctly.
        """
        try:
            target_dt = pd.Timestamp(target_str)

            # Per-region generation: sum current_output_mw of online generators
            gen_by_region: dict = {}
            for g in state.generators.values():
                if g.is_online:
                    key = normalize_region(g.region)
                    gen_by_region[key] = gen_by_region.get(key, 0.0) + g.current_output_mw

            # Per-region demand: find the exact hour in demand data
            if self._demand_data is not None and "datetime" in self._demand_data.columns:
                # Parse with format='mixed' to handle both date-only and datetime strings
                demand_dt = pd.to_datetime(self._demand_data["datetime"], format='mixed', dayfirst=False)
                hour_mask = demand_dt.dt.floor('h') == target_dt.floor('h')
                hour_demand = self._demand_data[hour_mask]

                if hour_demand.empty:
                    # Fallback: use state.regional_demand_mw which was populated earlier
                    demand_by_region = {
                        normalize_region(r): v
                        for r, v in state.regional_demand_mw.items()
                    }
                else:
                    demand_by_region: dict = {}
                    for _, row in hour_demand.iterrows():
                        key = normalize_region(str(row["region"]))
                        demand_by_region[key] = demand_by_region.get(key, 0.0) + float(row["demand_mwh"])
            else:
                demand_by_region = {
                    normalize_region(r): v
                    for r, v in state.regional_demand_mw.items()
                }

            # Compute excess: positive = surplus, negative = deficit
            all_regions = set(gen_by_region.keys()) | set(demand_by_region.keys())
            excess = {}
            for region in all_regions:
                g = gen_by_region.get(region, 0.0)
                d = demand_by_region.get(region, 0.0)
                excess[region] = g - d

            state.regional_excess_mw = excess

        except Exception as e:
            print(f"[SnapshotBuilder] Warning: regional_excess_mw computation failed ({e}). Using empty dict.")
            state.regional_excess_mw = {}



    def _build_transmission_lines(self, state: GridState):
        """
        Construct TransmissionLine objects based on the adjacency matrix.
        Assign capacities using known values where possible.
        """
        if self._adjacency is None or self._node_ids is None:
            return

        from .simulator import _NODE_COORDS
        
        # Populate _NODE_COORDS for simulator's distance calculations
        if not _NODE_COORDS:
            for _, row in self._node_ids.iterrows():
                region = str(row["region"])
                _NODE_COORDS[region] = (float(row["latitude"]), float(row["longitude"]))

        # Known net transfer capacities (MW) from ENTSO-E maps
        # Fix 6A: Use exact region strings including the mojibake/encoding artifacts
        KNOWN_CAPACITIES = {
            ("Madrid", "Castilla y Len"): 3000,
            ("Madrid", "Castilla la Mancha"): 3500,
            ("Madrid", "Comunidad Valenciana"): 2500,
            ("Catalua", "Aragn"): 2200,
            ("Catalua", "Comunidad Valenciana"): 2000,
            ("Andaluca", "Extremadura"): 2000,
            ("Andaluca", "Castilla la Mancha"): 1800,
            ("Andaluca", "Regin de Murcia"): 1200,
            ("Galicia", "Principado de Asturias"): 1500,
            ("Galicia", "Castilla y Len"): 2000,
            ("Pas Vasco", "Comunidad de Navarra"): 1200,
            ("Pas Vasco", "Cantabria"): 800,
            ("Castilla y Len", "Principado de Asturias"): 1200,
            ("Comunidad Valenciana", "Regin de Murcia"): 1000,
            ("Aragn", "Comunidad de Navarra"): 1000,
        }

        def get_capacity(r1, r2):
            for (a, b), cap in KNOWN_CAPACITIES.items():
                if (a == r1 and b == r2) or (a == r2 and b == r1):
                    return cap
            return 2000.0  # Default 400kV corridor capacity

        adj = self._adjacency
        nodes = self._node_ids
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0.01:  # Connected
                    region_i = nodes.iloc[i]["region"]
                    region_j = nodes.iloc[j]["region"]

                    capacity = get_capacity(region_i, region_j)

                    line_id = f"L_{region_i[:8]}_{region_j[:8]}_{i}_{j}"
                    line = TransmissionLine(
                        id=line_id,
                        from_node=region_i,
                        to_node=region_j,
                        capacity_mw=capacity,
                        current_flow_mw=0.0,  # Will be set by DCPF
                        is_active=True,
                    )
                    state.lines[line_id] = line
                    
        # Solve DCPF to get realistic initial flows based on physics, not a fake formula
        from .simulator import ChaosEngine
        engine = ChaosEngine(state)
        engine._solve_dcpf()
        
        # Transfer computed flows back to our state
        for line_id, line in engine.state.lines.items():
            state.lines[line_id].current_flow_mw = line.current_flow_mw

    def _charge_virtual_buffer(self, state: GridState, target_str: str):
        """
        Compute the accumulated energy surplus from midnight up to the target hour
        and charge the VirtualBuffer.

        Surplus per region = generation_regional - demand_regional (per hour)
        We sum all positive hourly surpluses from 00:00 to target hour.
        """
        buffer = VirtualBuffer()
        state.virtual_buffer = buffer

        try:
            # Parse with format='mixed' to handle both 'YYYY-MM-DD' and
            # 'YYYY-MM-DD HH:MM:SS' strings that appear in the dataset.
            target_dt = pd.Timestamp(target_str)
            day_start = target_dt.normalize()  # midnight

            # Parse datetime column once, using mixed format
            if "_demand_dt" not in self.__dict__:
                self._demand_dt_parsed = pd.to_datetime(
                    self._demand_data["datetime"], format='mixed', dayfirst=False
                )

            demand_dt = self._demand_dt_parsed
            day_mask = (demand_dt >= day_start) & (demand_dt <= target_dt)
            day_demand = self._demand_data[day_mask].copy()

            if "region" not in day_demand.columns:
                # Pivoted format — skip
                return

            # Get generation data for the same window
            day_gen_mask = (
                (pd.to_datetime(self._gen_data["datetime"], format='mixed', dayfirst=False) >= day_start) &
                (pd.to_datetime(self._gen_data["datetime"], format='mixed', dayfirst=False) <= target_dt)
            )
            day_gen = self._gen_data[day_gen_mask]

            if day_gen.empty or day_demand.empty:
                return

            # Join generation with metadata to get region per plant
            gen_with_region = day_gen.merge(
                self._metadata[["unit_id", "region"]],
                on="unit_id",
                how="left",
            ).dropna(subset=["region"])

            region_gen = (
                gen_with_region.groupby(["datetime", "region"])["generation_mw"]
                .sum()
                .reset_index()
            )
            region_demand = (
                day_demand.groupby(["datetime", "region"])["demand_mwh"]
                .sum()
                .reset_index()
            )

            merged = region_gen.merge(
                region_demand,
                on=["datetime", "region"],
                how="outer",
            ).fillna(0)

            merged["excess_mw"] = merged["generation_mw"] - merged["demand_mwh"]

            # Charge buffer — normalize region names so official CCAA names map correctly
            for _, row in merged.iterrows():
                if row["excess_mw"] > 0.1:
                    buffer.charge(
                        region=str(row["region"]),   # charge() normalizes internally
                        excess_mw=float(row["excess_mw"]),
                        duration_hours=1.0,
                    )

        except Exception as e:
            print(f"[SnapshotBuilder] Warning: buffer charge failed ({e}). Starting empty.")
            # Buffer stays empty — HealingAgent will fall back to ramp-up



    def build_predictive_snapshot(self, target_datetime: str, forecast_engine: dict) -> GridState:
        """
        Builds a grid state based on a future prediction rather than historical data.
        It uses the closest historical hour to set up the baseline topology (online generators, lines)
        but replaces regional demand with the AI-predicted demand for that hour.
        """
        import torch
        import numpy as np

        # 1. Build the base snapshot to get the topology
        state = self.build_snapshot(target_datetime)
        
        # 2. Extract date and hour from target_datetime
        dt = pd.to_datetime(target_datetime)
        target_date = dt.strftime("%Y-%m-%d")
        hour = dt.hour
        
        # 3. Get predictions from forecast_engine
        dataset = forecast_engine["dataset"]
        model = forecast_engine["model"]
        date_to_idx = forecast_engine["date_to_idx"]
        hybrid = forecast_engine.get("hybrid", None)
        
        if target_date in date_to_idx:
            idx = date_to_idx[target_date]
        else:
            available = list(date_to_idx.keys())
            if available:
                dts = pd.to_datetime(available)
                closest = int(np.argmin(np.abs((dts - pd.to_datetime(target_date)).total_seconds())))
                idx = date_to_idx[available[closest]]
            else:
                idx = 0
        idx = min(idx, len(dataset) - 1)
        
        x, _, _, _ = dataset[idx]
        x_batch = x.unsqueeze(0).cpu()
        edge_index = dataset.edge_index.cpu()
        edge_weight = dataset.edge_weight.cpu()
        
        with torch.no_grad():
            if hybrid is not None:
                y_pred = hybrid.predict(x_batch, edge_index, edge_weight, target_date=target_date)
            else:
                x_gat = x_batch[:, :, :, :16] if x_batch.shape[-1] > 16 else x_batch
                y_pred = model.cpu()(x_gat, edge_index, edge_weight).squeeze(0)
                
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu()

        # 4. Overwrite demand in state
        def inv_transform(scaled_array, node_idx):
            ns = dataset.scaler.mean.shape[-1]
            d = np.zeros((24, ns))
            d[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(d, node_idx=node_idx)[:, 0]

        NODE_TO_DEMAND_COL = {
            'Andalucía': 'Andalucía',
            'Aragón': 'Aragón',
            'Cantabria': 'Cantabria',
            'Castilla-La Mancha': 'Castilla-La Mancha',
            'Castilla y León': 'Castilla y León',
            'Cataluña': 'Cataluña',
            'País Vasco': 'País Vasco',
            'Principado de Asturias': 'Principado de Asturias',
            'Ceuta': 'Ceuta',
            'Melilla': 'Melilla',
            'Comunidad de Madrid': 'Comunidad de Madrid',
            'Comunidad Foral de Navarra': 'Comunidad Foral de Navarra',
            'Comunidad Valenciana': 'Comunidad Valenciana',
            'Extremadura': 'Extremadura',
            'Galicia': 'Galicia',
            'Islas Baleares': 'Islas Baleares',
            'Canarias': 'Canarias',
            'La Rioja': 'La Rioja',
            'Región de Murcia': 'Región de Murcia'
        }

        # Need to match keys in state.regional_demand_mw which might differ slightly
        # in terms of formatting, but we'll try direct matches or partial matches
        existing_regions = list(state.regional_demand_mw.keys())
        
        for i, node_name in enumerate(dataset.node_names):
            # The model outputs a shape like (Nodes, 24)
            pred_scaled = y_pred[i].numpy()
            pred_mw = inv_transform(pred_scaled, i)
            
            target_mw = max(0.0, float(pred_mw[hour]))
            
            # Find matching region in state
            mapped_name = NODE_TO_DEMAND_COL.get(node_name, node_name)
            
            match = None
            for existing in existing_regions:
                if existing == mapped_name or existing in mapped_name or mapped_name in existing:
                    match = existing
                    break
                    
            if match:
                state.regional_demand_mw[match] = target_mw
            else:
                # Try a fallback mapping (handling potential mojibake/encoding issues)
                for existing in existing_regions:
                    import unicodedata
                    def norm(s):
                        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').replace(' ', '').lower()
                    if norm(existing) == norm(mapped_name):
                        match = existing
                        break
                if match:
                    state.regional_demand_mw[match] = target_mw
                else:
                    state.regional_demand_mw[mapped_name] = target_mw
                    
        # 5. Re-solve DCPF since demands have changed
        from .simulator import ChaosEngine
        engine = ChaosEngine(state)
        engine._solve_dcpf()
        for line_id, line in engine.state.lines.items():
            state.lines[line_id].current_flow_mw = line.current_flow_mw

        # 6. Charge the virtual buffer (same logic as historical snapshot)
        self._charge_virtual_buffer(state, target_str)

        state.timestamp += " (Predicted)"
        print(f"[SnapshotBuilder] Injected predicted demand for {target_datetime} (Hour {hour})")
        print(f"  Virtual buffer: {state.virtual_buffer.total_stored_mwh():.0f} MWh stored")
        return state


    def get_available_dates(self, sample_n: int = 100) -> list:
        """Return a sample of available datetime strings for building snapshots."""
        self._load_data()
        unique_dates = self._gen_data.index.unique()
        if len(unique_dates) > sample_n:
            step = len(unique_dates) // sample_n
            sampled = unique_dates[::step]
        else:
            sampled = unique_dates
        return [str(d) for d in sampled]

    def get_date_range(self) -> tuple:
        """Return (min_date, max_date) of available data."""
        self._load_data()
        return str(self._gen_data.index.min()), str(self._gen_data.index.max())
