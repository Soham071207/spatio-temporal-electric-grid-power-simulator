import os
import json
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, field_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ── Structured logging (JSON in prod, pretty in dev) ──────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("api_requests_total",   "Total API requests",   ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Request latency", ["endpoint"])
HEAL_COST       = Histogram("heal_cost_eur",       "Healing cost in EUR per scenario", ["healer_type"])

# ── Thread pool for CPU-bound healing tasks ───────────────────────────────────
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

app = FastAPI(
    title="Spain Grid Digital Twin API",
    description=(
        "Data-driven digital twin of the Spanish electrical grid. "
        "Supports real-time chaos injection, ML-powered self-healing, "
        "cascade simulation, and 24-hour demand forecasting via GNN."
    ),
    version="2.0.0",
    contact={"name": "Digital Twin Team", "email": "twin@example.com"},
    license_info={"name": "MIT"},
)

# ── CORS (open for local dev — lock down origins in production) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import math as _math
import sys as _sys

# ── Project root & forecasting dir on sys.path (module-level, idempotent) ─────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FORECASTING_DIR = os.path.join(_PROJECT_ROOT, "forecasting")
for _p in (_PROJECT_ROOT, _FORECASTING_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# ── Region name mapping: UI/GeoJSON names → dataset node names (module-level) ────
UI_TO_DATASET = {
    'Andalucía': 'Andalucía', 'Aragón': 'Aragón',
    'Principado de Asturias': 'Principado de Asturias',
    'Illes Balears': 'Islas Baleares', 'Canarias': 'Islas Canarias',
    'Cantabria': 'Cantabria', 'Castilla y León': 'Castilla y León',
    'Castilla - La Mancha': 'Castilla la Mancha', 'Cataluña': 'Cataluña',
    'Comunitat Valenciana': 'Comunidad Valenciana', 'Extremadura': 'Extremadura',
    'Galicia': 'Galicia', 'Comunidad de Madrid': 'Comunidad de Madrid',
    'Región de Murcia': 'Región de Murcia',
    'Comunidad Foral de Navarra': 'Comunidad de Navarra',
    'País Vasco': 'País Vasco', 'La Rioja': 'La Rioja',
    'Ceuta': 'Comunidad de Ceuta', 'Melilla': 'Comunidad de Melilla',
}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request count and latency for every endpoint."""
    endpoint = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)
    return response

# Paths to our processed data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
PLANTS_FILE = os.path.join(DATA_DIR, "graphs", "plant_locations.json")
GEOJSON_FILE = os.path.join(DATA_DIR, "graphs", "spain_ccaa.geojson")

# Global state for dynamic digital twin visualization
grid_state = {
    "initialized": False,
    "plants": {},
    "regions": {},
    "edges": []
}

def init_state():
    """Initializes the global grid state with plants and regions."""
    if grid_state["initialized"]:
        return
        
    plants = {}
    if os.path.exists(PLANTS_FILE):
        with open(PLANTS_FILE, 'r', encoding='utf-8') as f:
            plants = json.load(f)
            
    # Filter out plants that failed geocoding and add status
    valid_plants = {}
    for k, v in plants.items():
        if v.get("lat") is not None:
            valid_plants[k] = {**v, "status": "online"}
    grid_state["plants"] = valid_plants
    
    regions = {}
    if os.path.exists(GEOJSON_FILE):
        import shapely.geometry as sg
        with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
            
        for feature in geojson.get("features", []):
            try:
                name = feature["properties"]["name"]
                geom = sg.shape(feature["geometry"])
                centroid = geom.centroid
                regions[name] = {"lon": centroid.x, "lat": centroid.y, "supplied_by": None}
            except Exception:
                pass
    grid_state["regions"] = regions
    
    # Calculate initial edges (plant to nearest region)
    recalculate_routing()
    grid_state["initialized"] = True

def recalculate_routing():
    """Recalculates supply mapping from regions to the nearest online plant."""
    edges = []
    plants = grid_state["plants"]
    regions = grid_state["regions"]
    
    for r_name, r_data in regions.items():
        best_plant = None
        best_dist = float('inf')
        
        for p_id, p_data in plants.items():
            if p_data["status"] == "offline":
                continue # Skip failed plants
                
            # cos-corrected distance (accurate for Spain's latitude band, without full Haversine overhead)
            _mid = _math.radians((p_data['lat'] + r_data['lat']) / 2)
            _dlat = p_data['lat'] - r_data['lat']
            _dlon = (p_data['lon'] - r_data['lon']) * _math.cos(_mid)
            dist = _dlat * _dlat + _dlon * _dlon
            if dist < best_dist:
                best_dist = dist
                best_plant = p_id
                
        if best_plant:
            regions[r_name]["supplied_by"] = best_plant
            edges.append({
                "source": best_plant,
                "target": r_name,
                "source_pos": [plants[best_plant]['lon'], plants[best_plant]['lat']],
                "target_pos": [r_data['lon'], r_data['lat']],
                "is_rerouted": False # Initial routes are not considered rerouted
            })
            
    grid_state["edges"] = edges

@app.get("/", summary="Root health ping")
def read_root():
    return {"status": "ok", "message": "Spain Grid Digital Twin API v2.0 is running"}


@app.get("/api/health", summary="Deep health check", tags=["Ops"])
def health_check():
    """Returns service health and component readiness."""
    components = {
        "snapshot_builder": chaos_v2_state.get("initialized", False),
        "healing_agent":    chaos_v2_state.get("healing_agent") is not None,
        "forecast_engine":  forecast_state.get("initialized", False),
        "chaos_engine_v1":  chaos_state.get("initialized", False),
    }
    all_ok = True  # components initialize lazily on first request — don't report degraded during startup
    return {
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "timestamp": time.time(),
        "version": "2.0.0",
    }


@app.get("/api/metrics", response_class=PlainTextResponse, summary="Prometheus metrics", tags=["Ops"])
def prometheus_metrics():
    """Exposes Prometheus metrics for scraping."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/topology")
def get_topology():
    """Returns the current dynamic topology."""
    if not grid_state["initialized"]:
        init_state()
        
    return {
        "nodes": {
            "plants": grid_state["plants"]
        },
        "edges": grid_state["edges"]
    }

@app.get("/api/regions")
def get_regions():
    """Returns the GeoJSON for the Spanish Autonomous Communities."""
    if not os.path.exists(GEOJSON_FILE):
        raise HTTPException(status_code=404, detail="GeoJSON not found")
        
    with open(GEOJSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.post("/api/simulate/chaos")
async def trigger_chaos(target_id: str):
    """Triggers a failure in the simulation engine and reroutes power."""
    if not grid_state["initialized"]:
        init_state()
        
    if target_id not in grid_state["plants"]:
        raise HTTPException(status_code=404, detail="Plant not found")
        
    # 1. Mark plant offline
    grid_state["plants"][target_id]["status"] = "offline"
    
    # 2. Reroute
    old_edges = { (e["source"], e["target"]) for e in grid_state["edges"] }
    recalculate_routing()
    
    # 3. Mark newly created edges as rerouted
    for edge in grid_state["edges"]:
        if (edge["source"], edge["target"]) not in old_edges:
            edge["is_rerouted"] = True
            
    return {
        "status": "chaos_injected",
        "target": target_id,
        "topology": {
            "nodes": {
                "plants": grid_state["plants"]
            },
            "edges": grid_state["edges"]
        }
    }

# ─────────────────────────────────────────────────────────
#  CHAOS ENGINE API ENDPOINTS
# ─────────────────────────────────────────────────────────

# Global chaos engine state
chaos_state = {
    "engine": None,
    "initial_state": None,
    "initialized": False,
}

def get_chaos_engine():
    """Lazily initialize the chaos engine with a demo grid state."""
    if chaos_state["initialized"]:
        return chaos_state["engine"]

    from digital_twin.simulator import ChaosEngine, build_demo_grid_state

    initial = build_demo_grid_state()
    chaos_state["initial_state"] = initial
    chaos_state["engine"] = ChaosEngine(initial)
    chaos_state["initialized"] = True
    log.info("chaos_engine_v1_init", generators=len(initial.generators))
    return chaos_state["engine"]


class GeneratorFailureRequest(BaseModel):
    generator_id: str

@app.post("/api/chaos/generator")
async def chaos_generator(req: GeneratorFailureRequest):
    """Trip a specific generator offline."""
    try:
        engine = get_chaos_engine()
        event = engine.inject_generator_failure(req.generator_id)
        return {
            "status": "success",
            "event": {"description": event.description, "impact_mw": event.impact_mw},
            "grid_summary": engine.state.summary(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class RenewableShockRequest(BaseModel):
    fuel_type: str
    reduction_pct: float

@app.post("/api/chaos/renewable-shock")
async def chaos_renewable_shock(req: RenewableShockRequest):
    """Reduce all generators of a renewable type by a percentage."""
    try:
        engine = get_chaos_engine()
        event = engine.inject_renewable_shock(req.fuel_type, req.reduction_pct)
        return {
            "status": "success",
            "event": {"description": event.description, "impact_mw": event.impact_mw},
            "grid_summary": engine.state.summary(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class DemandSurgeRequest(BaseModel):
    region: str
    increase_pct: float

@app.post("/api/chaos/demand-surge")
async def chaos_demand_surge(req: DemandSurgeRequest):
    """Spike demand in a specific region."""
    try:
        engine = get_chaos_engine()
        event = engine.inject_demand_surge(req.region, req.increase_pct)
        return {
            "status": "success",
            "event": {"description": event.description, "impact_mw": event.impact_mw},
            "grid_summary": engine.state.summary(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class CascadeRequest(BaseModel):
    max_depth: int = 10

@app.post("/api/chaos/cascade")
async def chaos_cascade(req: CascadeRequest):
    """Run a full cascading failure simulation on the current grid state."""
    from digital_twin.cascade import CascadeEngine

    engine = get_chaos_engine()
    cascade = CascadeEngine(engine)
    result = cascade.run_cascade(
        initial_trigger="API-triggered cascade",
        max_depth=req.max_depth,
    )
    return {
        "status": "success",
        "cascade": cascade.to_dict(result),
        "grid_summary": engine.state.summary(),
    }


class RandomChaosRequest(BaseModel):
    n_failures: int = 1

@app.post("/api/chaos/random")
async def chaos_random(req: RandomChaosRequest):
    """Random chaos monkey — trip N random components."""
    engine = get_chaos_engine()
    events = engine.run_random_chaos(n_failures=req.n_failures)
    return {
        "status": "success",
        "events": [{"description": e.description, "impact_mw": e.impact_mw} for e in events],
        "grid_summary": engine.state.summary(),
    }


@app.post("/api/chaos/reset")
async def chaos_reset():
    """Restore the grid to its initial pre-chaos state."""
    engine = get_chaos_engine()
    engine.reset()
    return {
        "status": "reset",
        "grid_summary": engine.state.summary(),
    }


@app.get("/api/resilience")
async def get_resilience():
    """Calculate the current grid resilience score with full breakdown."""
    from digital_twin.metrics import calculate_resilience_score

    engine = get_chaos_engine()
    initial = chaos_state["initial_state"]
    
    # Get cascade info if available
    cascade_depth = 0
    cascade_size = 0
    geo_spread = 0
    if engine.cascade_log:
        last = engine.cascade_log[-1]
        cascade_depth = last.get("cascade_depth", 0)
        cascade_size = last.get("cascade_size", 0)
    
    score = calculate_resilience_score(
        initial, engine.state,
        cascade_depth=cascade_depth,
        cascade_size=cascade_size,
        geographic_spread=geo_spread,
    )
    return {
        "resilience": score,
        "grid_summary": engine.state.summary(),
        "event_log": engine.get_event_log(),
    }


class WhatIfRequest(BaseModel):
    scenario_type: str  # "renewable_shock", "demand_surge", "generator_failure"
    target: str = ""
    value: float = 50.0

@app.post("/api/whatif")
async def what_if(req: WhatIfRequest):
    """Compare grid state before and after a scenario without permanently changing state."""
    import copy
    from digital_twin.simulator import ChaosEngine
    from digital_twin.metrics import calculate_resilience_score
    from digital_twin.redispatch import RedispatchEngine

    engine = get_chaos_engine()
    
    # Create a temporary copy to run the what-if on
    temp_engine = ChaosEngine(copy.deepcopy(engine.state))
    before = temp_engine.state.summary()
    
    # Run the scenario on the temp copy
    if req.scenario_type == "renewable_shock":
        temp_engine.inject_renewable_shock(req.target or "Wind", req.value)
    elif req.scenario_type == "demand_surge":
        temp_engine.inject_demand_surge(req.target or "Madrid", req.value)
    elif req.scenario_type == "generator_failure":
        temp_engine.inject_generator_failure(req.target)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown scenario type: {req.scenario_type}")
    
    after = temp_engine.state.summary()
    
    # Calculate resilience for both states
    initial = chaos_state["initial_state"]
    score_before = calculate_resilience_score(initial, engine.state)
    score_after = calculate_resilience_score(initial, temp_engine.state)
    
    # Estimate redispatch cost
    redispatch = RedispatchEngine()
    cost_estimate = redispatch.estimate_redispatch_cost(temp_engine.state)
    
    return {
        "before": before,
        "after": after,
        "resilience_before": score_before,
        "resilience_after": score_after,
        "redispatch_cost": cost_estimate,
        "deltas": {
            k: round(after[k] - before[k], 2) 
            for k in before 
            if isinstance(before[k], (int, float))
        },
    }


# Forecast Engine Global State
forecast_state = {
    "model": None,
    "dataset": None,
    "date_to_idx": {},
    "initialized": False
}
forecast_engine_lock = threading.Lock()

def get_forecast_engine():
    if forecast_state["initialized"]:
        return forecast_state
        
    with forecast_engine_lock:
        # Double-check inside lock
        if forecast_state["initialized"]:
            return forecast_state
            
        import torch
        from dataset import SpainElectricityDataset
        from gnn_regional import GATForecaster
        
        print("[Backend] Initializing Regional GAT Forecasting Engine...", flush=True)
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_root, "data")
        dataset = SpainElectricityDataset(split="test", year_start=2020, year_end=2027, data_dir=data_dir)
    
        in_channels = 17  # regional checkpoint was trained on 17-channel input
        hidden_channels = 64
        pred_horizon = 24
    
        base_dir = os.path.dirname(os.path.dirname(__file__))
        regional_path = os.path.join(base_dir, "data", "models", "best_gat_regional.pth")
    
        model_regional = None
        if os.path.exists(regional_path):
            try:
                m = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
                ck = torch.load(regional_path, map_location="cpu")
                m.load_state_dict(ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck)
                m.eval()
                model_regional = m
                print(f"[Backend] Loaded regional model from {regional_path}")
            except Exception as e:
                print(f"[Backend] Could not load regional model: {e}")
    
        if model_regional is not None:
            forecast_state["model"] = model_regional
            forecast_state["hybrid"] = None
            print("[Backend] Running with regional model (no ensemble)")
        else:
            print("[Backend] WARNING: No model weights found!")
    
        # Precompute date to index lookup
        date_to_idx = {}
        if hasattr(dataset, "time_idx"):
            for idx in range(len(dataset)):
                dt = pd.to_datetime(dataset.time_idx[idx + dataset.seq_length])
                d_str = dt.strftime("%Y-%m-%d")
                if d_str not in date_to_idx:
                    date_to_idx[d_str] = idx
                
        forecast_state["dataset"] = dataset
        forecast_state["date_to_idx"] = date_to_idx
        forecast_state["initialized"] = True
        print(f"[Backend] Forecasting engine ready with {len(date_to_idx)} test dates.")
    return forecast_state

@app.get("/api/forecast")
def get_forecast(date: str, region: str):
    """
    Returns 24-hour forecast vs actual data for a specific date and region
    using the trained 15-channel ML/DL Hybrid GAT forecaster.
    """
    try:
        import torch
        import numpy as np
        import pandas as pd
        
        engine = get_forecast_engine()
        dataset = engine["dataset"]
        model = engine["model"]
        date_to_idx = engine["date_to_idx"]
        
        dataset_node = UI_TO_DATASET.get(region, region)
        region_idx = 0
        for i, name in enumerate(dataset.node_names):
            if name == dataset_node:
                region_idx = i
                break
            
        # Find index for date or closest available
        if date in date_to_idx:
            idx = date_to_idx[date]
        else:
            # Fallback to index based on hash or latest available
            if len(date_to_idx) > 0:
                available_dates = list(date_to_idx.keys())
                # Pick closest date
                try:
                    target_dt = pd.to_datetime(date)
                    dts = pd.to_datetime(available_dates)
                    closest_idx = np.argmin(np.abs((dts - target_dt).total_seconds()))
                    idx = date_to_idx[available_dates[closest_idx]]
                except Exception:
                    idx = len(dataset) // 2
            else:
                idx = 0
                
        idx = min(idx, len(dataset) - 1)
        
        x, y_true, y_holiday, _ = dataset[idx]
        x_batch = x.unsqueeze(0)
        
        with torch.no_grad():
            hybrid = engine.get("hybrid", None)
            if hybrid is not None:
                # ML/DL Hybrid: GBT corrects GAT raw outputs
                y_pred = hybrid.predict(x_batch, dataset.edge_index, dataset.edge_weight, target_date=date)
            else:
                x_gat = x_batch[:, :, :, :17] if x_batch.shape[-1] > 17 else x_batch
                y_pred = model(x_gat, dataset.edge_index, dataset.edge_weight).squeeze(0)
            
        # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand) for inv_transform
        actual_scaled = y_true[region_idx, :, 0].numpy()
        # y_pred from gnn_regional: (N, PredHorizon, 2) — take channel 0 (demand)
        _yp = y_pred[region_idx]
        pred_scaled = (_yp[:, 0] if _yp.ndim == 2 else _yp).numpy()
        
        # Inverse transform using dataset scaler
        def inv_transform(scaled_array, node_idx):
            ns = dataset.scaler.mean.shape[-1]
            dummy = np.zeros((24, ns))
            dummy[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]
            
        actual_mw = inv_transform(actual_scaled, region_idx)
        pred_mw = inv_transform(pred_scaled, region_idx)
        
        # Build hourly response
        hourly_data = []
        for h in range(24):
            hourly_data.append({
                "hour": h,
                "actual_mw": round(float(max(0, actual_mw[h])), 1),
                "predicted_mw": round(float(max(0, pred_mw[h])), 1)
            })
            
        return {
            "region": dataset.node_names[region_idx],
            "date": date,
            "hourly_data": hourly_data,
        }
    except Exception as e:
        print(f"Error in forecast API: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/hover")
def get_forecast_hover(date: str, region: str):
    """Quick hover prediction — returns next hour + next day aggregate MW."""
    try:
        import torch
        import numpy as np
        
        engine = get_forecast_engine()
        dataset = engine["dataset"]
        model = engine["model"]
        date_to_idx = engine["date_to_idx"]
        
        dataset_node = UI_TO_DATASET.get(region, region)
        region_idx = 0
        for i, name in enumerate(dataset.node_names):
            if name == dataset_node:
                region_idx = i
                break
        
        if date in date_to_idx:
            idx = date_to_idx[date]
        else:
            available = list(date_to_idx.keys())
            if available:
                target_dt = pd.to_datetime(date)
                dts = pd.to_datetime(available)
                closest = int(np.argmin(np.abs((dts - target_dt).total_seconds())))
                idx = date_to_idx[available[closest]]
            else:
                idx = 0
        idx = min(idx, len(dataset) - 1)
        
        x, y_true, _, _ = dataset[idx]
        # Force everything to CPU to avoid cuda/cpu device mismatch
        x_batch = x.unsqueeze(0).cpu()
        edge_index = dataset.edge_index.cpu()
        edge_weight = dataset.edge_weight.cpu()
        
        with torch.no_grad():
            hybrid = engine.get("hybrid", None)
            if hybrid is not None:
                # Use hybrid corrector (same path as /api/forecast)
                y_pred = hybrid.predict(x_batch, edge_index, edge_weight, target_date=date)
            else:
                model_cpu = model.cpu() if model is not None else None
                if model_cpu is None:
                    raise RuntimeError("No model loaded")
                x_gat = x_batch[:, :, :, :17] if x_batch.shape[-1] > 17 else x_batch
                y_pred = model_cpu(x_gat, edge_index, edge_weight).squeeze(0)
        
        # Ensure output is on CPU
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu()
        
        # y_pred from gnn_regional: (N, PredHorizon, 2) — take channel 0 (demand)
        _yp = y_pred[region_idx]
        pred_scaled = (_yp[:, 0] if _yp.ndim == 2 else _yp).numpy()
        # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand) for inv_transform
        actual_scaled = y_true[region_idx, :, 0].numpy()
        
        def inv_transform(arr, node_idx):
            ns = dataset.scaler.mean.shape[-1]
            d = np.zeros((24, ns))
            d[:, 0] = arr
            return dataset.scaler.inverse_transform(d, node_idx=node_idx)[:, 0]
        
        pred_mw = inv_transform(pred_scaled, region_idx)
        actual_mw = inv_transform(actual_scaled, region_idx)
        
        return {
            "region": dataset.node_names[region_idx],
            "next_hour_predicted_mw": round(float(max(0, pred_mw[0])), 1),
            "next_hour_actual_mw": round(float(max(0, actual_mw[0])), 1),
            "next_day_predicted_mw": round(float(sum(max(0, v) for v in pred_mw)), 1),
            "next_day_actual_mw": round(float(sum(max(0, v) for v in actual_mw)), 1),
            "peak_predicted_mw": round(float(max(0, float(max(pred_mw)))), 1),
        }
    except Exception as e:
        print(f"Hover forecast error: {e}")
        import traceback; traceback.print_exc()
        return {"region": region, "next_hour_predicted_mw": 0, "next_day_predicted_mw": 0, "peak_predicted_mw": 0}


@app.get("/api/forecast/weekend")
def get_forecast_weekend(date: str, region: str):
    """Returns 48-hour forecast (Sat + Sun) for the weekend containing the given date."""
    try:
        import torch
        import numpy as np
        
        engine = get_forecast_engine()
        dataset = engine["dataset"]
        model = engine["model"]
        date_to_idx = engine["date_to_idx"]
        
        dataset_node = UI_TO_DATASET.get(region, region)
        region_idx = 0
        for i, name in enumerate(dataset.node_names):
            if name == dataset_node:
                region_idx = i
                break
        
        # Find Saturday of the week containing this date
        target_dt = pd.to_datetime(date)
        days_to_sat = (5 - target_dt.weekday()) % 7
        if target_dt.weekday() == 6:  # Sunday
            saturday = target_dt - pd.Timedelta(days=1)
        else:
            saturday = target_dt + pd.Timedelta(days=days_to_sat)
        sunday = saturday + pd.Timedelta(days=1)
        
        sat_str = saturday.strftime("%Y-%m-%d")
        sun_str = sunday.strftime("%Y-%m-%d")
        
        # Force graph tensors to CPU once (they may be on CUDA if hybrid init moved them)
        edge_index = dataset.edge_index.cpu()
        edge_weight = dataset.edge_weight.cpu()
        hybrid = engine.get("hybrid", None)
        
        def get_day_data(day_str):
            if day_str in date_to_idx:
                idx = date_to_idx[day_str]
            else:
                available = list(date_to_idx.keys())
                if available:
                    dt = pd.to_datetime(day_str)
                    dts = pd.to_datetime(available)
                    closest = int(np.argmin(np.abs((dts - dt).total_seconds())))
                    idx = date_to_idx[available[closest]]
                else:
                    idx = 0
            idx = min(idx, len(dataset) - 1)
            
            x, y_true, _, _ = dataset[idx]
            x_batch = x.unsqueeze(0).cpu()
            
            with torch.no_grad():
                if hybrid is not None:
                    y_pred = hybrid.predict(x_batch, edge_index, edge_weight, target_date=day_str)
                else:
                    x_gat = x_batch[:, :, :, :17] if x_batch.shape[-1] > 17 else x_batch
                    y_pred = model.cpu()(x_gat, edge_index, edge_weight).squeeze(0)
                # gnn_regional outputs (N, PredHorizon, 2) — flatten to (N, PredHorizon) demand only
                if y_pred.ndim == 3:
                    y_pred = y_pred[:, :, 0]
            
            if hasattr(y_pred, 'cpu'):
                y_pred = y_pred.cpu()
            
            pred_scaled = y_pred[region_idx].numpy()
            # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
            actual_scaled = y_true[region_idx, :, 0].numpy()
            
            def inv_transform(arr, node_idx):
                ns = dataset.scaler.mean.shape[-1]
                d = np.zeros((24, ns))
                d[:, 0] = arr
                return dataset.scaler.inverse_transform(d, node_idx=node_idx)[:, 0]
            
            return inv_transform(actual_scaled, region_idx), inv_transform(pred_scaled, region_idx)
        
        sat_actual, sat_pred = get_day_data(sat_str)
        sun_actual, sun_pred = get_day_data(sun_str)
        
        hourly_data = []
        for h in range(24):
            hourly_data.append({
                "hour": f"Sat {h}h",
                "actual_mw": round(float(max(0, sat_actual[h])), 1),
                "predicted_mw": round(float(max(0, sat_pred[h])), 1)
            })
        for h in range(24):
            hourly_data.append({
                "hour": f"Sun {h}h",
                "actual_mw": round(float(max(0, sun_actual[h])), 1),
                "predicted_mw": round(float(max(0, sun_pred[h])), 1)
            })
        
        return {
            "region": dataset.node_names[region_idx],
            "saturday": sat_str,
            "sunday": sun_str,
            "hourly_data": hourly_data,
        }
    except Exception as e:
        print(f"Weekend forecast error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/dates")
def get_forecast_dates():
    """
    Returns the sorted list of all available test-set dates that the model
    can generate forecasts for. Used by the DateNavigator in the frontend.
    """
    try:
        engine = get_forecast_engine()
        dates = sorted(engine["date_to_idx"].keys())
        return {"dates": dates, "count": len(dates)}
    except Exception as e:
        print(f"Dates endpoint error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/all-regions")
def get_all_regions_forecast(date: str):
    """
    Runs a single forward pass for the given date and returns predicted peak MW
    and total predicted MWh for every region. Used to colour the Spain map
    as a heat-map of predicted energy demand.
    """
    try:
        import torch
        import numpy as np
        import pandas as pd

        engine = get_forecast_engine()
        dataset = engine["dataset"]
        model = engine["model"]
        date_to_idx = engine["date_to_idx"]

        # Resolve dataset index
        if date in date_to_idx:
            idx = date_to_idx[date]
        else:
            available = list(date_to_idx.keys())
            if available:
                target_dt = pd.to_datetime(date)
                dts = pd.to_datetime(available)
                closest = int(np.argmin(np.abs((dts - target_dt).total_seconds())))
                idx = date_to_idx[available[closest]]
            else:
                idx = 0
        idx = min(idx, len(dataset) - 1)

        x, y_true, _, _ = dataset[idx]
        x_batch = x.unsqueeze(0)

        with torch.no_grad():
            hybrid = engine.get("hybrid", None)
            if hybrid is not None:
                y_pred = hybrid.predict(x_batch, dataset.edge_index, dataset.edge_weight, target_date=date)
            else:
                x_gat = x_batch[:, :, :, :17] if x_batch.shape[-1] > 17 else x_batch
                y_pred = model(x_gat, dataset.edge_index, dataset.edge_weight).squeeze(0)
                # gnn_regional outputs (N, PredHorizon, 2) — flatten to (N, PredHorizon) demand only
                if y_pred.ndim == 3:
                    y_pred = y_pred[:, :, 0]

        # Node names match dataset order
        NODE_NAMES = dataset.node_names  # list of 19 region strings

        # Reverse: dataset_name -> node_idx (UI_TO_DATASET is the module-level mapping)
        name_to_idx = {n: i for i, n in enumerate(NODE_NAMES)}

        def inv_transform(scaled_array, node_idx):
            ns = dataset.scaler.mean.shape[-1]
            dummy = np.zeros((24, ns))
            dummy[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]

        result = {}
        for ui_name, dataset_name in UI_TO_DATASET.items():
            node_idx = name_to_idx.get(dataset_name)
            if node_idx is None:
                result[ui_name] = {"peak_predicted_mw": 0, "total_predicted_mwh": 0, "peak_actual_mw": 0}
                continue

            pred_scaled = y_pred[node_idx].numpy()
            # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
            actual_scaled = y_true[node_idx, :, 0].numpy()

            pred_mw = inv_transform(pred_scaled, node_idx)
            actual_mw = inv_transform(actual_scaled, node_idx)

            result[ui_name] = {
                "peak_predicted_mw": round(float(max(0, float(np.max(pred_mw)))), 1),
                "total_predicted_mwh": round(float(sum(max(0, float(v)) for v in pred_mw)), 1),
                "peak_actual_mw": round(float(max(0, float(np.max(actual_mw)))), 1),
            }

        return {"date": date, "regions": result}
    except Exception as e:
        print(f"All-regions forecast error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
#  CHAOS ENGINE V2 — Data-Driven Self-Healing Endpoints
# ─────────────────────────────────────────────────────────

chaos_v2_state = {
    "snapshot_builder": None,
    "healing_agent": None,
    "scenario_engine": None,
    "current_snapshot": None,
    "initialized": False,
}

def get_chaos_v2():
    """Lazily initialize the v2 chaos engine components."""
    if chaos_v2_state["initialized"]:
        return chaos_v2_state

    import sys
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, project_root)

    from digital_twin.snapshot_builder import SnapshotBuilder
    from digital_twin.scenario_engine import ScenarioEngine
    from digital_twin.healing_agent import HealingAgent

    chaos_v2_state["snapshot_builder"] = SnapshotBuilder(
        data_dir=os.path.join(project_root, "data")
    )
    import time
    chaos_v2_state["scenario_engine"] = ScenarioEngine(seed=int(time.time()))
    chaos_v2_state["healing_agent"] = HealingAgent()

    # Load trained healing model
    model_path = os.path.join(project_root, "data", "models", "healing_agent.pkl")
    if os.path.exists(model_path):
        chaos_v2_state["healing_agent"].load(model_path)
        log.info("healing_agent_loaded", path=model_path)
    else:
        log.warning("healing_agent_not_found", path=model_path)

    chaos_v2_state["initialized"] = True
    log.info("chaos_v2_initialized")
    return chaos_v2_state


class SnapshotRequest(BaseModel):
    datetime: str = "2025-06-15 14:00:00"

@app.get("/api/chaos/v2/snapshot")
async def chaos_v2_snapshot(datetime: str = "2025-06-15 14:00:00"):
    """Build a real data-calibrated grid state for a specific datetime."""
    try:
        state_dict = get_chaos_v2()
        sb = state_dict["snapshot_builder"]
        grid_state = sb.build_snapshot(datetime)
        chaos_v2_state["current_snapshot"] = grid_state
        return {
            "status": "success",
            "datetime": datetime,
            "grid_summary": grid_state.summary(),
            "generators": len(grid_state.generators),
            "transmission_lines": len(grid_state.lines),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RunScenarioRequest(BaseModel):
    datetime: str = "2025-06-15 14:00:00"
    scenario_type: str = "single_gen_failure"
    params: dict = {}

    @field_validator("scenario_type")
    @classmethod
    def validate_scenario_type(cls, v: str) -> str:
        valid = {"single_gen_failure", "n_of_k_failure", "renewable_collapse",
                 "demand_spike", "storm_regional", "compound", "heatwave"}
        if v not in valid:
            raise ValueError(f"scenario_type must be one of: {sorted(valid)}")
        return v

    @field_validator("datetime")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"datetime must be a valid ISO timestamp, got: {v!r}")
        return v

@app.post("/api/chaos/v2/run-scenario")
async def chaos_v2_run_scenario(req: RunScenarioRequest):
    """Run a specific chaos scenario on real grid data."""
    import copy
    try:
        state_dict = get_chaos_v2()
        sb = state_dict["snapshot_builder"]
        se = state_dict["scenario_engine"]

        grid_state = sb.build_snapshot(req.datetime)
        result = se.run_scenario(grid_state, req.scenario_type, req.params)

        return {
            "status": "success",
            "scenario_id": result.scenario_id,
            "scenario_type": result.scenario_type,
            "description": result.description,
            "resilience_after_fault": result.resilience_score_after_fault,
            "resilience_after_healing": result.resilience_score_after_healing,
            "generation_lost_mw": result.generation_lost_mw,
            "unserved_energy_mw": result.unserved_energy_mw,
            "healing_cost_eur": result.healing_cost_eur,
            "cascade_depth": result.cascade_depth,
            "cascade_size": result.cascade_size,
            "geographic_spread": result.geographic_spread,
            "frequency_deviation_hz": result.frequency_deviation_hz,
            "initial_state": result.initial_state_summary,
            "post_fault_state": result.post_fault_summary,
            "post_healing_state": result.post_healing_summary,
            "cascade_history": result.cascade_history if hasattr(result, "cascade_history") else [],
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chaos/v2/heal")
async def chaos_v2_heal(req: RunScenarioRequest):
    """
    Run a scenario, then compare Rule-Based vs AI Healing.
    This is the key IEEE comparison endpoint.
    """
    import copy
    try:
        state_dict = get_chaos_v2()
        sb = state_dict["snapshot_builder"]
        se = state_dict["scenario_engine"]
        ha = state_dict["healing_agent"]

        # Build the snapshot
        grid_state = sb.build_snapshot(req.datetime)
        initial_summary = grid_state.summary()

        # Fix #1: Apply time-of-day renewable capacity factors
        # Solar plants output 0 at night; wind follows diurnal patterns based on REE data.
        from digital_twin.simulator import ChaosEngine as _ChaosEngine
        try:
            sim_hour = int(pd.Timestamp(req.datetime).hour)
            _tmp_engine = _ChaosEngine(grid_state)
            _tmp_engine.apply_time_of_day_capacity_factors(sim_hour)
            grid_state = _tmp_engine.state
        except Exception as _tod_err:
            print(f"[ToD CF] Warning: {_tod_err}")

        initial_summary = grid_state.summary()

        # --- Bug 3 Fix: Inject fault into a SINGLE copy, then deepcopy AFTER injection ---
        # This guarantees both healing branches receive the IDENTICAL post-fault state.
        from digital_twin.simulator import ChaosEngine
        from digital_twin.cascade import CascadeEngine
        from digital_twin.redispatch import RedispatchEngine
        from digital_twin.metrics import calculate_resilience_score

        # 1. Inject fault once
        engine_fault = ChaosEngine(grid_state)
        desc = se._inject_scenario(engine_fault, req.scenario_type, req.params)
        post_inject_state = copy.deepcopy(engine_fault.state)

        # 2. Two independent cascade simulations from identical post-inject state
        engine_rules = ChaosEngine(copy.deepcopy(post_inject_state))
        engine_ai    = ChaosEngine(copy.deepcopy(post_inject_state))
        engine_rules.event_log = list(engine_fault.event_log)
        engine_ai.event_log    = list(engine_fault.event_log)

        # Run cascades on both
        ce_rules = CascadeEngine(engine_rules)
        res_rules = ce_rules.run_cascade(initial_trigger=desc, max_depth=8)

        ce_ai = CascadeEngine(engine_ai)
        res_ai = ce_ai.run_cascade(initial_trigger=desc, max_depth=8)

        post_fault_summary = engine_rules.state.summary()

        # ── Rule-based healing (CPU-bound — run in thread pool) ──────────────
        redispatch = RedispatchEngine()
        loop = asyncio.get_event_loop()
        rule_result = await loop.run_in_executor(
            _EXECUTOR, redispatch.execute_redispatch, engine_rules.state
        )
        rule_resilience = calculate_resilience_score(grid_state, engine_rules.state)
        HEAL_COST.labels("rule_based").observe(rule_result.get("total_cost_eur", 0))

        # ── AI healing (CPU-bound — run in thread pool) ───────────────────────
        if ha.is_trained:
            ai_result = await loop.run_in_executor(
                _EXECUTOR, ha.apply_healing, engine_ai.state
            )
            ai_resilience = calculate_resilience_score(grid_state, engine_ai.state)
            HEAL_COST.labels("ai_healing").observe(
                sum(a.get("cost_eur", 0) for a in ai_result.get("actions", []))
            )
        else:
            ai_result = {"status": "AI model not trained", "actions": [],
                         "buffer_dispatched_mw": 0, "buffer_summary": {}}
            ai_resilience = {"score": 0}

        return {
            "status": "success",
            "fault_description": desc,
            "initial_state": initial_summary,
            "post_fault_state": post_fault_summary,
            "cascade_history_rules": ce_rules.to_dict(res_rules).get("steps", []),
            "buffer_summary": ai_result.get("buffer_summary", {}),
            "buffer_dispatched_mw": ai_result.get("buffer_dispatched_mw", 0),
            "comparison": {
                "rule_based": {
                    "resilience_score": rule_resilience["score"],
                    "deficit_after_mw": rule_result.get("deficit_after_mw", 0),
                    "cost_eur": rule_result.get("total_cost_eur", 0),
                    "actions": rule_result.get("actions", []),
                    "post_healing_state": engine_rules.state.summary(),
                },
                "ai_healing": {
                    "resilience_score": ai_resilience["score"],
                    "deficit_after_mw": ai_result.get("post_healing_deficit_mw", 0),
                    "predicted_cost_eur": ai_result.get("predicted_cost_eur", 0),
                    "actions": ai_result.get("actions", []),
                    "post_healing_state": engine_ai.state.summary(),
                },
            },
            "cascade_history": ce_ai.to_dict(res_ai).get("steps", [])
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chaos/v2/predictive-heal")
async def chaos_v2_predictive_heal(req: RunScenarioRequest):
    """
    Run a scenario, then compare Rule-Based vs AI Healing on a *predicted* future grid state.
    """
    import copy
    try:
        # Get both forecasting engine and chaos engine
        forecast_state = get_forecast_engine()
        state_dict = get_chaos_v2()
        
        sb = state_dict["snapshot_builder"]
        se = state_dict["scenario_engine"]
        ha = state_dict["healing_agent"]

        # Build the PREDICTIVE snapshot
        grid_state = sb.build_predictive_snapshot(req.datetime, forecast_state)
        initial_summary = grid_state.summary()

        from digital_twin.simulator import ChaosEngine
        from digital_twin.cascade import CascadeEngine
        from digital_twin.redispatch import RedispatchEngine
        from digital_twin.metrics import calculate_resilience_score

        # --- Bug 3 Fix: Inject fault once, then deepcopy ---
        engine_fault = ChaosEngine(grid_state)
        desc = se._inject_scenario(engine_fault, req.scenario_type, req.params)
        post_inject_state = copy.deepcopy(engine_fault.state)

        engine_rules = ChaosEngine(copy.deepcopy(post_inject_state))
        engine_ai = ChaosEngine(copy.deepcopy(post_inject_state))
        engine_rules.event_log = list(engine_fault.event_log)
        engine_ai.event_log = list(engine_fault.event_log)

        # Run cascades on both
        ce_rules = CascadeEngine(engine_rules)
        res_rules = ce_rules.run_cascade(initial_trigger=desc, max_depth=8)
        
        ce_ai = CascadeEngine(engine_ai)
        res_ai = ce_ai.run_cascade(initial_trigger=desc, max_depth=8)

        post_fault_summary = engine_rules.state.summary()

        # ── Rule-based healing ──
        redispatch = RedispatchEngine()
        rule_result = redispatch.execute_redispatch(engine_rules.state)
        rule_resilience = calculate_resilience_score(grid_state, engine_rules.state)

        # ── AI healing ──
        if ha.is_trained:
            ai_result = ha.apply_healing(engine_ai.state)
            ai_resilience = calculate_resilience_score(grid_state, engine_ai.state)
        else:
            ai_result = {"status": "AI model not trained", "actions": [], "buffer_dispatched_mw": 0, "buffer_summary": {}}
            ai_resilience = {"score": 0}

        return {
            "status": "success",
            "fault_description": desc,
            "initial_state": initial_summary,
            "post_fault_state": post_fault_summary,
            "cascade_history_rules": ce_rules.to_dict(res_rules).get("steps", []),
            "buffer_summary": ai_result.get("buffer_summary", {}),
            "buffer_dispatched_mw": ai_result.get("buffer_dispatched_mw", 0),
            "comparison": {
                "rule_based": {
                    "resilience_score": rule_resilience["score"],
                    "deficit_after_mw": rule_result.get("deficit_after_mw", 0),
                    "cost_eur": rule_result.get("total_cost_eur", 0),
                    "actions": rule_result.get("actions", []),
                    "post_healing_state": engine_rules.state.summary(),
                },
                "ai_healing": {
                    "resilience_score": ai_resilience["score"],
                    "deficit_after_mw": ai_result.get("post_healing_deficit_mw", 0),
                    "predicted_cost_eur": ai_result.get("predicted_cost_eur", 0),
                    "actions": ai_result.get("actions", []),
                    "post_healing_state": engine_ai.state.summary(),
                },
            },
            "cascade_history": ce_ai.to_dict(res_ai).get("steps", [])
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




class BatchSimRequest(BaseModel):
    datetime: str = "2025-06-15 14:00:00"
    n_scenarios: int = 50

    @field_validator("n_scenarios")
    @classmethod
    def validate_n_scenarios(cls, v: int) -> int:
        if v < 1 or v > 500:
            raise ValueError("n_scenarios must be between 1 and 500")
        return v

@app.post("/api/chaos/v2/batch-simulate")
async def chaos_v2_batch(req: BatchSimRequest):
    """Run N random scenarios and return aggregate resilience statistics."""
    try:
        state_dict = get_chaos_v2()
        sb = state_dict["snapshot_builder"]
        se = state_dict["scenario_engine"]

        grid_state = sb.build_snapshot(req.datetime)
        results = se.generate_random_scenarios(grid_state, n_scenarios=req.n_scenarios)

        if not results:
            return {"status": "error", "message": "No scenarios generated"}

        scores_fault = [r.resilience_score_after_fault for r in results]
        scores_healed = [r.resilience_score_after_healing for r in results]
        costs = [r.healing_cost_eur for r in results]
        gen_lost = [r.generation_lost_mw for r in results]

        import numpy as np
        return {
            "status": "success",
            "n_scenarios": len(results),
            "datetime": req.datetime,
            "grid_summary": grid_state.summary(),
            "statistics": {
                "resilience_after_fault": {
                    "mean": round(float(np.mean(scores_fault)), 1),
                    "min": round(float(np.min(scores_fault)), 1),
                    "max": round(float(np.max(scores_fault)), 1),
                    "std": round(float(np.std(scores_fault)), 1),
                },
                "resilience_after_healing": {
                    "mean": round(float(np.mean(scores_healed)), 1),
                    "min": round(float(np.min(scores_healed)), 1),
                    "max": round(float(np.max(scores_healed)), 1),
                },
                "generation_lost_mw": {
                    "mean": round(float(np.mean(gen_lost)), 1),
                    "max": round(float(np.max(gen_lost)), 1),
                },
                "healing_cost_eur": {
                    "mean": round(float(np.mean(costs)), 0),
                    "max": round(float(np.max(costs)), 0),
                },
            },
            "scenario_breakdown": {
                stype: len([r for r in results if r.scenario_type == stype])
                for stype in set(r.scenario_type for r in results)
            },
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chaos/v2/worst-case")
async def chaos_v2_worst_case(datetime: str = "2025-06-15 14:00:00", n_search: int = 100):
    """Run adversarial search to find the worst-case scenario for a given grid snapshot."""
    try:
        state_dict = get_chaos_v2()
        sb = state_dict["snapshot_builder"]
        se = state_dict["scenario_engine"]

        grid_state = sb.build_snapshot(datetime)
        results = se.generate_random_scenarios(grid_state, n_scenarios=n_search)

        if not results:
            return {"status": "error", "message": "No scenarios generated"}

        # Find the worst case (lowest resilience after fault)
        worst = min(results, key=lambda r: r.resilience_score_after_fault)

        return {
            "status": "success",
            "datetime": datetime,
            "n_searched": len(results),
            "worst_case": {
                "scenario_type": worst.scenario_type,
                "description": worst.description,
                "resilience_score": worst.resilience_score_after_fault,
                "generation_lost_mw": worst.generation_lost_mw,
                "cascade_depth": worst.cascade_depth,
                "cascade_size": worst.cascade_size,
                "geographic_spread": worst.geographic_spread,
                "frequency_deviation_hz": worst.frequency_deviation_hz,
                "healing_cost_eur": worst.healing_cost_eur,
                "post_fault_state": worst.post_fault_summary,
            },
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
