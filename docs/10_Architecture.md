---
tags: [architecture, backend, frontend]
---
# Digital Twin Architecture

## Overview
The digital twin is an interactive 3D application that simulates the Spanish electrical grid, including generation sources (nodes), autonomous communities (regions), and power flows (edges). It supports a "Chaos Engine" for real-time resilience testing.

## Backend (FastAPI)
- **File:** `backend/main.py`
- **Port:** `8000`
- **Role:** Maintains the canonical state of the grid (`grid_state`) and runs the ML/DL Hybrid Forecasting Engine. Calculates real-time supply topology using a greedy routing algorithm that pairs available plant capacity with regional demand based on Haversine distance.
- **Endpoints:**
  - `GET /api/topology`: Returns the current nodes (plants) and active routing edges.
  - `GET /api/regions`: Returns the Spain GeoJSON for the frontend.
  - `GET /api/forecast/all-regions`: Triggers inference on the GATv2 model for all 19 regions for a specific date and returns predicted/actual loads.
  - `POST /api/simulate/chaos`: Fails a target power plant (sets status to `offline`), triggers `recalculate_routing()`, and returns the updated topology.

## Frontend (React + DeckGL + Vite)
- **File:** `frontend/src/App.jsx`, `frontend/src/index.css`
- **Port:** `5173`
- **Role:** Real-time visualizer of the backend state.
- **Layers:**
  1. **GeoJsonLayer (`regions-layer`):** Renders Spain's autonomous communities.
  2. **ScatterplotLayer (`supply-radius-layer`):** Renders dynamic green coverage circles (computed on frontend via `maxRadius * 1.6`) around active plants.
  3. **ArcLayer (`flow-arcs-layer`):** Renders power transmission lines. Cyan-to-Green for standard routes, Orange-to-Red for chaos reroutes.
  4. **SimpleMeshLayer (`plants-layer`):** Renders spinning, upside-down 3D cones for power plants. Sizing and color are dynamically driven by the plant's `tech_code`.

## Pipeline
1. `download_entsoe_data.py`: Fetches hourly generation.
2. `backend/main.py`: Loads data, runs routing algorithm, serves API.
3. `frontend/src/App.jsx`: Polls state, visualizes 3D digital twin.
