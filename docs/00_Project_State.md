---
tags: [dashboard, moc, index]
---
# Project State Dashboard

This file serves as the single source of truth for the AI assistant to instantly understand the current state of the project without needing to read the entire chat history.

## Current Status
- **Phase:** ✅ COMPLETE — All core systems built, trained, and integrated.
- **Data Acquisition:**
  - ENTSO-E and REE hourly generation/demand data successfully integrated for 11 years (2015-2026).
- **Backend:**
  - FastAPI server (`backend/main.py`, ~1350 lines) — fully production-ready.
  - Endpoints: `/api/topology`, `/api/regions`, `/api/forecast`, `/api/forecast/hover`, `/api/forecast/all-regions`, `/api/forecast/weekend`, `/api/forecast/dates`, `/api/chaos/*` (v1 + v2), `/api/resilience`, `/api/whatif`, `/api/chaos/v2/heal`, `/api/chaos/v2/batch-simulate`, `/api/chaos/v2/worst-case`.
  - Prometheus metrics at `/api/metrics`, Structlog JSON logging.
  - Async thread-pool for CPU-bound healing tasks.
- **Frontend:**
  - React + Vite + DeckGL frontend (`frontend/src/`).
  - Premium dark-blue UI — Spain map heatmap, 24h + 48h forecast charts.
  - **Tabbed right panel: Forecast | Chaos Engine** — ChaosDashboard fully integrated as sidebar panel (AI vs Rules comparison, resilience gauges, 7 scenario types).
  - Full-screen 3D ChaosGameSimulator overlay via "3D Simulator" button.
- **ML/DL Stack:**
  - GATv2 + LSTM hybrid (16-channel, 3 specialist models: asymmetric/holiday/weekend).
  - Regional GATForecaster (17-channel, demand + excess generation dual output).
  - HybridCorrector (20MB HistGBT), HealingAgent (7MB HistGBT).
  - Best Val Loss: 0.0598 (Epoch 83), Avg MAPE: ~3.1% across 19 regions.
- **Chaos Engine v2:**
  - SnapshotBuilder → ScenarioEngine → CascadeEngine → RedispatchEngine → HealingAgent pipeline fully wired.
  - 7 scenario types: single_gen_failure, n_of_k_failure, renewable_collapse, demand_spike, storm_regional, compound, heatwave.
  - AI vs Rule-Based healing comparison with resilience scoring fully operational.
  - Batch simulation (up to 500 scenarios) and worst-case adversarial search endpoints live.
- **Documentation:**
  - IEEE paper draft: `FinSight_IEEE_Experimental_Paper_v24.docx`
  - Spain GNN Digital Twin TRD: `Spain_Energy_GNN_TRD.docx`
  - Mid-sem presentation: `EDI_MidSem_Presentation.pptx`

## Quick Links
- [[10_Architecture]]: System architecture, tech stack, and component interactions.
- [[20_Data_Dictionary]]: Schema for all data files.
- [[30_Decisions_Log]]: Critical decisions made during the project.

## AI Memory Anchors
- **Workspace Root:** `c:\Users\soham\Desktop\edi-1`
- **Data Target:** Spain (Regional for demand, Plant-level for generation).
- **Core Model:** Spatio-Temporal Graph Neural Network (ST-GNN) acting as a surrogate simulator.
- **Goal:** Digital Twin capable of resilience scoring via Chaos Engineering (Synthetic failures).
