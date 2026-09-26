---
tags: [decisions, log, architecture]
---
# Decisions Log

If something looks confusing, check here first. This logs *why* we did things a certain way.

## D1: Demand Data API Restrictions
**Date:** 2026-08-24
**Decision:** We are using an `e-sios` personal token instead of the public `apidatos.ree.es` API.
**Reason:** The public REData API endpoint `demanda/evolucion` completely blocks hourly resolution queries when `geo_limit=ccaa` (it returns `400 Bad Request`). We tried chunking by month and by day, but the server fundamentally rejects hourly regional data on the public tier. We must wait for the e-sios token.

## D2: Weather Variables
**Date:** 2026-08-24
**Decision:** We are using `shortwave_radiation` instead of `surface_solar_radiation_down`.
**Reason:** Open-Meteo's historical archive API rejected the latter string. `shortwave_radiation` is the correct parameter for solar irradiance in W/m².

## D3: Electrical Graph ($A_{elec}$)
**Date:** 2026-08-24
**Decision:** Used shared land borders as a proxy for the electrical adjacency matrix.
**Reason:** We did not have a digitized REE high-voltage transmission map. HV lines almost entirely follow land borders. We explicitly added the HVDC submarine link (the Rómulo project) connecting Valencia to the Balearic Islands to prevent the islands from being totally isolated in the GNN. The Canary Islands, Ceuta, and Melilla remain isolated as they have no direct AC ties to the mainland graph.

## D4: E-sios API Rate Limiting & Responsibility
**Date:** 2026-08-24
**Decision:** All download scripts must be perfectly idempotent and heavily cached. They must NEVER redownload data that already exists locally.
**Reason:** Per the e-sios terms of service for personal tokens: "uso responsable de la API: no realice peticiones masivas, redundantes o innecesarias" (responsible API use: do not make massive, redundant, or unnecessary requests, such as previously downloaded information). Any scripts we write to fetch REE data will strictly check the local `data/` directory first before making an HTTP request.

## D5: Frontend Supply Radius Calculation
**Date:** 2026-08-24
**Decision:** We calculate the supply radius dynamically on the frontend based on the maximum distance to supplied regions, multiplied by 1.6.
**Reason:** It avoids needing to pre-calculate and transmit heavy geographic bounds from the backend on every chaos simulation. The 1.6 multiplier ensures the 3D radius circles overlap realistically and fully cover the region's borders, avoiding visual gaps.

## D6: 3D Plant Visualization
**Date:** 2026-08-24
**Decision:** Power plants are represented as revolving upside-down 3D cones using DeckGL's `SimpleMeshLayer` and a custom-built cone mesh.
**Reason:** DeckGL's native `ColumnLayer` only supports cylinders, which felt too basic for the "sci-fi tracking" aesthetic required for the digital twin. The custom mesh provides a distinct, premium visual anchor for nodes.

## D7: 16-Channel Architecture & Weekend Specialist
**Date:** 2026-09-04
**Decision:** Upgraded the ST-GNN architecture from 15 to 16 input channels by introducing an explicit `is_weekend` binary feature, and introduced a 3rd "Weekend" specific GAT model.
**Reason:** The model was heavily over-predicting demand on weekends. Even with time-encodings, the GNN struggled to drop demand steeply enough. The `is_weekend` flag acts as a hard indicator, and the dedicated weekend model combined with a relaxed HybridCorrector clamp (`max_delta=0.60` on weekends) perfectly tracks weekend load collapses.

## D8: Backend Dataset Unpacking Fix
**Date:** 2026-09-05
**Decision:** Fixed tuple unpacking across `backend/main.py`, `infer_2026.py`, and `test_2026_scenarios.py` to `x, y_true, y_holiday, y_weekend = dataset[idx]`.
**Reason:** The frontend was crashing and returning `0` values because the 16-channel upgrade changed the dataset output from a 3-tuple to a 4-tuple. The backend failed to unpack it silently causing the `/api/forecast` inference pipeline to abort. Properly capturing `y_weekend` resolves the zero-value graphs in the UI.
