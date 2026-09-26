---
tags: [data, schema]
---
# Data Dictionary

This document enforces the data structures we use, preventing the AI from hallucinating column names or shapes.

## 1. Processed Weather Data
**Location:** `data/processed/weather/*.csv` (Also duplicated in `data/excel/weather/`)
**Format:** 1 CSV per region.
**Rows:** 96,432 (2015-01-01 to 2025-12-31, hourly)
**Columns:**
- `region`: String (e.g. 'Andalucia')
- `datetime`: ISO-8601 string (e.g. '2015-01-01T00:00')
- `temperature_2m`: Float (°C)
- `relative_humidity_2m`: Float (%)
- `wind_speed_10m`: Float (km/h)
- `shortwave_radiation`: Float (W/m²)

## 2. Spatial Graphs
**Location:** `data/processed/graphs/`
- `node_ids.csv`: Maps integer index `[0-18]` to Region string and lat/lon.
- `A_geo.npy`: 19x19 float matrix. Geographic adjacency via Haversine distance and Gaussian kernel (sigma=200km).
- `A_elec.npy`: 19x19 float matrix. Electrical adjacency (binary). 1 = connected by land border or HVDC cable.

## 3. Processed Demand Data & API
**Location:** `data/processed/demand/*.csv`
*(Currently fully populated. The scripts fetch 11 years (2015-2026) of hourly regional data using the e-sios personal token, bypassing the public API limits).*

## 4. Power Plant & Generation Data
**Location:** `data/processed/generation/`
*(Fully populated by ENTSO-E API).*
- `installed_capacity_ES_*.csv`: The MW capacity of different technologies in Spain.
- `aggregated_generation_ES_*.csv`: Hourly MW output by generation type (Wind, Solar, Gas, etc.).
- `unit_generation_ES_*.csv`: Plant-level telemetry. The core input for the Chaos Engine to simulate specific generator failures.

## 5. Machine Learning Features (16 Channels)
The final `dataset.py` constructs a 16-channel array per region for the GNN:
- **Indices 0-4:** Demand, Temp, Humidity, Wind, Solar.
- **Indices 5-6:** Pop Density, Industry Index.
- **Index 7:** Wholesale Price (ENTSO-E).
- **Indices 8-13:** Cyclic Time Encodings (Sin/Cos).
- **Index 14:** `is_holiday` (Binary).
- **Index 15:** `is_weekend` (Binary).
