# Spatio-Temporal Electric Grid Power Simulator

A Graph Attention Network (GAT)-based digital twin for Spain's electric grid. Provides 24-hour regional power demand forecasting, cascade failure simulation, and chaos engineering scenarios — with a full-stack React + FastAPI dashboard.

## Architecture

```
forecasting/       # GAT model, training, inference
digital_twin/      # Simulator, cascade, healing agent, virtual buffer
backend/           # FastAPI server (REST API + WebSocket)
frontend/          # React + Vite dashboard
scripts/           # Data pipeline, evaluation, figure generation
data/
  raw/             # Downloaded JSON from REE API
  excel/           # Regional demand & weather Excel files
  processed/       # Processed CSVs, graphs, parquet (gitignored if large)
  models/          # Trained .pth / .pkl checkpoints
paper_figures/     # Publication-quality figures and LaTeX tables
docs/              # Architecture, decisions log, experiment log
```

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Frontend dependencies
```bash
cd frontend
npm install
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Fill in your ENTSOE_API_KEY
```

### 4. Download & process data (first time only)
```bash
python download_demand.py
python download_weather.py
python download_entsoe_data.py
python download_prices.py
python download_grid_data.py
python disaggregate_demand.py
python build_graphs.py
```

### 5. Train the model (optional — pretrained checkpoints in `data/models/`)
```bash
python forecasting/train.py
python forecasting/train_holiday_focus.py
python forecasting/train_weekend_focus.py
python forecasting/train_regional.py
```

### 6. Launch the app
```bash
start_app.bat          # Windows
# OR manually:
uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
```

The dashboard will be available at `http://localhost:5173`.

## Key Features

- **24-hour regional forecasting** across 19 Spanish autonomous communities
- **GAT-based architecture** with asymmetric loss, holiday/weekend specialists, and hybrid corrector
- **Cascade failure simulation** with self-healing agent
- **Chaos engineering** — inject faults (drought, heatwave, geomagnetic storms, etc.)
- **3D grid visualiser** and Spain choropleth map
- **Virtual battery buffer economics** model

## Model Checkpoints

Pre-trained models are included in `data/models/`:
| File | Description |
|---|---|
| `best_gat_forecaster_asymmetric.pth` | Main GAT forecaster (asymmetric loss) |
| `best_gat_forecaster_holiday.pth` | Holiday-specialist model |
| `best_gat_forecaster_weekend.pth` | Weekend-specialist model |
| `best_gat_regional.pth` | Regional GNN forecaster |
| `hybrid_corrector.pkl` | Ensemble corrector |
| `healing_agent.pkl` | RL-based self-healing agent |

## Data Sources

- **Demand**: Red Eléctrica de España (REE) API
- **Generation**: ENTSO-E Transparency Platform
- **Weather**: Open-Meteo / Meteorological services
- **Grid topology**: REE network data

## License

MIT
