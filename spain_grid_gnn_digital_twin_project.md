# Spain Electricity GNN + Digital Twin + Chaos Engineering Project

## 1. Project Vision

Build a research-grade, data-calibrated **Digital Twin of the Spanish electricity system** that combines:

1. Electricity demand forecasting
2. Renewable and generation forecasting
3. Graph Neural Networks (GNNs)
4. A reduced-order representation of the Spanish power grid
5. Power-plant and transmission contingency simulation
6. Automated "Chaos Engineering" inspired scenario generation
7. Redispatch / mitigation optimization
8. Grid resilience scoring
9. Extreme-weather and compound-disaster scenarios

The project should NOT claim to be an exact real-time replica of Spain's physical grid. It is a **research-grade, reduced-order digital twin calibrated using publicly available Spanish and European electricity data**.

---

# 2. Original Idea

The original project was:

**GNN-based electricity forecasting for Spain**

The model learns temporal and spatial relationships in Spanish electricity data and forecasts:

- Electricity demand
- Total generation
- Wind generation
- Solar generation
- Hydro generation
- Other generation technologies
- Potentially regional generation/load

The main motivation was that electricity systems have strong temporal and geographical dependencies, making Graph Neural Networks a useful approach compared with purely time-series models.

Potential models:

- GCN
- GraphSAGE
- GAT
- Temporal GNN
- STGCN
- DCRNN
- GNN + LSTM/GRU
- GNN + Transformer

Baseline models should also be included:

- Persistence
- Linear Regression
- Random Forest / XGBoost
- LSTM/GRU

Evaluation metrics:

- MAE
- RMSE
- MAPE
- R²
- Forecast horizon performance
- Performance under different weather/regime conditions

---

# 3. Updated Core Idea

Extend the forecasting project into:

## "A GNN-Based Digital Twin with Chaos Engineering for Resilience Assessment of the Spanish Power Grid"

The forecasting model becomes one component of a larger digital twin.

The digital twin represents:

- Generation assets
- Generation clusters
- Demand regions
- Grid nodes
- Transmission connections
- Storage
- Weather conditions
- Historical operating states

The system can then inject synthetic failures and extreme events to answer:

> "What happens to the Spanish electricity system if a generator, transmission asset, or region is affected by a technical failure or extreme weather event?"

The simulator then estimates:

- Lost generation
- Available reserve
- Generation deficit
- Redispatch requirements
- Potential unserved energy
- System risk
- Recovery time
- Resilience score
- Recommended mitigation actions

---

# 4. Important Data Assumption

Real-time telemetry from every Spanish power plant is NOT required.

For large generation units, historical generation and unit metadata can be used to estimate expected output.

For example:

Plant A:
- Installed capacity = 1000 MW
- Historical average output = 820 MW
- Typical output range = 500–980 MW
- Historical hourly/seasonal profile available

If Plant A fails during a simulated event, the simulator can estimate the lost generation using the expected output for that time/context.

This must be described as a **modelled estimate**, not real-time telemetry.

For smaller/distributed assets where unit-level data is unavailable, use aggregated regional/technology-level generation.

---

# 5. Primary Data Sources

## REE / Red Eléctrica

Use REE as the primary source for Spanish electricity-system data.

Target data:

- Electricity demand
- Generation by technology
- Installed capacity
- Renewable generation
- Generation structure
- Regional/CCAA data where available
- Historical system states
- e·sios/API data where access is available

Official source:

https://www.ree.es/

e·sios:

https://api.esios.ree.es/

## ENTSO-E Transparency Platform

Use ENTSO-E for European electricity-system information and unit-level data.

Target datasets:

- Actual Generation per Generation Unit
- Production / Generation Unit metadata
- Installed capacity
- Generation unit technology
- Location / bidding zone
- Outage / availability information where available
- Cross-border flows
- Transmission-related information where available

Official source:

https://transparency.entsoe.eu/

Important limitation:

Unit-level data is strongest for large generation units (not every small distributed generator). Do not claim complete plant-level telemetry for every asset in Spain.

## Weather Data

Potential sources:

- AEMET
- ERA5 / Copernicus
- Open-Meteo for prototyping
- Other reputable historical weather datasets

Variables:

- Temperature
- Wind speed
- Wind direction
- Precipitation
- Solar irradiance
- Extreme weather indicators

Weather is used both for forecasting and for scenario generation.

---

# 6. Digital Twin Representation

Represent the electricity system as a graph.

## Nodes

Possible node types:

- Large power plants
- Generation clusters
- Grid/substation nodes
- Regional demand nodes
- Storage facilities
- Interconnection nodes

Each node can contain:

- Latitude/longitude or region
- Technology
- Installed capacity
- Historical average generation
- Current/estimated generation
- Availability
- Forecast generation
- Demand
- Weather features
- Node type

## Edges

Possible edges:

- Transmission connection
- Electrical connectivity
- Geographic relationship
- Power-flow relationship
- Regional connectivity

Edge features:

- Approximate capacity
- Voltage level where available
- Historical flow
- Distance
- Connection type
- Status

Where detailed transmission data is unavailable, use a carefully documented reduced-order topology.

---

# 7. GNN Component

The GNN should model relationships between different parts of the electricity system.

Input:

- Historical generation
- Demand
- Weather
- Capacity
- Time features
- Node features
- Graph connectivity

Output:

- Future demand
- Future generation
- Regional states
- Expected generation at individual/cluster nodes
- Potential system stress indicators

Potential architecture:

Historical data
    ↓
Feature engineering
    ↓
Graph construction
    ↓
Temporal encoder
    ↓
GNN
    ↓
Forecast
    ↓
Digital twin state

Start simple with:

**GNN + LSTM/GRU**

Then experiment with:

- GAT
- GraphSAGE
- STGCN
- Temporal GNN
- GNN + Transformer

Do not overcomplicate the first implementation.

---

# 8. Failure / Contingency Simulator

The digital twin should support controlled failure injection.

Example:

Plant A:
- Capacity = 1000 MW
- Estimated generation = 820 MW

Failure:

Plant A = OFF

Then:

Lost generation ≈ 820 MW

The simulator evaluates the remaining system.

---

# 9. Scenario Types

## Scenario 1 — Technical Power Plant Failure

Example:

- Generator suddenly unavailable
- Duration = 2–24 hours
- Severity = 100% outage

Process:

Plant fails
→ generation loss
→ calculate reserve
→ identify available generation
→ redispatch
→ calculate remaining deficit
→ calculate resilience

---

## Scenario 2 — Flood

Example:

Severe rainfall/flooding affects a region.

Possible impacts:

- Hydro plant shutdown
- Thermal plant safety shutdown
- Substation outage
- Transmission line outage
- Regional demand disruption

Chain:

Weather event
→ geographical impact
→ affected assets
→ asset outage
→ generation/transmission loss
→ redispatch
→ resilience score

---

## Scenario 3 — Hurricane / Extreme Wind

Spain-specific implementation should focus on realistic extreme-wind/storm events rather than assuming every hurricane scenario is common.

Possible effects:

- Wind farm curtailment
- Transmission line outage
- Substation outage
- Temporary generation reduction

Chain:

Extreme wind
→ asset vulnerability
→ generation/transmission reduction
→ network stress
→ redispatch

---

## Scenario 4 — Wildfire

Possible effects:

- Transmission corridor unavailable
- Substation affected
- Regional generation disconnected

Chain:

Wildfire
→ transmission asset outage
→ power-flow rerouting
→ congestion
→ potential overload
→ mitigation

---

## Scenario 5 — Extreme Heatwave

Model multiple effects:

- Demand increases
- Thermal generation availability may decrease
- Transmission capacity may be affected
- Hydro conditions can change
- Solar generation may change depending on assumptions

Example:

Demand +15%
Thermal availability -10%
Solar +/− modelled change
Wind based on weather
→ calculate system stress

---

## Scenario 6 — Renewable Generation Drop

Examples:

- Low wind
- Cloud cover
- Regional solar reduction
- Combined low-wind + high-demand event

---

## Scenario 7 — Transmission Failure

A transmission connection is removed from the graph.

Then calculate:

- New connectivity
- Congestion
- Available alternative paths
- Potential generation that can no longer reach demand

This is more difficult and should be implemented after the generation-failure simulator works.

---

# 10. Compound Events

A major novelty opportunity is simulation of multiple simultaneous events.

Examples:

### Compound A

Plant failure
+
high demand

### Compound B

Flood
+
hydro plant outage
+
transmission outage

### Compound C

Heatwave
+
low wind
+
large generator outage

### Compound D

Extreme weather
+
transmission failure
+
multiple generation outages

The system should determine whether the grid remains balanced.

---

# 11. Chaos Engineering-Inspired Engine

Inspired by the philosophy of Netflix Chaos Engineering / Chaos Monkey.

Do NOT claim this is literally Netflix Chaos Monkey applied to the grid.

Call it something like:

**Grid Chaos Engine (GCE)**

Purpose:

Automatically generate realistic disruptions and test system resilience.

The engine should be constrained by physical and statistical assumptions.

---

# 12. Chaos Engine Modes

## Mode A — Random Chaos

Randomly generate scenarios.

Example:

- Random plant
- Random outage duration
- Random severity
- Random weather event
- Random transmission failure

Run thousands of simulations.

Goal:

Build a large synthetic contingency dataset.

---

## Mode B — Targeted Chaos

Identify important/critical assets and test them.

Example:

Find asset with high:

- Generation capacity
- Network centrality
- Limited replacement capacity
- Regional importance

Then simulate its failure.

Goal:

Identify critical infrastructure.

---

## Mode C — Adaptive Chaos

The engine learns which scenarios produce high stress.

Loop:

Generate scenario
→ simulate
→ calculate resilience
→ generate harder/related scenario
→ simulate
→ repeat

Goal:

Search for the most damaging realistic scenarios.

Potential optimization:

Find:

S* = argmin R(S)

where:

S = disruption scenario

R(S) = resilience score

Subject to:

S being physically/statistically plausible.

---

# 13. Chaos Levels

## Level 1 — Minor

- One small asset
- Partial generation reduction

## Level 2 — Moderate

- One large generator outage
- Or one transmission outage

## Level 3 — Severe

- Large generator + weather event
- Multiple assets

## Level 4 — Extreme

- Regional extreme weather
- Multiple generation/transmission failures

## Level 5 — Maximum Realistic Stress

- High demand
- Low renewable generation
- Major generator outage
- Transmission constraint
- Weather-related asset failures

The goal is NOT to manufacture impossible disaster scenarios.

---

# 14. Redispatch / Mitigation Engine

After a failure, determine how remaining assets can compensate.

Inputs:

- Current generation
- Available capacity
- Ramp/dispatch assumptions
- Location
- Connectivity
- Storage
- Imports/exports
- Demand

Example:

Plant A fails:

Lost:
820 MW

Potential response:

Hydro +300 MW
Gas +350 MW
Battery +100 MW
Imports +70 MW

Total:
820 MW

System balanced.

The engine should calculate this rather than hard-code which plant replaces another.

Possible optimization methods:

- Linear Programming
- Mixed Integer Linear Programming
- Network optimization
- PyPSA
- pandapower
- scipy.optimize

Start with a simplified optimization model.

---

# 15. Resilience Metrics

Do not only report "success/failure."

Calculate:

- Generation deficit
- Reserve margin
- Unserved energy
- Percentage of demand served
- Renewable share
- Number of affected assets
- Recovery time
- Redispatch magnitude
- Storage usage
- Import requirement
- Transmission congestion where modelled
- Resilience score

Possible resilience score:

0–20   LOW RISK
20–40  MODERATE
40–60  HIGH
60–80  VERY HIGH
80–100 CRITICAL

The exact scoring formula must be justified experimentally rather than arbitrarily presented as an official grid standard.

---

# 16. Example Simulation

Initial state:

Demand = 31,420 MW
Generation = 33,180 MW
Reserve = 1,760 MW

Event:

Flood in Andalusia

Affected assets:

- Plant A
- Substation B
- Transmission Line C

Estimated generation loss:

1,240 MW

Initial post-failure:

Generation = 31,940 MW

Reserve becomes much smaller.

Redispatch:

Hydro +400 MW
Gas +500 MW
Storage +150 MW
Imports +100 MW

Total compensation:

1,150 MW

Remaining deficit:

90 MW

System result:

- Demand not fully served
- Risk = HIGH
- Required mitigation = additional import / load management / alternative dispatch

The exact numbers above are illustrative only. Real values must come from the dataset/model.

---

# 17. Training Data Strategy

Real catastrophic failures are rare.

Therefore use:

## Real data

For calibration and validation:

- Historical demand
- Generation
- Weather
- Plant metadata
- Availability/outage information
- Flows where available

## Synthetic data

For resilience training:

- Generator failures
- Transmission failures
- Weather disruptions
- Renewable drops
- Demand spikes
- Compound failures

This produces a synthetic contingency dataset.

Example:

10,000–100,000+ simulations

Each simulation becomes:

Input:
- Grid state
- Weather
- Demand
- Generation
- Failure scenario

Output:
- New grid state
- Deficit
- Redispatch
- Risk
- Resilience

---

# 18. Potential ML Task

The GNN can eventually learn:

Given:

Grid state + disruption

Predict:

- Post-contingency generation
- Regional stress
- Deficit
- Resilience
- Potentially recommended response

This can make the GNN a fast surrogate for expensive simulation.

Architecture:

Physics/simulation engine
        ↓
Generate many scenarios
        ↓
Synthetic training dataset
        ↓
Train GNN surrogate
        ↓
New unseen scenario
        ↓
GNN rapidly predicts response

This is potentially one of the strongest technical directions.

---

# 19. Recommended Software Stack

Python:

- pandas
- numpy
- scipy
- scikit-learn
- PyTorch
- PyTorch Geometric
- XGBoost
- matplotlib
- geopandas
- networkx

Power-system simulation:

- PyPSA
- pandapower

Data:

- REE API / downloadable datasets
- ENTSO-E Transparency Platform
- AEMET / ERA5 / Copernicus

Visualization:

- Plotly
- Dash or Streamlit
- GeoPandas / Folium
- NetworkX

Database:

- PostgreSQL/PostGIS if needed
- SQLite for early prototype

---

# 20. Development Roadmap

## Phase 1 — Data

Collect:

- Spain demand
- Generation by technology
- Installed capacity
- Weather
- Large generation unit metadata
- Unit generation where available

Build clean datasets.

---

## Phase 2 — Forecasting

Build:

- Baseline models
- LSTM/GRU
- GNN
- GNN + temporal model

Compare performance.

---

## Phase 3 — Digital Twin

Create:

- Generation nodes
- Demand nodes
- Regional nodes
- Grid edges
- Asset metadata

Create a simplified Spain graph.

---

## Phase 4 — Basic Failure Simulator

Implement:

- Plant OFF
- Generation loss
- Reserve calculation
- Simple redispatch

Do NOT start with complex weather events.

---

## Phase 5 — Optimization

Add:

- Available capacity
- Redispatch optimization
- Storage
- Imports/exports
- Constraints

---

## Phase 6 — Weather Scenarios

Add:

- Flood
- Extreme wind
- Heatwave
- Wildfire
- Low wind
- Solar reduction

Map weather conditions to asset impacts using documented assumptions.

---

## Phase 7 — Chaos Engine

Add:

- Random scenarios
- Targeted scenarios
- Compound scenarios
- Scenario severity
- Duration
- Adaptive stress testing

---

## Phase 8 — GNN Surrogate

Train a GNN on simulation-generated contingency scenarios.

Compare:

Full simulator vs GNN prediction.

Measure:

- MAE
- RMSE
- Classification accuracy
- Risk prediction accuracy
- Inference speed

---

# 21. Research Experiments

Experiment 1:
Forecasting performance.

Experiment 2:
Normal grid-state prediction.

Experiment 3:
Single generator failure.

Experiment 4:
Transmission failure.

Experiment 5:
Weather-induced outage.

Experiment 6:
Compound failure.

Experiment 7:
Random Chaos Engine.

Experiment 8:
Targeted Chaos Engine.

Experiment 9:
Adaptive Chaos Engine.

Experiment 10:
GNN surrogate vs physics/optimization simulator.

Experiment 11:
Unseen failure scenarios.

Experiment 12:
Robustness under different seasons and demand conditions.

---

# 22. Strong Research Questions

### RQ1

Can a graph-based model improve electricity forecasting by incorporating spatial/network relationships?

### RQ2

Can a reduced-order digital twin reproduce important system-level behavior of the Spanish electricity system?

### RQ3

How resilient is the Spanish electricity system under synthetic generation, transmission, and weather contingencies?

### RQ4

Can automated chaos-engineering-inspired scenario generation identify critical assets and high-risk compound events?

### RQ5

Can a GNN learn to approximate post-contingency grid behavior significantly faster than repeated full simulations?

RQ5 is especially interesting if the implementation is strong.

---

# 23. What We Can Claim

We CAN claim:

- Data-calibrated digital twin
- Reduced-order Spanish electricity system model
- Historical/modelled plant output estimates
- Synthetic contingency generation
- Weather-induced scenario simulation
- Automated resilience testing
- GNN-based forecasting
- GNN-based contingency prediction if validated
- Optimization-based redispatch
- Identification of critical scenarios/assets

---

# 24. What We Must NOT Claim

Do NOT claim:

- Exact real-time replica of Spain
- Complete telemetry of every power plant
- Exact internal power-plant operating state
- Operator-grade grid control
- Guaranteed real-world stability
- Exact prediction of unexpected physical failures
- Exact AC power-flow behavior unless actually implemented and validated
- Official REE/ENTSO-E operational recommendations

Always distinguish:

REAL DATA
vs
MODELLED ESTIMATE
vs
SYNTHETIC SCENARIO
vs
SIMULATED RESULT

---

# 25. Final Proposed Architecture

                    SPAIN ELECTRICITY DATA
                             |
            +----------------+----------------+
            |                |                |
           REE            ENTSO-E          WEATHER
            |                |                |
            +----------------+----------------+
                             |
                      DATA PROCESSING
                             |
                             v
                    FEATURE ENGINEERING
                             |
                             v
                      SPAIN GRID GRAPH
                             |
               +-------------+-------------+
               |                           |
               v                           v
        FORECASTING GNN              DIGITAL TWIN
               |                           |
               |                           v
               |                   SCENARIO ENGINE
               |                           |
               |                  +--------+--------+
               |                  |                 |
               |                  v                 v
               |             Single Failure     Weather Event
               |                  |                 |
               |                  +--------+--------+
               |                           |
               |                           v
               |                    CHAOS ENGINE
               |                           |
               |                 +---------+---------+
               |                 |                   |
               |                 v                   v
               |             Random Chaos       Adaptive Chaos
               |                 |                   |
               |                 +---------+---------+
               |                           |
               |                           v
               |                    GRID SIMULATOR
               |                           |
               |                           v
               |                   REDISPATCH OPT
               |                           |
               +---------------------------+
                           |
                           v
                    RESILIENCE METRICS
                           |
                           v
                    DIGITAL TWIN UI
                           |
                           v
                 Research Experiments


# 26. Final Project Concept

The project is no longer simply:

"Forecast Spanish electricity using a GNN."

It is:

**A GNN-Based Digital Twin with Chaos Engineering for Forecasting, Contingency Simulation, and Resilience Assessment of the Spanish Power Grid.**

Core pipeline:

Real Spanish data
→ Forecasting
→ Graph representation
→ Digital twin
→ Failure injection
→ Chaos scenario generation
→ Redispatch optimization
→ Resilience evaluation
→ GNN surrogate
→ Fast prediction of unseen contingencies

The central idea is to use real historical Spanish electricity data to construct a realistic reduced-order digital twin, then use controlled synthetic disruptions to explore scenarios that are difficult or rare to observe in historical data.

The project should prioritize scientific validity over complexity. A smaller, well-validated digital twin with clearly documented assumptions is better than an enormous but unsupported simulation.
