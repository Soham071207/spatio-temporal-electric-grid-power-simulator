import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import unicodedata
import holidays

# Approximate Data for Spain's Autonomous Communities
# Density in pop/km2
POPULATION_DENSITY = {
    'Andalucía': 97, 'Aragón': 28, 'Cantabria': 109, 'Castilla-La Mancha': 26,
    'Castilla y León': 25, 'Cataluña': 240, 'País Vasco': 300, 'Principado de Asturias': 95,
    'Ceuta': 4200, 'Melilla': 6200, 'Comunidad de Madrid': 830, 'Comunidad Foral de Navarra': 63,
    'Comunidad Valenciana': 215, 'Extremadura': 25, 'Galicia': 91, 'Islas Baleares': 235,
    'Canarias': 300, 'La Rioja': 62, 'Región de Murcia': 133
}

# Industry Intensity Score (0 to 1) indicating manufacturing/heavy industry share
INDUSTRY_INDEX = {
    'Andalucía': 0.11, 'Aragón': 0.22, 'Cantabria': 0.18, 'Castilla-La Mancha': 0.16,
    'Castilla y León': 0.19, 'Cataluña': 0.20, 'País Vasco': 0.24, 'Principado de Asturias': 0.19,
    'Ceuta': 0.02, 'Melilla': 0.02, 'Comunidad de Madrid': 0.10, 'Comunidad Foral de Navarra': 0.28,
    'Comunidad Valenciana': 0.15, 'Extremadura': 0.12, 'Galicia': 0.17, 'Islas Baleares': 0.05,
    'Canarias': 0.06, 'La Rioja': 0.25, 'Región de Murcia': 0.15
}

def normalize_name(name):
    """Normalize node names to match weather file names."""
    # Strip accents
    n = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Replace spaces with underscores
    n = n.replace(' ', '_')
    return n

class SpainElectricityDataset(Dataset):
    def __init__(self, data_dir="data", seq_length=24, pred_horizon=24, split="train", year_start=2020, year_end=2026):
        """
        Args:
            data_dir (str): Root data directory
            seq_length (int): Number of historical hours to use as input
            pred_horizon (int): Number of future hours to predict
            split (str): 'train', 'val', or 'test'
            year_start (int): Start year for dataset
            year_end (int): End year for dataset
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.split = split
        self.year_start = year_start
        self.year_end = year_end
        
        # Region → holidays library subdivision code for per-region holiday detection
        self.REGION_SUBDIV = {
            'Andalucía': 'AN', 'Aragón': 'AR', 'Cantabria': 'CB',
            'Castilla-La Mancha': 'CM', 'Castilla y León': 'CL',
            'Cataluña': 'CT', 'País Vasco': 'PV', 'Principado de Asturias': 'AS',
            'Ceuta': 'CE', 'Melilla': 'ML', 'Comunidad de Madrid': 'MD',
            'Comunidad Foral de Navarra': 'NC', 'Comunidad Valenciana': 'VC',
            'Extremadura': 'EX', 'Galicia': 'GA', 'Islas Baleares': 'IB',
            'Canarias': 'CN', 'La Rioja': 'RI', 'Región de Murcia': 'MC'
        }
        
        # Load adjacency matrix
        A = np.load(os.path.join(data_dir, "processed", "graphs", "A_elec.npy"))
        
        # Convert adjacency matrix to edge_index for PyTorch Geometric
        src, dst = np.nonzero(A)
        self.edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
        self.edge_weight = torch.tensor(A[src, dst], dtype=torch.float)
        self.num_nodes = A.shape[0]
        
        # Load node list
        nodes_df = pd.read_csv(os.path.join(data_dir, "processed", "graphs", "node_ids.csv"))
        self.node_names = nodes_df['region'].tolist()
        
        # Load and align temporal data (Demand + Weather + Time)
        full_data, full_time_idx = self._load_data()
        
        T_full, N, F = full_data.shape
        gap = self.seq_length  # Prevent data leakage between splits
        train_end = int(T_full * 0.8)
        val_start = train_end + gap
        val_end = int(T_full * 0.9)
        test_start = val_end + gap
        
        # ===== SPLIT-AWARE SCALING =====
        # Layout: [0]=demand, [1-4]=weather, [5]=pop_density_norm, [6]=industry_norm,
        #         [7]=wholesale_price, [8-13]=cyclical encodings, [14]=is_holiday
        #
        # IMPORTANT: pop_density (5) and industry_index (6) are STATIC per node —
        # they are the same value at every timestep for a given node. Applying a
        # per-node temporal z-score (NodeAwareScaler) to a constant time-series
        # always produces mean==constant and std==0 → zero after scaling. To prevent
        # this, we pre-normalize them across nodes ONCE (before the temporal scaler)
        # and exclude them from the temporal scaler block.
        #
        # The temporal scaler (NodeAwareScaler) only handles dynamically-varying
        # features: demand (0), weather (1-4), and wholesale_price (7).  Because
        # these are at non-contiguous indices we collect them into a 6-column view,
        # fit/transform, then write back — keeping the overall (T,N,15) layout.
        #
        # Only feature 0 (demand) ever needs inverse_transform for MW recovery.
        # num_scaled_features is set to 6 so that existing inv_transform helpers
        # (which build a (24, num_scaled_features) dummy and populate column 0) work
        # correctly: inverse_transform uses node-aware mean/std derived from the
        # 6-column training slice, and column 0 is still demand.
        self.num_scaled_features = 7  # demand(0), weather(1-5), price(6) in the scaler's internal 7-col space (_dyn_cols=[0,1,2,3,4,5,8])

        # --- Pre-normalize static features across nodes ---
        # pop_density: spans ~4 to ~6200 → log-transform then z-score across N
        pop_vals = full_data[0, :, 5]  # shape (N,) — same at every timestep
        log_pop = np.log1p(pop_vals)
        pop_mean, pop_std = log_pop.mean(), log_pop.std()
        if pop_std == 0:
            pop_std = 1.0
        full_data[:, :, 5] = (log_pop - pop_mean) / pop_std  # broadcast over T

        # industry_index: already 0–1 → z-score across nodes
        ind_vals = full_data[0, :, 6]  # shape (N,)
        ind_mean, ind_std = ind_vals.mean(), ind_vals.std()
        if ind_std == 0:
            ind_std = 1.0
        full_data[:, :, 6] = (ind_vals - ind_mean) / ind_std  # broadcast over T

        class NodeAwareScaler:
            def __init__(self):
                self.mean = None
                self.std = None

            def fit(self, data):
                # data is (T, N, 6): demand + weather + price
                self.mean = np.nanmean(data, axis=0)  # (N, 6)
                self.std = np.nanstd(data, axis=0)    # (N, 6)
                self.std[self.std == 0] = 1.0

            def transform(self, data):
                return (data - self.mean) / self.std

            def inverse_transform(self, data, node_idx=None):
                if node_idx is not None:
                    return data * self.std[node_idx] + self.mean[node_idx]
                return data * self.std + self.mean

        self.scaler = NodeAwareScaler()
        # Build the 6-column training view: demand(0), weather(1-4), price(7)
        _dyn_cols = [0, 1, 2, 3, 4, 5, 8]
        train_scalable = full_data[:train_end, :, :][:, :, _dyn_cols]
        self.scaler.fit(train_scalable)
        
        # Apply split logic
        if self.split == 'train':
            self.time_idx = full_time_idx[:train_end]
            self.data_matrix = full_data[:train_end]
        elif self.split == 'val':
            self.time_idx = full_time_idx[val_start:val_end]
            self.data_matrix = full_data[val_start:val_end]
        else:
            self.time_idx = full_time_idx[test_start:]
            self.data_matrix = full_data[test_start:]
            
        T_split = self.data_matrix.shape[0]

        # Scale the 6 dynamic features (demand, weather, price) in-place.
        # Pop density (5) and industry index (6) were pre-normalized above and
        # are passed through unchanged. Cyclical/holiday (8-14) stay unscaled.
        _dyn_cols = [0, 1, 2, 3, 4, 5, 8]
        dyn_scaled = self.scaler.transform(self.data_matrix[:, :, _dyn_cols])

        self.scaled_data = self.data_matrix.copy().astype(np.float32)
        for out_col, in_col in enumerate(_dyn_cols):
            self.scaled_data[:, :, in_col] = dyn_scaled[:, :, out_col]

        self.tensor_data = torch.tensor(self.scaled_data, dtype=torch.float32)
        
    def _load_data(self):
        """
        Loads demand, weather, and time data, aligns them, and returns a 3D numpy array
        of shape (Time, Nodes, Features)
        Features: [demand_mwh, temp, humidity, wind, solar, sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_holiday]
        """
        # 1. Load regional demand
        demand_df = pd.read_csv(os.path.join(self.data_dir, "processed", "grid_telemetry", "demand_regional_disaggregated.csv"))
        demand_df['datetime'] = pd.to_datetime(demand_df['datetime'])
        
        # Filter years
        mask = (demand_df['datetime'].dt.year >= self.year_start) & (demand_df['datetime'].dt.year <= self.year_end)
        demand_df = demand_df[mask]
        
        # Fallback if no data in range
        if len(demand_df) == 0:
            demand_df = pd.read_csv(os.path.join(self.data_dir, "processed", "grid_telemetry", "demand_regional_disaggregated.csv"))
            demand_df['datetime'] = pd.to_datetime(demand_df['datetime'])
            
        # 2. Load regional generation (created via pre-processing script)
        gen_df = pd.read_csv(os.path.join(self.data_dir, "processed", "grid_telemetry", "generation_regional_aggregated.csv"))
        gen_df['datetime'] = pd.to_datetime(gen_df['datetime'], format='mixed')
        
        # Filter years
        mask = (gen_df['datetime'].dt.year >= self.year_start) & (gen_df['datetime'].dt.year <= self.year_end)
        gen_df = gen_df[mask]
        
        # Fallback if no data in range
        if len(gen_df) == 0:
            gen_df = pd.read_csv(os.path.join(self.data_dir, "processed", "grid_telemetry", "generation_regional_aggregated.csv"))
            gen_df['datetime'] = pd.to_datetime(gen_df['datetime'], format='mixed')
            
        # Ensure we have a uniform time index (hourly)
        time_idx = pd.date_range(start=demand_df['datetime'].min(), end=demand_df['datetime'].max(), freq='h')
        
        demand_pivot = demand_df.pivot(index='datetime', columns='region', values='demand_mwh').fillna(0)
        demand_pivot = demand_pivot.reindex(time_idx, method='ffill').fillna(0)
        
        # Reindex generation to match time_idx
        gen_df.set_index('datetime', inplace=True)
        gen_pivot = gen_df.reindex(time_idx, method='ffill').fillna(0)
        
        T = len(demand_pivot)
        N = self.num_nodes
        # Feature list: [demand_mwh, excess_mw, temp, humidity, wind, solar, pop_density, industry, price, sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month, is_holiday, is_weekend]
        F = 17 
        
        data_matrix = np.zeros((T, N, F))
        
        # Load wholesale prices
        prices_path = os.path.join(self.data_dir, "processed", "prices", "day_ahead_prices.csv")
        if os.path.exists(prices_path):
            prices_df = pd.read_csv(prices_path)
            prices_df['datetime'] = pd.to_datetime(prices_df['datetime'])
            prices_df.set_index('datetime', inplace=True)
            # Reindex to match the demand time index, fill missing with mean
            prices_df = prices_df.reindex(time_idx, method='ffill')
            if prices_df['price_eur_mwh'].isna().any():
                prices_df['price_eur_mwh'] = prices_df['price_eur_mwh'].fillna(prices_df['price_eur_mwh'].mean())
            prices_array = prices_df['price_eur_mwh'].values
        else:
            prices_array = np.zeros(T)
            
        # Time encodings
        hour = time_idx.hour.values
        dow = time_idx.dayofweek.values
        month = time_idx.month.values
        
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        sin_dow = np.sin(2 * np.pi * dow / 7.0)
        cos_dow = np.cos(2 * np.pi * dow / 7.0)
        sin_month = np.sin(2 * np.pi * (month - 1) / 12.0)
        cos_month = np.cos(2 * np.pi * (month - 1) / 12.0)
        
        # Holidays — per-region detection using autonomous community subdivision codes
        es_holidays_national = holidays.Spain(years=range(self.year_start, self.year_end + 1))
        # Build per-region holiday calendars
        regional_holiday_cals = {}
        for demand_col_name, subdiv_code in self.REGION_SUBDIV.items():
            try:
                regional_holiday_cals[demand_col_name] = holidays.Spain(
                    subdiv=subdiv_code, years=range(self.year_start, self.year_end + 1)
                )
            except Exception:
                regional_holiday_cals[demand_col_name] = es_holidays_national
        
        # National holiday array (used as fallback)
        is_holiday_national = np.array([
            1.0 if dt.date() in es_holidays_national else 0.0 
            for dt in time_idx
        ])
        
        weather_dir = os.path.join(self.data_dir, "processed", "weather")
        
        # Explicit mapping from node_ids.csv names to demand dataframe columns
        NODE_TO_DEMAND_COL = {
            'Andalucía': 'Andalucía',
            'Aragón': 'Aragón',
            'Cantabria': 'Cantabria',
            'Castilla la Mancha': 'Castilla-La Mancha',
            'Castilla y León': 'Castilla y León',
            'Cataluña': 'Cataluña',
            'País Vasco': 'País Vasco',
            'Principado de Asturias': 'Principado de Asturias',
            'Comunidad de Ceuta': 'Ceuta',
            'Comunidad de Melilla': 'Melilla',
            'Comunidad de Madrid': 'Comunidad de Madrid',
            'Comunidad de Navarra': 'Comunidad Foral de Navarra',
            'Comunidad Valenciana': 'Comunidad Valenciana',
            'Extremadura': 'Extremadura',
            'Galicia': 'Galicia',
            'Islas Baleares': 'Islas Baleares',
            'Islas Canarias': 'Canarias',
            'La Rioja': 'La Rioja',
            'Región de Murcia': 'Región de Murcia'
        }

        for i, node_name in enumerate(self.node_names):
            # Demand
            demand_col = NODE_TO_DEMAND_COL.get(node_name)
            # Find closest match if exact match fails due to encoding
            if demand_col not in demand_pivot.columns:
                for col in demand_pivot.columns:
                    if normalize_name(demand_col) == normalize_name(col):
                        demand_col = col
                        break

            if demand_col in demand_pivot.columns:
                data_matrix[:, i, 0] = demand_pivot[demand_col].values
                
            # Excess MW
            if demand_col in gen_pivot.columns:
                gen_arr = gen_pivot[demand_col].values
                demand_arr = demand_pivot[demand_col].values if demand_col in demand_pivot.columns else np.zeros_like(gen_arr)
                # Excess = max(0, Gen - Demand)
                data_matrix[:, i, 1] = np.maximum(0, gen_arr - demand_arr)
                
            # Weather
            norm_name = normalize_name(node_name)
            weather_path = os.path.join(weather_dir, f"{norm_name}.csv")
            
            if os.path.exists(weather_path):
                w_df = pd.read_csv(weather_path)
                w_df['datetime'] = pd.to_datetime(w_df['datetime'])
                w_df.set_index('datetime', inplace=True)
                
                # Reindex to match demand timeframe
                w_df = w_df.reindex(time_idx, method='ffill').fillna(0)
                
                data_matrix[:, i, 2] = w_df['temperature_2m'].values
                data_matrix[:, i, 3] = w_df['relative_humidity_2m'].values
                data_matrix[:, i, 4] = w_df['wind_speed_10m'].values
                data_matrix[:, i, 5] = w_df['shortwave_radiation'].values
            
            # Static Features
            data_matrix[:, i, 6] = POPULATION_DENSITY.get(demand_col, 50)
            data_matrix[:, i, 7] = INDUSTRY_INDEX.get(demand_col, 0.1)
            
            # Wholesale Price
            data_matrix[:, i, 8] = prices_array
            
            # Time encodings (same for all nodes at given t)
            data_matrix[:, i, 9] = sin_hour
            data_matrix[:, i, 10] = cos_hour
            data_matrix[:, i, 11] = sin_dow
            data_matrix[:, i, 12] = cos_dow
            data_matrix[:, i, 13] = sin_month
            data_matrix[:, i, 14] = cos_month
            
            # Holidays & Weekends
            regional_cal = regional_holiday_cals.get(demand_col)
            if regional_cal is not None:
                is_holiday = np.array([1.0 if dt.date() in regional_cal else 0.0 for dt in time_idx])
            else:
                is_holiday = is_holiday_national
            data_matrix[:, i, 15] = is_holiday
            data_matrix[:, i, 16] = (dow >= 5).astype(float)
                
        return data_matrix, time_idx

    def __len__(self):
        return self.tensor_data.shape[0] - self.seq_length - self.pred_horizon + 1

    def __getitem__(self, idx):
        x = self.tensor_data[idx:idx+self.seq_length]
        # Target: [Demand_MW, Excess_MW]
        y = self.tensor_data[idx+self.seq_length : idx+self.seq_length+self.pred_horizon, :, 0:2] 
        # Extract holiday flag from RAW unscaled data (binary 0/1 preserved)
        y_holiday = torch.tensor(
            self.data_matrix[idx+self.seq_length : idx+self.seq_length+self.pred_horizon, :, 15],
            dtype=torch.float32
        )
        # Extract weekend flag from RAW unscaled data (binary 0/1 preserved)
        y_weekend = torch.tensor(
            self.data_matrix[idx+self.seq_length : idx+self.seq_length+self.pred_horizon, :, 16],
            dtype=torch.float32
        )
        
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)  # Changed to keep feature dimension: (Nodes, PredHorizon, 2)
        y_holiday = y_holiday.T  # (Nodes, PredHorizon)
        y_weekend = y_weekend.T  # (Nodes, PredHorizon)
        return x, y, y_holiday, y_weekend
