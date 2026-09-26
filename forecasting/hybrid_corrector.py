"""
Hybrid ML/DL Corrector
======================
Two-stage forecasting pipeline:
  Stage 1 (DL): GAT Graph Neural Network produces raw 24-hour predictions
  Stage 2 (ML): Gradient Boosted Tree learns to correct GAT errors using
                rich tabular features (hour, holiday, density, temperature, etc.)

The tree model learns *when* and *how much* the GAT over/under-predicts,
producing dramatically better final forecasts — especially on holidays,
weekends, and low-density regions where the GAT struggles most.
"""
import os
import numpy as np
import torch
import pandas as pd
import holidays
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader


class HybridCorrector:
    """
    ML/DL hybrid: GAT predictions + tabular features --> corrected forecast.
    
    Now supports 3 GAT models:
      - model_asym: base asymmetric loss model
      - model_hol: holiday-focused model
      - model_weekend: weekend-focused model (optional)
    
    Training:
        corrector = HybridCorrector(model_asym, model_hol, dataset, model_weekend=model_wk)
        corrector.fit(train_dataset, val_dataset)
        corrector.save("data/models/hybrid_corrector.pkl")
    
    Inference:
        corrector.predict(x_batch, edge_index, edge_weight, target_date, y_holiday, y_weekend)
    """

    def __init__(self, model_asym, model_hol, dataset, model_weekend=None):
        self.model_asym = model_asym
        self.model_hol = model_hol
        self.model_weekend = model_weekend
        self.dataset = dataset
        self.num_nodes = dataset.num_nodes
        self.node_names = dataset.node_names
        
        # One HistGBT per node — each node has different demand patterns
        self.correctors = {}
        
        print(f"[Hybrid] Initialized for {self.num_nodes} nodes (weekend_model={'yes' if model_weekend else 'no'})", flush=True)

    def _build_features(self, gat_pred_asym, gat_pred_hol, gat_pred_weekend, x_seq, node_idx, hour, target_date=None, y_holiday_val=0.0, y_weekend_val=0.0):
        f = np.zeros(20)
        f[0] = gat_pred_asym
        f[1] = gat_pred_hol
        f[2] = gat_pred_weekend if gat_pred_weekend is not None else 0.5 * gat_pred_asym + 0.5 * gat_pred_hol
        f[3] = 0.5 * gat_pred_asym + 0.5 * gat_pred_hol  # base blend
        f[4] = gat_pred_asym - gat_pred_hol  # asym-hol disagreement
        f[5] = (gat_pred_asym - (gat_pred_weekend if gat_pred_weekend is not None else gat_pred_asym))  # asym-weekend disagreement
        f[6] = hour
        f[7] = np.sin(2 * np.pi * hour / 24.0)
        f[8] = np.cos(2 * np.pi * hour / 24.0)
        
        if target_date is not None:
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date)
            f[9] = target_date.dayofweek
            f[10] = target_date.month
            f[11] = 1.0 if target_date.dayofweek >= 5 else 0.0  # is_weekend
            
        f[12] = y_holiday_val
        f[13] = y_weekend_val
        
        if x_seq is not None:
            # Static node features extracted dynamically from historical context
            f[14] = float(x_seq[-1, 5]) # population density
            f[15] = float(x_seq[-1, 6]) # industry index
            
            last_6h = x_seq[-6:, :]  
            f[16] = float(last_6h[:, 1].mean())  # avg temperature
            f[17] = float(last_6h[:, 0].mean())   # avg demand
            if last_6h.shape[0] >= 2:
                demand_vals = last_6h[:, 0]
                f[18] = float(demand_vals[-1] - demand_vals[0])  # demand trend
            # is_weekend from input sequence (feature 15 in dataset)
            f[19] = float(x_seq[-1, 15]) if x_seq.shape[1] > 15 else 0.0
        
        return f

    @torch.no_grad()
    def fit(self, train_dataset, val_dataset=None, max_train=20000, max_val=2000):
        """
        Generate GAT predictions on training data using fast batches, then fit per-node GBT correctors.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Hybrid] Stage 1: Generating GAT predictions (max_train={max_train}) on {device}...", flush=True)
        
        node_X = {i: [] for i in range(self.num_nodes)}
        node_y = {i: [] for i in range(self.num_nodes)}
        
        datasets_to_use = [("train", train_dataset, max_train)]
        if val_dataset is not None:
            datasets_to_use.append(("val", val_dataset, max_val))
            
        self.model_asym.to(device)
        self.model_hol.to(device)
        if self.model_weekend is not None:
            self.model_weekend.to(device)
        
        for split_name, ds, max_samples in datasets_to_use:
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            count = 0
            for batch in loader:
                # Handle both 3-tuple (old) and 4-tuple (new) datasets
                if len(batch) == 4:
                    x, y_true, y_holiday, y_weekend = batch
                else:
                    x, y_true, y_holiday = batch
                    y_weekend = torch.zeros_like(y_holiday)
                batch_size = x.size(0)
                if count >= max_samples:
                    break
                
                x_gpu = x.to(device)
                edge_index_gpu = ds.edge_index.to(device)
                edge_weight_gpu = ds.edge_weight.to(device) if ds.edge_weight is not None else None
                
                # Get GAT predictions for entire batch at once (slice to 16 features — models trained on 16-ch)
                x_gat = x_gpu[:, :, :, :16] if x_gpu.shape[-1] > 16 else x_gpu
                p_asym = self.model_asym(x_gat, edge_index_gpu, edge_weight_gpu).cpu().numpy()  
                p_hol = self.model_hol(x_gat, edge_index_gpu, edge_weight_gpu).cpu().numpy()
                if self.model_weekend is not None:
                    p_wk = self.model_weekend(x_gat, edge_index_gpu, edge_weight_gpu).cpu().numpy()
                else:
                    p_wk = None
                y_actual = y_true.numpy()  
                x_np = x.numpy()
                
                for b in range(batch_size):
                    if count + b >= max_samples: break
                    sample_idx = count + b
                    if hasattr(ds, 'time_idx') and (sample_idx + ds.seq_length) < len(ds.time_idx):
                        target_dt = pd.to_datetime(ds.time_idx[sample_idx + ds.seq_length])
                    else:
                        target_dt = None
                        
                    for node_idx in range(self.num_nodes):
                        x_seq = x_np[b, node_idx]
                        for hour in range(24):
                            yh_val = float(y_holiday[b, node_idx, hour].item())
                            yw_val = float(y_weekend[b, node_idx, hour].item())
                            p_wk_val = p_wk[b, node_idx, hour] if p_wk is not None else None
                            feat = self._build_features(
                                p_asym[b, node_idx, hour],
                                p_hol[b, node_idx, hour],
                                p_wk_val,
                                x_seq, node_idx, hour,
                                target_date=target_dt,
                                y_holiday_val=yh_val,
                                y_weekend_val=yw_val
                            )
                            node_X[node_idx].append(feat)
                            node_y[node_idx].append(y_actual[b, node_idx, hour, 0])  # demand channel only
                
                count += batch_size
                if count % 2048 == 0 or count >= max_samples:
                    print(f"  Processed {count} samples from {split_name}...", flush=True)
            
            print(f"  {split_name}: {count} samples processed", flush=True)
        
        # Stage 2: Train per-node GBT correctors
        print("\n[Hybrid] Stage 2: Training per-node Gradient Boosted Tree correctors...", flush=True)
        
        for node_idx in range(self.num_nodes):
            X = np.array(node_X[node_idx])
            y = np.array(node_y[node_idx])
            
            if len(X) == 0:
                print(f"  Node {node_idx} ({self.node_names[node_idx]}): SKIP (no data)", flush=True)
                continue
            
            gbr = HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                max_bins=128,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42
            )
            gbr.fit(X, y)
            self.correctors[node_idx] = gbr
            
            # Quick train score
            train_pred = gbr.predict(X)
            mae = np.mean(np.abs(train_pred - y))
            print(f"  Node {node_idx:2d} ({self.node_names[node_idx]:30s}): "
                  f"Train MAE={mae:.4f} (scaled) | {len(X)} samples | "
                  f"{gbr.n_iter_} iterations", flush=True)
        
        print(f"\n[Hybrid] Training complete. {len(self.correctors)} node correctors fitted.", flush=True)

    @torch.no_grad()
    def predict(self, x_batch, edge_index, edge_weight, target_date=None, y_holiday=None, y_weekend=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Only move models if not already on the target device (avoids overhead on repeated hover calls)
        if next(self.model_asym.parameters()).device.type != device.type:
            self.model_asym.to(device)
            self.model_hol.to(device)
            if self.model_weekend is not None:
                self.model_weekend.to(device)
        
        x_gpu = x_batch.to(device)
        edge_index_gpu = edge_index.to(device)
        edge_weight_gpu = edge_weight.to(device) if edge_weight is not None else None
        
        # Slice to 16 features for GAT models (trained on 16-ch dataset)
        x_gat = x_gpu[:, :, :, :16] if x_gpu.shape[-1] > 16 else x_gpu
        p_asym = self.model_asym(x_gat, edge_index_gpu, edge_weight_gpu).cpu().squeeze(0).numpy()
        p_hol = self.model_hol(x_gat, edge_index_gpu, edge_weight_gpu).cpu().squeeze(0).numpy()
        if self.model_weekend is not None:
            p_wk = self.model_weekend(x_gat, edge_index_gpu, edge_weight_gpu).cpu().squeeze(0).numpy()
        else:
            p_wk = None
        x_np = x_batch.cpu().squeeze(0).numpy()  # (N, S, F)
        
        # Determine day type for adaptive blending and clamping
        is_weekend_day = False
        is_holiday_day = False
        if target_date is not None:
            dt = pd.to_datetime(target_date) if isinstance(target_date, str) else target_date
            is_weekend_day = dt.dayofweek >= 5
        if y_holiday is not None and y_holiday.max().item() > 0.5:
            is_holiday_day = True
        if y_weekend is not None and y_weekend.max().item() > 0.5:
            is_weekend_day = True
        
        result = np.zeros((self.num_nodes, 24))
        
        for node_idx in range(self.num_nodes):
            if node_idx not in self.correctors:
                # Fallback: context-aware blend
                if is_weekend_day and p_wk is not None:
                    result[node_idx] = 0.3 * p_asym[node_idx] + 0.1 * p_hol[node_idx] + 0.6 * p_wk[node_idx]
                elif is_holiday_day:
                    result[node_idx] = 0.2 * p_asym[node_idx] + 0.6 * p_hol[node_idx] + (0.2 * p_wk[node_idx] if p_wk is not None else 0.2 * p_asym[node_idx])
                else:
                    result[node_idx] = 0.7 * p_asym[node_idx] + 0.3 * p_hol[node_idx]
                continue
            
            x_seq = x_np[node_idx]  # (S, F)
            features = []
            for hour in range(24):
                yh_val = float(y_holiday[0, node_idx, hour].item()) if y_holiday is not None else 0.0
                yw_val = float(y_weekend[0, node_idx, hour].item()) if y_weekend is not None else (1.0 if is_weekend_day else 0.0)
                p_wk_val = p_wk[node_idx, hour] if p_wk is not None else None
                feat = self._build_features(
                    p_asym[node_idx, hour],
                    p_hol[node_idx, hour],
                    p_wk_val,
                    x_seq, node_idx, hour,
                    target_date=target_date,
                    y_holiday_val=yh_val,
                    y_weekend_val=yw_val
                )
                features.append(feat)
            
            X = np.array(features)  # (24, 20)
            corrected = self.correctors[node_idx].predict(X)

            # ADAPTIVE CLAMP: wider corrections for weekends/holidays
            # where the GAT base models systematically over-predict
            if is_weekend_day and p_wk is not None:
                gat_blend = 0.3 * p_asym[node_idx] + 0.1 * p_hol[node_idx] + 0.6 * p_wk[node_idx]
            elif is_holiday_day:
                gat_blend = 0.2 * p_asym[node_idx] + 0.6 * p_hol[node_idx] + (0.2 * p_wk[node_idx] if p_wk is not None else 0.2 * p_asym[node_idx])
            else:
                gat_blend = 0.7 * p_asym[node_idx] + 0.3 * p_hol[node_idx]
            
            delta = corrected - gat_blend
            
            # Adaptive max_delta: allow much wider corrections on non-normal days
            if is_weekend_day or is_holiday_day:
                max_delta = 0.60  # ~100-150 MW correction allowed
            else:
                max_delta = 0.25  # ~40-60 MW correction on normal days
            
            delta_clamped = np.clip(delta, -max_delta, max_delta)
            result[node_idx] = gat_blend + delta_clamped

            
        return torch.tensor(result, dtype=torch.float32)

    def save(self, path):
        """Save the trained correctors to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            'correctors': self.correctors,
            'node_names': self.node_names,
            'num_nodes': self.num_nodes,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"[Hybrid] Saved correctors to {path}", flush=True)

    def load_correctors(self, path):
        """Load pre-trained correctors from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.correctors = data['correctors']
        print(f"[Hybrid] Loaded {len(self.correctors)} node correctors from {path}", flush=True)


def build_hybrid(model_asym, model_hol, train_dataset, val_dataset, test_dataset,
                 save_path="data/models/hybrid_corrector.pkl"):
    """
    Convenience function: build, train, save, and return a HybridCorrector.
    """
    corrector = HybridCorrector(model_asym, model_hol, test_dataset)
    corrector.fit(train_dataset, val_dataset, max_train=20000, max_val=5000)
    corrector.save(save_path)
    return corrector

if __name__ == '__main__':
    import torch
    from dataset import SpainElectricityDataset
    from gnn import GATForecaster
    
    print('Loading datasets...')
    train_ds = SpainElectricityDataset(split='train', year_start=2015)
    val_ds = SpainElectricityDataset(split='val', year_start=2015)
    test_ds = SpainElectricityDataset(split='test', year_start=2015)
    
    model_asym = GATForecaster(16, 64, 1, 24, heads=4, dropout=0.2)
    model_hol = GATForecaster(16, 64, 1, 24, heads=4, dropout=0.2)
    model_wk = GATForecaster(16, 64, 1, 24, heads=4, dropout=0.2)
    
    model_asym.load_state_dict(torch.load('data/models/best_gat_forecaster_asymmetric.pth', map_location='cpu')['state_dict'])
    model_hol.load_state_dict(torch.load('data/models/best_gat_forecaster_holiday.pth', map_location='cpu')['state_dict'])
    model_wk.load_state_dict(torch.load('data/models/best_gat_forecaster_weekend.pth', map_location='cpu')['state_dict'])
    
    corrector = HybridCorrector(model_asym, model_hol, test_ds, model_weekend=model_wk)
    corrector.fit(train_ds, val_ds, max_train=25000, max_val=5000)
    corrector.save('data/models/hybrid_corrector.pkl')

