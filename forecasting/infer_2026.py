import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import SpainElectricityDataset
from gnn import GATForecaster

def run_2026_inference():
    print("Loading test dataset (2020-2026) -> Val split ends in late 2024, Test split is ~2025/2026")
    
    # We load data from 2020 to 2026. 
    dataset = SpainElectricityDataset(split="test", year_start=2015, year_end=2026)
    
    in_channels = 15  # existing checkpoints trained on 15-channel input
    hidden_channels = 64
    pred_horizon = 24
    
    # Load Asymmetric Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_asym = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint = torch.load(os.path.join("data", "models", "best_gat_forecaster_asymmetric.pth"), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_asym.load_state_dict(checkpoint['state_dict'])
    else:
        model_asym.load_state_dict(checkpoint)
    model_asym.eval()
    
    # Load Holiday Model
    model_hol = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    hol_checkpoint = torch.load(os.path.join("data", "models", "best_gat_forecaster_holiday.pth"), map_location=device, weights_only=False)
    if isinstance(hol_checkpoint, dict) and 'state_dict' in hol_checkpoint:
        model_hol.load_state_dict(hol_checkpoint['state_dict'])
    else:
        model_hol.load_state_dict(hol_checkpoint)
    model_hol.eval()
    
    from hybrid_corrector import HybridCorrector
    hybrid = HybridCorrector(model_asym, model_hol, dataset)
    hybrid_path = os.path.join("data", "models", "hybrid_corrector.pkl")
    if os.path.exists(hybrid_path):
        hybrid.load_correctors(hybrid_path)
    else:
        print("Warning: hybrid_corrector.pkl not found!")
    
    local_dir = "forecast_results"
    os.makedirs(local_dir, exist_ok=True)
    
    node_idx = 10  # Comunidad de Madrid
    region_name = dataset.node_names[node_idx]
    
    # Pick 4 evenly spaced test indices
    step = max(1, len(dataset) // 4)
    indices_to_test = [i * step for i in range(4)]
    
    for idx in indices_to_test:
        if idx >= len(dataset):
            idx = len(dataset) - 1
            
        target_datetime = dataset.time_idx[idx + dataset.seq_length]
        date_str = pd.to_datetime(target_datetime).strftime("%Y-%m-%d")
        print(f"Running inference for {date_str} (Index: {idx})")
        
        x, y_true, y_holiday, y_weekend = dataset[idx]
        x_batch = x.unsqueeze(0)
        
        with torch.no_grad():
            model_asym.cpu()
            model_hol.cpu()
            y_pred_asym = model_asym(x_batch, dataset.edge_index, dataset.edge_weight).squeeze(0)
            y_pred_hybrid = hybrid.predict(x_batch, dataset.edge_index, dataset.edge_weight, target_date=date_str)
            
        # MAJOR HOLIDAY BYPASS: Trust Base GNN for major rare holidays where ML corrector fails
        # Good Friday is movable — compute it dynamically for the forecast year.
        import holidays as _holidays
        _year = pd.to_datetime(date_str).year
        _es = _holidays.Spain(years=_year)
        _good_friday = [str(d) for d, name in _es.items() if 'Good Friday' in name or 'Viernes Santo' in name]
        major_holidays = ["-12-25", "-01-01", "-01-06", "-05-01", "-08-15"] + _good_friday
        if any(date_str == h or date_str.endswith(h) for h in major_holidays):
            y_pred_hybrid = y_pred_asym
            print(f"  -> Holiday Bypass Activated for {date_str}. Trusting Base GNN.")
            
            
        # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
        actual_scaled = y_true[node_idx, :, 0].numpy()
        pred_scaled_asym = y_pred_asym[node_idx].numpy()
        pred_scaled_hybrid = y_pred_hybrid[node_idx].numpy()
        
        # Inverse transform
        def inv_transform(scaled_array, node_idx):
            num_scaled = dataset.num_scaled_features
            dummy = np.zeros((pred_horizon, num_scaled))
            dummy[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]
            
        actual_mw = inv_transform(actual_scaled, node_idx)
        pred_mw_asym = inv_transform(pred_scaled_asym, node_idx)
        pred_mw_hybrid = inv_transform(pred_scaled_hybrid, node_idx)
        
        mae_asym = np.mean(np.abs(pred_mw_asym - actual_mw))
        mae_hybrid = np.mean(np.abs(pred_mw_hybrid - actual_mw))
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_mw, label=f"Actual {region_name} (MW)", marker='o', linewidth=2, color='blue')
        plt.plot(pred_mw_asym, label=f"GAT Asymmetric (MAE: {mae_asym:.0f} MW)", marker='^', linewidth=1.5, linestyle='-.', color='green')
        plt.plot(pred_mw_hybrid, label=f"ML/DL Hybrid (MAE: {mae_hybrid:.0f} MW)", marker='D', linewidth=2, linestyle='--', color='#FF5722')
        
        plt.ylim(0, max(max(actual_mw), max(pred_mw_asym), max(pred_mw_hybrid)) * 1.2)
        plt.title(f"LIVE 2026 DATA: Forecast vs Actual - {date_str}")
        plt.xlabel("Hours of the Day")
        plt.ylabel("Demand (MW)")
        plt.legend()
        plt.grid(True)
        
        filename = f"live_2026_{date_str}_v2.png"
        
        local_path = os.path.join(local_dir, filename)
        plt.savefig(local_path)
        plt.close()
        
        print(f"Saved {filename}")

if __name__ == "__main__":
    run_2026_inference()
