import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import SpainElectricityDataset
from gnn import GATForecaster
import shutil

def run_2025_inference():
    print("Loading expanded dataset (including 2025 data)...")
    
    # We load data from 2015 to 2025. 
    # Because split="test" takes the final 10%, the test set will be located in late 2024/2025
    dataset = SpainElectricityDataset(split="test", year_start=2015, year_end=2025)
    
    in_channels = 15  # existing checkpoints trained on 15-channel input
    hidden_channels = 64
    pred_horizon = 24
    
    # Load Asymmetric Model
    model_asym = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint_asym = torch.load(os.path.join("data", "models", "best_gat_forecaster_asymmetric.pth"), map_location="cpu")
    model_asym.load_state_dict(checkpoint_asym['state_dict'] if 'state_dict' in checkpoint_asym else checkpoint_asym)
    model_asym.eval()
    
    # Load Holiday Model
    model_hol = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint_hol = torch.load(os.path.join("data", "models", "best_gat_forecaster_holiday.pth"), map_location="cpu")
    model_hol.load_state_dict(checkpoint_hol['state_dict'] if 'state_dict' in checkpoint_hol else checkpoint_hol)
    model_hol.eval()
    
    # Load Weekend Model
    model_wk = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint_wk = torch.load(os.path.join("data", "models", "best_gat_forecaster_weekend.pth"), map_location="cpu")
    model_wk.load_state_dict(checkpoint_wk['state_dict'] if 'state_dict' in checkpoint_wk else checkpoint_wk)
    model_wk.eval()
    
    # Load Hybrid Corrector
    from hybrid_corrector import HybridCorrector
    hybrid = HybridCorrector(model_asym, model_hol, dataset, model_weekend=model_wk)
    hybrid.load_correctors(os.path.join("data", "models", "hybrid_corrector.pkl"))
    
    # Test indices targeting specific days
    step = max(1, len(dataset) // 4)
    indices_to_test = [i * step for i in range(4)]
    
    local_dir = "forecast_results"
    os.makedirs(local_dir, exist_ok=True)
    
    node_idx = 10  # Comunidad de Madrid
    region_name = dataset.node_names[node_idx]
    
    for idx in indices_to_test:
        # Prevent out of bounds if dataset is slightly smaller
        if idx >= len(dataset):
            print(f"Skipping index {idx}, out of bounds.")
            continue
        
        target_datetime = dataset.time_idx[idx + dataset.seq_length]
        date_str = pd.to_datetime(target_datetime).strftime("%Y-%m-%d")
        print(f"Running inference for {date_str} (Index: {idx})")
        
        x, y_true, y_holiday, y_weekend = dataset[idx]  # 4-tuple since weekend feature added
        x_batch = x.unsqueeze(0)
        
        with torch.no_grad():
            y_pred = hybrid.predict(x_batch, dataset.edge_index, dataset.edge_weight, target_date=date_str)
            
        # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
        actual_scaled = y_true[node_idx, :, 0].numpy()
        pred_scaled = y_pred[node_idx].numpy()
        
        # Inverse transform
        def inv_transform(scaled_array, node_idx):
            num_scaled = dataset.num_scaled_features  # 7
            dummy = np.zeros((pred_horizon, num_scaled))
            dummy[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]
            
        actual_mw = inv_transform(actual_scaled, node_idx)
        pred_mw = inv_transform(pred_scaled, node_idx)
        
        mae = np.mean(np.abs(pred_mw - actual_mw))
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_mw, label=f"Actual {region_name} (MW)", marker='o', linewidth=2, color='blue')
        plt.plot(pred_mw, label=f"Hybrid GAT (MAE: {mae:.0f} MW)", marker='^', linewidth=2, linestyle='-.', color='green')
        
        plt.ylim(0, max(max(actual_mw), max(pred_mw)) * 1.2)
        plt.title(f"2025 Forecast vs Actual - {region_name} - {date_str}")
        plt.xlabel("Hours of the Day")
        plt.ylabel("Demand (MW)")
        plt.legend()
        plt.grid(True)
        
        filename = f"forecast_2025_{date_str}.png"
        
        local_path = os.path.join(local_dir, filename)
        plt.savefig(local_path)
        plt.close()
        
        print(f"Saved {filename}")

if __name__ == "__main__":
    run_2025_inference()

