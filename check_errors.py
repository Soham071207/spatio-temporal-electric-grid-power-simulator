import os
import torch
import numpy as np
import pandas as pd
from forecasting.dataset import SpainElectricityDataset
from forecasting.gnn import GATForecaster

def check_errors():
    dataset = SpainElectricityDataset(split="test", year_start=2020, year_end=2025)
    
    in_channels = 15
    hidden_channels = 64
    pred_horizon = 24
    
    model_asym = GATForecaster(15, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint = torch.load(os.path.join("data", "models", "best_gat_forecaster_asymmetric.pth"))
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_asym.load_state_dict(checkpoint['state_dict'])
    else:
        model_asym.load_state_dict(checkpoint)
    model_asym.eval()

    target_date = pd.to_datetime("2025-07-20 00:00:00")
    
    # find the index for 2025-07-20
    time_idx_series = pd.Series(dataset.time_idx)
    idx_matches = time_idx_series[time_idx_series == target_date].index
    if len(idx_matches) == 0:
        print("Date not found in test set!")
        # Let's just find the closest date in dataset
        print(f"Test set start: {dataset.time_idx[0]}, end: {dataset.time_idx[-1]}")
        
        # Or search in the whole data
        dataset = SpainElectricityDataset(split="train", year_start=2020, year_end=2025)
        time_idx_series = pd.Series(dataset.time_idx)
        idx_matches = time_idx_series[time_idx_series == target_date].index
        if len(idx_matches) == 0:
            print("Date not found in dataset at all")
            return
        else:
            print(f"Found in train set at index {idx_matches[0]}")
            idx = idx_matches[0] - dataset.seq_length
            
    else:
        print(f"Found in test set at index {idx_matches[0]}")
        idx = idx_matches[0] - dataset.seq_length
        
    x, y_true, y_holiday, y_weekend = dataset[idx]  # 4-tuple since weekend feature added
    x_batch = x.unsqueeze(0)
    
    with torch.no_grad():
        y_pred_asym = model_asym(x_batch, dataset.edge_index, dataset.edge_weight).squeeze(0)
        
    print(f"Error evaluation for {target_date.strftime('%Y-%m-%d')}")
    
    for node_idx, region_name in enumerate(dataset.node_names):
        # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
        actual_scaled = y_true[node_idx, :, 0].numpy()
        pred_scaled_asym = y_pred_asym[node_idx].numpy()
        
        def inv_transform(scaled_array, node_idx):
            num_scaled = dataset.num_scaled_features
            dummy = np.zeros((pred_horizon, num_scaled))
            dummy[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]
            
        actual_mw = inv_transform(actual_scaled, node_idx)
        pred_mw_asym = inv_transform(pred_scaled_asym, node_idx)
        
        mape = np.mean(np.abs((actual_mw - pred_mw_asym) / actual_mw)) * 100
        print(f"{region_name:25s} MAPE: {mape:8.2f}% | Actual: {np.mean(actual_mw):.1f} | Pred: {np.mean(pred_mw_asym):.1f}")

if __name__ == "__main__":
    check_errors()
