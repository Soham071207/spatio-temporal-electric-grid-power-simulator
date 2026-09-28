import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasting.dataset import SpainElectricityDataset
from forecasting.gnn import GATForecaster
from forecasting.hybrid_corrector import HybridCorrector

def run_scenarios():
    print("Loading datasets...")
    csv_path = os.path.join("data", "processed", "grid_telemetry", "demand_regional_disaggregated.csv")
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df = df[(df['datetime'].dt.year >= 2020) & (df['datetime'].dt.year < 2027)]
    df = df.sort_values(['datetime', 'region']).drop_duplicates(subset=['datetime', 'region'])
    
    unique_dates = df['datetime'].unique()
    unique_dates = pd.Series(pd.to_datetime(unique_dates)).sort_values().values
    
    T = len(unique_dates)
    val_end = int(T * 0.9)
    print(f"Total hours: {T}, Test set starts at absolute index: {val_end} (Date: {unique_dates[val_end]})")
    
    # Load datasets
    train_dataset = SpainElectricityDataset(split="train", year_start=2020, year_end=2027)
    val_dataset = SpainElectricityDataset(split="val", year_start=2020, year_end=2027)
    test_dataset = SpainElectricityDataset(split="test", year_start=2020, year_end=2027)
    
    in_channels = 16
    hidden_channels = 64
    pred_horizon = 24
    
    # Load asymmetric model
    model_asym = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint_path = os.path.join("data", "models", "best_gat_forecaster_asymmetric.pth")
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_asym.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    asym_path = os.path.join("data", "models", "best_gat_forecaster_asymmetric.pth")
    checkpoint_asym = torch.load(asym_path, map_location=device, weights_only=False)
    model_asym.load_state_dict(checkpoint_asym['state_dict'] if isinstance(checkpoint_asym, dict) and 'state_dict' in checkpoint_asym else checkpoint_asym)
    model_asym.eval()
    
    # Load holiday-focused model
    model_hol = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    hol_path = os.path.join("data", "models", "best_gat_forecaster_holiday.pth")
    checkpoint_hol = torch.load(hol_path, map_location=device, weights_only=False)
    model_hol.load_state_dict(checkpoint_hol['state_dict'] if isinstance(checkpoint_hol, dict) and 'state_dict' in checkpoint_hol else checkpoint_hol)
    model_hol.eval()
    
    # Build Hybrid Corrector
    print("\n--- Building ML/DL Hybrid Corrector ---")
    corrector_path = os.path.join("data", "models", "hybrid_corrector.pkl")
    hybrid = HybridCorrector(model_asym, model_hol, test_dataset)
    
    if os.path.exists(corrector_path):
        hybrid.load_correctors(corrector_path)
    else:
        print("Pre-trained corrector not found. Training now (takes a minute)...")
        hybrid.fit(train_dataset, val_dataset)
        hybrid.save(corrector_path)
    print("--- Hybrid Corrector Ready ---\n")
    
    scenarios = {
        "Normal Day (Spring Wed)": "2025-06-11",
        "Weekend (Sunday)": "2025-08-10",
        "Heat Wave Day (Mid-July)": "2025-07-20",
        "Holiday (Assumption of Mary)": "2025-08-15",
        "Autumn Weekday (Thursday)": "2025-10-09",
        "Summer Weekday (August)": "2025-08-05",
        "Early Autumn (September)": "2025-09-15",
        "Christmas Eve": "2025-12-24",
        "Winter Weekday (Dec)": "2025-12-10",
        "Holiday (Constitution Day)": "2025-12-06"
    }
    
    local_dir = "forecast_results"
    os.makedirs(local_dir, exist_ok=True)
    
    node_idx = 7  # Principado de Asturias
    region_name = test_dataset.node_names[node_idx]
    
    generated_files = []
    
    for label, date_str in scenarios.items():
        target_dt = pd.to_datetime(f"{date_str} 00:00:00")
        abs_idx = np.argmin(np.abs(pd.Series(unique_dates) - target_dt))
        actual_dt = unique_dates[abs_idx]
        test_idx = abs_idx - val_end
        
        if test_idx < 0 or test_idx >= len(test_dataset):
            print(f"Skipping {label} ({date_str}): Not in test set range.")
            continue
            
        print(f"Running {label} for {actual_dt} (Test Idx: {test_idx})")
        
        x, y_true, y_holiday, y_weekend = test_dataset[test_idx]
        x_batch = x.unsqueeze(0)
        
        with torch.no_grad():
            model_asym.cpu()
            model_hol.cpu()
            y_pred_asym = model_asym(x_batch, test_dataset.edge_index, test_dataset.edge_weight).squeeze(0)
            y_pred_hybrid = hybrid.predict(x_batch, test_dataset.edge_index, test_dataset.edge_weight, target_date=date_str, y_holiday=y_holiday.unsqueeze(0))
            
        actual_scaled = y_true[node_idx].numpy()
        pred_scaled_asym = y_pred_asym[node_idx].numpy()
        pred_scaled_hybrid = y_pred_hybrid[node_idx].numpy()
        
        def inv_transform(scaled_array):
            num_scaled = test_dataset.num_scaled_features
            dummy = np.zeros((pred_horizon, num_scaled))
            dummy[:, 0] = scaled_array
            return test_dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]
            
        actual_mw = inv_transform(actual_scaled)
        pred_mw_asym = inv_transform(pred_scaled_asym)
        pred_mw_hybrid = inv_transform(pred_scaled_hybrid)
        
        # Compute error metrics
        mae_asym = np.mean(np.abs(pred_mw_asym - actual_mw))
        mae_hybrid = np.mean(np.abs(pred_mw_hybrid - actual_mw))
        improvement = ((mae_asym - mae_hybrid) / mae_asym) * 100
        
        # Check if the target day has holiday flag set for this region
        is_hol = y_holiday[node_idx].max().item() > 0.5
        tag = "[HOLIDAY]" if is_hol else "[Normal]"
        arrow = "v" if improvement > 0 else "^"
        print(f"  {tag} | MAE Asym: {mae_asym:.1f} MW -> Hybrid: {mae_hybrid:.1f} MW "
              f"({arrow}{abs(improvement):.1f}%)")
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual_mw, label=f"Actual {region_name} (MW)", 
                 marker='o', linewidth=2.5, color='#2196F3', markersize=6)
        plt.plot(pred_mw_asym, label=f"GAT Asymmetric (MAE: {mae_asym:.0f} MW)", 
                 marker='^', linewidth=1.5, linestyle='-.', color='#4CAF50', alpha=0.6)
        plt.plot(pred_mw_hybrid, label=f"ML/DL Hybrid (MAE: {mae_hybrid:.0f} MW)", 
                 marker='D', linewidth=2.5, linestyle='--', color='#FF5722', markersize=5)
        
        plt.fill_between(range(24), actual_mw, pred_mw_hybrid, alpha=0.1, color='#FF5722')
        
        plt.ylim(0, max(max(actual_mw), max(pred_mw_asym), max(pred_mw_hybrid)) * 1.15)
        hol_tag = " *HOLIDAY*" if is_hol else ""
        plt.title(f"{label}: {date_str} ({region_name}){hol_tag}", fontsize=14, fontweight='bold')
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Demand (MW)", fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"scenario_{date_str}_v2.png"
        local_path = os.path.join(local_dir, filename)
        plt.savefig(local_path, dpi=150)
        plt.close()
        
        print(f"  Saved {filename}")
        generated_files.append((label, date_str, filename))
        
    print("\nDONE.")
    return generated_files

if __name__ == "__main__":
    run_scenarios()
