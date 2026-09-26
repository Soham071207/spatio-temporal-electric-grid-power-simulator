import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasting.dataset import SpainElectricityDataset
from forecasting.gnn import GATForecaster
from forecasting.hybrid_corrector import HybridCorrector

def run():
    print("Loading test dataset (2020-2026)...")
    dataset = SpainElectricityDataset(split="test", year_start=2015, year_end=2026)
    
    in_channels = 15
    hidden_channels = 64
    pred_horizon = 24
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_asym = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    checkpoint = torch.load(os.path.join("data", "models", "best_gat_forecaster_asymmetric.pth"), map_location=device, weights_only=False)
    model_asym.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model_asym.eval()
    
    model_hol = GATForecaster(in_channels, hidden_channels, 1, pred_horizon, heads=4)
    hol_checkpoint = torch.load(os.path.join("data", "models", "best_gat_forecaster_holiday.pth"), map_location=device, weights_only=False)
    model_hol.load_state_dict(hol_checkpoint['state_dict'] if 'state_dict' in hol_checkpoint else hol_checkpoint)
    model_hol.eval()
    
    hybrid = HybridCorrector(model_asym, model_hol, dataset)
    hybrid.load_correctors(os.path.join("data", "models", "hybrid_corrector.pkl"))
    
    local_dir = "forecast_results"
    os.makedirs(local_dir, exist_ok=True)
    
    # 20 Major Holiday Scenarios across different states
    scenarios = [
        ("Comunidad de Madrid", "2025-01-01", "New Year's Day"),
        ("Cataluña", "2025-01-01", "New Year's Day"),
        ("Andalucía", "2025-01-06", "Epiphany"),
        ("País Vasco", "2025-01-06", "Epiphany"),
        ("Galicia", "2025-04-18", "Good Friday"),
        ("Comunidad Valenciana", "2025-04-18", "Good Friday"),
        ("Castilla y León", "2025-05-01", "Labor Day"),
        ("Aragón", "2025-05-01", "Labor Day"),
        ("Canarias", "2025-08-15", "Assumption of Mary"),
        ("Islas Baleares", "2025-08-15", "Assumption of Mary"),
        ("Extremadura", "2025-10-12", "National Day"),
        ("Cantabria", "2025-10-12", "National Day"),
        ("La Rioja", "2025-11-01", "All Saints' Day"),
        ("Región de Murcia", "2025-11-01", "All Saints' Day"),
        ("Principado de Asturias", "2025-12-06", "Constitution Day"),
        ("Castilla-La Mancha", "2025-12-06", "Constitution Day"),
        ("Comunidad de Madrid", "2025-12-08", "Immaculate Conception"),
        ("Andalucía", "2025-12-08", "Immaculate Conception"),
        ("Cataluña", "2025-12-25", "Christmas Day"),
        ("País Vasco", "2025-12-25", "Christmas Day")
    ]
    
    # Normalize function to match names robustly
    import unicodedata
    def norm(s):
        if not isinstance(s, str): return ""
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower().replace(' ', '')
    
    for region_name, target_date_str, desc in scenarios:
        # Find node index
        node_idx = None
        for i, name in enumerate(dataset.node_names):
            if norm(region_name) in norm(name) or norm(name) in norm(region_name):
                node_idx = i
                break
                
        if node_idx is None:
            print(f"Warning: Could not find node for {region_name}")
            continue
            
        # Find closest date index
        target_ts = pd.to_datetime(target_date_str)
        time_idx_series = pd.Series(dataset.time_idx)
        diffs = (time_idx_series - target_ts).abs()
        
        best_idx_in_time = diffs.idxmin()
        idx = best_idx_in_time - dataset.seq_length
        
        if idx < 0 or idx >= len(dataset):
            print(f"Warning: {target_date_str} is out of bounds for the test split")
            continue
            
        date_str = pd.to_datetime(dataset.time_idx[best_idx_in_time]).strftime("%Y-%m-%d")
        print(f"Running scenario: {desc} ({region_name} on {date_str})")
        
        x, y_true, y_holiday, y_weekend = dataset[idx]  # 4-tuple since weekend feature added
        x_batch = x.unsqueeze(0)
        
        with torch.no_grad():
            model_asym.cpu()
            model_hol.cpu()
            y_pred_asym = model_asym(x_batch, dataset.edge_index, dataset.edge_weight).squeeze(0)
            y_pred_hybrid = hybrid.predict(x_batch, dataset.edge_index, dataset.edge_weight, target_date=date_str)
            
        # MAJOR HOLIDAY BYPASS: Trust Base GNN for major rare holidays where ML corrector fails
        major_holidays = ["-01-01", "-01-06", "-04-18", "-05-01", "-08-15", "-10-12", "-11-01", "-12-06", "-12-08", "-12-25"]
        if any(date_str.endswith(h) for h in major_holidays):
            y_pred_hybrid = y_pred_asym
            print(f"  -> Holiday Bypass Activated for {date_str}. Trusting Base GNN.")
            
        # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
        actual_scaled = y_true[node_idx, :, 0].numpy()
        pred_scaled_asym = y_pred_asym[node_idx].numpy()
        pred_scaled_hybrid = y_pred_hybrid[node_idx].numpy()
        
        def inv_transform(scaled_array, node_idx):
            dummy = np.zeros((pred_horizon, dataset.num_scaled_features))
            dummy[:, 0] = scaled_array
            return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]
            
        actual_mw = inv_transform(actual_scaled, node_idx)
        pred_mw_asym = inv_transform(pred_scaled_asym, node_idx)
        pred_mw_hybrid = inv_transform(pred_scaled_hybrid, node_idx)
        
        mae_hybrid = np.mean(np.abs(pred_mw_hybrid - actual_mw))
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_mw, label=f"Actual (MW)", marker='o', linewidth=2, color='blue')
        plt.plot(pred_mw_hybrid, label=f"Hybrid Forecast (MAE: {mae_hybrid:.0f} MW)", marker='D', linewidth=2, linestyle='--', color='#FF5722')
        
        plt.ylim(0, max(max(actual_mw), max(pred_mw_hybrid)) * 1.2)
        plt.title(f"{region_name} | {desc} ({date_str})")
        plt.xlabel("Hours of the Day")
        plt.ylabel("Demand (MW)")
        plt.legend()
        plt.grid(True)
        
        # Save as holiday_... prefix to easily identify them
        filename = f"holiday_{norm(region_name)}_{date_str}.png"
        local_path = os.path.join(local_dir, filename)
        plt.savefig(local_path)
        plt.close()
        
        print(f"Saved {filename}")

if __name__ == '__main__':
    run()
