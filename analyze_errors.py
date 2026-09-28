import os
import torch
import numpy as np
import pandas as pd
from forecasting.dataset import SpainElectricityDataset, POPULATION_DENSITY, INDUSTRY_INDEX, normalize_name
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
    
    # We will test 50 random samples evenly distributed
    step = max(1, len(dataset) // 50)
    indices_to_test = [i * step for i in range(50)]
    
    results = []
    
    def inv_transform(scaled_array, node_idx):
        dummy = np.zeros((pred_horizon, dataset.num_scaled_features))
        dummy[:, 0] = scaled_array
        return dataset.scaler.inverse_transform(dummy, node_idx=node_idx)[:, 0]

    for node_idx in range(dataset.num_nodes):
        region_name = dataset.node_names[node_idx]
        norm_name = normalize_name(region_name)
        
        # Match Density and Industry
        density = 50
        industry = 0.1
        for k, v in POPULATION_DENSITY.items():
            if normalize_name(k) in norm_name or norm_name in normalize_name(k):
                density = v
                break
        for k, v in INDUSTRY_INDEX.items():
            if normalize_name(k) in norm_name or norm_name in normalize_name(k):
                industry = v
                break
                
        results.append({
            'node_idx': node_idx,
            'region': region_name,
            'density': density,
            'industry': industry,
            'maes': [],
            'mapes': [],
            'actual_means': []
        })
        
    for idx in indices_to_test:
        if idx >= len(dataset): continue
        x, y_true, y_holiday, y_weekend = dataset[idx]  # 4-tuple since weekend feature added
        x_batch = x.unsqueeze(0)
        
        with torch.no_grad():
            model_asym.cpu()
            model_hol.cpu()
            y_pred_asym = model_asym(x_batch, dataset.edge_index, dataset.edge_weight).squeeze(0)
            target_date_str = pd.to_datetime(dataset.time_idx[idx + dataset.seq_length]).strftime("%Y-%m-%d")
            y_pred_hybrid = hybrid.predict(x_batch, dataset.edge_index, dataset.edge_weight, target_date=target_date_str)
            
            major_holidays = ["-01-01", "-01-06", "-04-18", "-05-01", "-08-15", "-10-12", "-11-01", "-12-06", "-12-08", "-12-25"]
            if any(target_date_str.endswith(h) for h in major_holidays):
                y_pred_hybrid = y_pred_asym
        
        for node_idx in range(dataset.num_nodes):
            # y_true shape: (N, PredHorizon, 2) — take channel 0 (demand)
            actual_scaled = y_true[node_idx, :, 0].numpy()
            pred_scaled_hybrid = y_pred_hybrid[node_idx].numpy()  # already (24,)
            
            actual_mw = inv_transform(actual_scaled, node_idx)
            pred_mw_hybrid = inv_transform(pred_scaled_hybrid, node_idx)
            
            # Avoid division by zero by clamping actual_mw
            safe_actual = np.clip(np.abs(actual_mw), 1.0, None)
            
            mae = np.mean(np.abs(pred_mw_hybrid - actual_mw))
            mape = np.mean(np.abs((actual_mw - pred_mw_hybrid) / safe_actual)) * 100
            
            results[node_idx]['maes'].append(mae)
            results[node_idx]['mapes'].append(mape)
            results[node_idx]['actual_means'].append(np.mean(actual_mw))

    import csv
    with open('error_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Region', 'Density', 'Industry', 'Mean_Demand_MW', 'Avg_MAE', 'Avg_MAPE'])
        for r in results:
            avg_mae = np.mean(r['maes'])
            avg_mape = np.mean(r['mapes'])
            avg_demand = np.mean(r['actual_means'])
            writer.writerow([r['region'], r['density'], r['industry'], f"{avg_demand:.2f}", f"{avg_mae:.2f}", f"{avg_mape:.2f}"])
            
    print("Analysis saved to error_analysis.csv")

if __name__ == '__main__':
    run()
