import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, 'forecasting')
from dataset import SpainElectricityDataset
from gnn import GATForecaster
from hybrid_corrector import HybridCorrector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds = SpainElectricityDataset(split='test', year_start=2015, year_end=2026)
ei = ds.edge_index.to(device)
ew = ds.edge_weight.to(device) if ds.edge_weight is not None else None

ck_asym = torch.load('data/models/best_gat_forecaster_asymmetric.pth', map_location='cpu')
model_asym = GATForecaster(15, 64, 1, 24, heads=4, dropout=0.2).to(device)
model_asym.load_state_dict(ck_asym['state_dict'])
model_asym.eval()

ck_hol = torch.load('data/models/best_gat_forecaster_holiday.pth', map_location='cpu')
model_hol = GATForecaster(15, 64, 1, 24, heads=4, dropout=0.2).to(device)
model_hol.load_state_dict(ck_hol['state_dict'])
model_hol.eval()

corrector = HybridCorrector(model_asym, model_hol, ds)
corrector.load_correctors('data/models/hybrid_corrector.pkl')

def inv(scaled_arr, node_idx):
    ns = ds.num_scaled_features
    d = np.zeros((24, ns))
    d[:, 0] = scaled_arr
    return ds.scaler.inverse_transform(d, node_idx=node_idx)[:, 0]

output_dir = r"C:\Users\soham\.gemini\antigravity-ide\brain\5d65c399-2b60-4d9e-9545-1ae6cc434e1b"

def plot_date(test_idx, date_str, label, filename):
    x, y_true, _ = ds[test_idx]
    x_gpu = x.unsqueeze(0).to(device)
    with torch.no_grad():
        y_asym = model_asym(x_gpu, ei, ew).squeeze(0).cpu().numpy()
    y_hybrid = corrector.predict(x_gpu, ei, ew, target_date=date_str)

    # Calculate total load
    tru_total = np.zeros(24)
    ra_total = np.zeros(24)
    hyb_total = np.zeros(24)
    
    for node_i in range(ds.num_nodes):
        tru = inv(y_true[node_i].numpy(), node_i)
        ra  = inv(y_asym[node_i], node_i)
        hyb = inv(y_hybrid[node_i].numpy(), node_i)
        
        tru_total += tru
        ra_total += ra
        hyb_total += hyb
        
    plt.figure(figsize=(10, 6))
    plt.plot(tru_total, label='True Load', color='black', linewidth=2)
    plt.plot(ra_total, label='GAT Asym (Baseline)', color='blue', linestyle='--')
    plt.plot(hyb_total, label='Hybrid Corrector', color='red')
    plt.title(f'Total Load Forecast: {label} ({date_str})')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print('Generating plots...')
plot_date(3246,  '2025-10-09', 'PREV EXPLOSION (Autumn Thu)', 'clamp_eval_2025-10-09.png')
plot_date(1926,  '2025-08-15', 'Holiday: Assumption of Mary', 'clamp_eval_2025-08-15.png')
plot_date(5070,  '2025-12-24', 'Holiday: Christmas Eve', 'clamp_eval_2025-12-24.png')
plot_date(4638,  '2025-12-06', 'Holiday: Constitution Day', 'clamp_eval_2025-12-06.png')
plot_date(366,   '2025-06-11', 'Normal Spring Wednesday', 'clamp_eval_2025-06-11.png')
plot_date(1686,  '2025-08-05', 'Summer Weekday', 'clamp_eval_2025-08-05.png')
plot_date(1806,  '2025-08-10', 'Weekend Sunday', 'clamp_eval_2025-08-10.png')
plot_date(2670,  '2025-09-15', 'Early Autumn', 'clamp_eval_2025-09-15.png')
print('Plots saved.')
