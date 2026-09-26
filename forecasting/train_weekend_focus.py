"""
train_weekend_focus.py — Weekend-Specialised GAT Retrainer
============================================================
• Loads from the existing best checkpoint (warm start)
• Applies 8x oversampling of weekend samples via WeightedRandomSampler
• Weekend boost = 8.0 in loss, holiday boost = 5.0 (secondary)
• Runs 150 epochs with CosineAnnealingWarmRestarts(T_0=40)
• Saves to best_gat_forecaster_weekend.pth (separate file)
"""

import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import SpainElectricityDataset
from gnn import GATForecaster
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# ── Loss ──
class WeekendFocusLoss(nn.Module):
    def __init__(self, under_penalty=3.0, over_penalty=1.0, holiday_boost=5.0,
                 weekend_boost=8.0, peak_boost=1.5, delta=0.5):
        super().__init__()
        self.under_penalty = under_penalty
        self.over_penalty  = over_penalty
        self.holiday_boost = holiday_boost
        self.weekend_boost = weekend_boost
        self.peak_boost    = peak_boost
        self.delta         = delta

    def forward(self, y_pred, y_true, y_holiday, y_weekend=None):
        diff     = y_pred - y_true
        abs_diff = torch.abs(diff)
        huber    = torch.where(abs_diff <= self.delta,
                               0.5 * diff**2,
                               self.delta * (abs_diff - 0.5 * self.delta))
        direction_w = torch.where(diff < 0, self.under_penalty, self.over_penalty)
        
        # Start with base weight 1.0
        boost_w = torch.ones_like(y_holiday)
        # Weekend boost (primary focus of this model)
        if y_weekend is not None:
            boost_w = torch.where(y_weekend > 0.5, self.weekend_boost, boost_w)
        # Holiday boost (secondary, stacks with weekend if holiday falls on weekend)
        boost_w = torch.where(y_holiday > 0.5, torch.clamp(boost_w * self.holiday_boost / self.weekend_boost, min=self.holiday_boost), boost_w)
        
        H        = y_pred.shape[-1]
        peak_mask = torch.zeros(H, device=y_pred.device)
        peak_mask[8:22] = 1.0
        peak_w   = (1.0 + (self.peak_boost - 1.0) * peak_mask).unsqueeze(0).unsqueeze(0)
        return torch.mean(huber * direction_w * boost_w * peak_w)


def metrics(y_true, y_pred):
    mse  = torch.mean((y_true - y_pred)**2).item()
    mae  = torch.mean(torch.abs(y_true - y_pred)).item()
    return mse, np.sqrt(mse), mae


def evaluate(model, dl, criterion, ei, ew):
    model.eval()
    total, yt, yp, yw_all = 0, [], [], []
    with torch.no_grad():
        for x, y, yh, yw in dl:
            x, y, yh, yw = x.to(ei.device), y.to(ei.device), yh.to(ei.device), yw.to(ei.device)
            p      = model(x, ei, ew)
            total += criterion(p, y, yh, yw).item() * x.size(0)
            yt.append(y); yp.append(p); yw_all.append(yw)
    avg     = total / len(dl.dataset)
    yt_cat  = torch.cat(yt)
    yp_cat  = torch.cat(yp)
    yw_cat  = torch.cat(yw_all)
    _, rmse, mae = metrics(yt_cat, yp_cat)

    # Weekend-specific MAE
    w_mask = (yw_cat > 0.5)
    if w_mask.any():
        w_mae = torch.mean(torch.abs(yt_cat[w_mask] - yp_cat[w_mask])).item()
    else:
        w_mae = float('nan')

    return avg, rmse, mae, w_mae


def train_epoch(model, dl, opt, criterion, ei, ew):
    model.train()
    total = 0
    for x, y, yh, yw in dl:
        x, y, yh, yw = x.to(ei.device), y.to(ei.device), yh.to(ei.device), yw.to(ei.device)
        opt.zero_grad()
        p    = model(x, ei, ew)
        loss = criterion(p, y, yh, yw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)


def build_weekend_sampler(ds, factor=8.0):
    """Oversample weekend days by `factor`x."""
    w = []
    for idx in range(len(ds)):
        s = idx + ds.seq_length
        e = s + ds.pred_horizon
        # Feature 15 = is_weekend
        has_weekend = float(ds.data_matrix[s:e, :, 15].max() > 0.5)
        w.append(factor if has_weekend else 1.0)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float64),
                                 num_samples=len(w), replacement=True)


def run():
    print("=" * 60, flush=True)
    print("  GAT Weekend-Focused Retrainer (150 epochs, 8x weekend)", flush=True)
    print("=" * 60, flush=True)

    train_ds = SpainElectricityDataset(split="train", year_start=2015)
    val_ds   = SpainElectricityDataset(split="val", year_start=2015)

    sampler  = build_weekend_sampler(train_ds, factor=8.0)
    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    ei, ew   = train_ds.edge_index.to(device), train_ds.edge_weight.to(device)
    model    = GATForecaster(16, 64, 1, 24, heads=4, dropout=0.2).to(device)

    models_dir   = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)
    warm_path    = os.path.join(models_dir, "best_gat_forecaster_asymmetric.pth")
    save_path    = os.path.join(models_dir, "best_gat_forecaster_weekend.pth")

    # ── Warm-start from the asymmetric checkpoint ──
    if os.path.exists(warm_path):
        try:
            ck = torch.load(warm_path, map_location="cpu", weights_only=False)
            sd = ck['state_dict'] if 'state_dict' in ck else ck
            # Handle in_channels mismatch (old=15, new=16) by loading partial weights
            model_sd = model.state_dict()
            for key in sd:
                if key in model_sd and sd[key].shape == model_sd[key].shape:
                    model_sd[key] = sd[key]
            model.load_state_dict(model_sd)
            print(f"[+] Warm-started from {warm_path} (with shape-safe loading)", flush=True)
        except Exception as e:
            print(f"[!] Could not warm-start: {e} — training from scratch", flush=True)
    else:
        print("[!] No checkpoint found — training from scratch", flush=True)

    opt      = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-6)
    criterion = WeekendFocusLoss(under_penalty=3.0, over_penalty=1.0,
                                  weekend_boost=8.0, holiday_boost=5.0, peak_boost=1.5, delta=0.5)

    best_val  = float('inf')
    patience  = 0
    num_epochs = 100

    print(f"\nLoss: WeekendFocusLoss(under=3.0, weekend=8x, holiday=5x, peak=1.5)", flush=True)
    print(f"Optimizer: Adam(lr=5e-4) | Sched: CosineAnnealingWR(T_0=40)", flush=True)
    print(f"Weekend Sampler: 8x oversampling", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Ep':>4} {'LR':>8} {'Train':>8} {'Val':>8} {'RMSE':>7} "
          f"{'MAE':>7} {'W-MAE':>7}  Note", flush=True)
    print("-" * 90, flush=True)

    for epoch in range(1, num_epochs + 1):
        tl                       = train_epoch(model, train_dl, opt, criterion, ei, ew)
        vl, rmse, mae, w_mae = evaluate(model, val_dl, criterion, ei, ew)
        lr                       = opt.param_groups[0]['lr']
        sched.step()

        mark = ""
        if vl < best_val:
            best_val  = vl
            patience  = 0
            mark      = " << BEST"
            torch.save({
                'state_dict': model.state_dict(),
                'in_channels': 16, 'hidden_channels': 64,
                'pred_horizon': 24, 'heads': 4, 'dropout': 0.2,
                'best_val_loss': best_val, 'epoch': epoch,
                'loss_config': {'type': 'WeekendFocusLoss',
                                'under_penalty': 3.0, 'over_penalty': 1.0,
                                'weekend_boost': 8.0, 'holiday_boost': 5.0,
                                'peak_boost': 1.5, 'delta': 0.5},
                'num_scaled_features': 6,
            }, save_path)
        else:
            patience += 1

        w_str = f"{w_mae:.4f}" if not np.isnan(w_mae) else "  n/a "
        print(f"{epoch:4d} {lr:.6f} {tl:8.4f} {vl:8.4f} {rmse:7.4f} "
              f"{mae:7.4f} {w_str:>7}{mark}", flush=True)

        if patience >= 100:
            print(f"\nEarly stopping at epoch {epoch}.", flush=True)
            break

    print(f"\nDone. Best val loss: {best_val:.4f}", flush=True)
    print(f"Saved: {save_path}", flush=True)


if __name__ == "__main__":
    run()
