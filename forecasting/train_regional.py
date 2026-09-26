import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import SpainElectricityDataset
from gnn_regional import GATForecaster
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# ── IMPROVED LOSS (kept) ──
class AdaptiveGridLoss(nn.Module):
    def __init__(self, under_penalty=3.0, over_penalty=1.0, holiday_boost=5.0,
                 peak_boost=1.3, delta=0.5):
        super().__init__()
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty
        self.holiday_boost = holiday_boost
        self.peak_boost = peak_boost
        self.delta = delta

    def forward(self, y_pred, y_true, y_holiday):
        diff = y_pred - y_true
        abs_diff = torch.abs(diff)
        huber = torch.where(abs_diff <= self.delta, 0.5 * diff**2,
                            self.delta * (abs_diff - 0.5 * self.delta))
        direction_w = torch.where(diff < 0, self.under_penalty, self.over_penalty)
        holiday_w = torch.where(y_holiday > 0.5, self.holiday_boost, 1.0).unsqueeze(-1)
        H = y_pred.shape[-2]
        peak_mask = torch.zeros(H, device=y_pred.device)
        peak_mask[8:22] = 1.0
        peak_w = (1.0 + (self.peak_boost - 1.0) * peak_mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        return torch.mean(huber * direction_w * holiday_w * peak_w)

def calculate_metrics(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    return mse, np.sqrt(mse), mae

def evaluate(model, dl, criterion, ei, ew, device):
    model.eval()
    total, yt, yp = 0, [], []
    with torch.no_grad():
        for x, y, yh, yw in dl:
            x, y, yh = x.to(device), y.to(device), yh.to(device)
            p = model(x, ei, ew)
            total += criterion(p, y, yh).item() * x.size(0)
            yt.append(y); yp.append(p)
    avg = total / len(dl.dataset)
    _, rmse, mae = calculate_metrics(torch.cat(yt), torch.cat(yp))
    return avg, rmse, mae

# ── SIMPLE TRAIN (no gradient accumulation) ──
def train_epoch(model, dl, opt, criterion, ei, ew, device):
    model.train()
    total = 0
    for x, y, yh, yw in dl:
        x, y, yh = x.to(device), y.to(device), yh.to(device)
        opt.zero_grad()
        p = model(x, ei, ew)
        loss = criterion(p, y, yh)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

# ── HOLIDAY SAMPLER (kept) ──
def build_holiday_sampler(ds, factor=5.0):
    w = []
    for idx in range(len(ds)):
        s = idx + ds.seq_length
        e = s + ds.pred_horizon
        has = float(ds.data_matrix[s:e, :, 14].max() > 0.5)  # Feature 14 = is_holiday (shifted after price addition)
        w.append(factor if has else 1.0)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float64),
                                  num_samples=len(w), replacement=True)

def run():
    print("=" * 60, flush=True)
    print("  OPTIMISED GAT Forecaster v2 (GPU-enabled)", flush=True)
    print("=" * 60, flush=True)

    # ── Device setup ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}", flush=True)
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)

    print("\nLoading datasets...", flush=True)
    train_ds = SpainElectricityDataset(split="train", year_start=2015)
    val_ds = SpainElectricityDataset(split="val", year_start=2015)

    sampler = build_holiday_sampler(train_ds, factor=5.0)
    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    # Move edge data to device
    ei = train_ds.edge_index.to(device)
    ew = train_ds.edge_weight.to(device)

    # Model on device
    model = GATForecaster(17, 64, 1, 24, heads=4, dropout=0.2).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    models_dir = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "best_gat_regional.pth")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    warmup_epochs = 5
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    criterion = AdaptiveGridLoss(under_penalty=3.0, over_penalty=1.0,
                                  holiday_boost=5.0, peak_boost=1.3, delta=0.5)

    best_val = float('inf')
    patience = 0
    num_epochs = 100

    print(f"\nLoss: AdaptiveGridLoss(under=3.0, holiday=5x, peak=1.3)", flush=True)
    print(f"Scheduler: LinearWarmup(5ep) -> CosineAnnealingWR(T_0=20)", flush=True)
    print("-" * 80, flush=True)

    for epoch in range(1, num_epochs + 1):
        tl = train_epoch(model, train_dl, opt, criterion, ei, ew, device)
        vl, rmse, mae = evaluate(model, val_dl, criterion, ei, ew, device)
        lr = opt.param_groups[0]['lr']
        sched.step()

        mark = ""
        if vl < best_val:
            best_val = vl
            patience = 0
            mark = " << NEW BEST"
            state = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "in_channels": 17,
                    "hidden_channels": 64,
                    "pred_horizon": 24,
                    "heads": 4,
                    "dropout": 0.2
                }
            torch.save(state, save_path)
        else:
            patience += 1

        print(f"Epoch {epoch:03d} | LR: {lr:.6f} | Train: {tl:.4f} | "
              f"Val: {vl:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}{mark}", flush=True)

        if patience >= 40:
            print(f"\nEarly stopping at epoch {epoch}.", flush=True)
            break

    print(f"\nDone. Best val loss: {best_val:.4f}", flush=True)
    print(f"Saved: {save_path}", flush=True)

if __name__ == "__main__":
    run()

