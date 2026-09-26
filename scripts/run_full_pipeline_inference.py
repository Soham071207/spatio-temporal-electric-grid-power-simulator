# -*- coding: utf-8 -*-
"""
Stage 1b: Full Pipeline Inference (GAT × 3 + HybridCorrector)
==============================================================
Runs the complete production forecasting stack:
  - GATForecaster (asymmetric) — base model
  - GATForecaster (holiday-focused)
  - GATForecaster (weekend-focused)
  - HybridCorrector (19 per-node HistGBT models)

This is what the live API uses. Produces the REAL paper numbers.
"""

import os, sys, json, time, warnings, pickle
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "forecasting"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

RESULTS_DIR = os.path.join(BASE, "paper_figures", "real_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_json(obj, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  [saved] {path}")

def hr(t=""):
    print("\n" + "=" * 65)
    if t: print(f"  {t}"); print("=" * 65)


class LegacyGATWrapper(torch.nn.Module):
    """
    Wraps a GATForecaster trained with old arch (out_channels=1).
    Since GATForecaster now returns (B,N,pred_h,1) when out_channels=1,
    this wrapper simply squeezes the last dimension to return (B,N,pred_h)
    so HybridCorrector's indexing works.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, edge_index, edge_weight=None):
        return self.base(x, edge_index, edge_weight).squeeze(-1)


def run():
    hr("STAGE 1b — Full Pipeline Inference (All 4 Models)")

    from forecasting.dataset import SpainElectricityDataset
    from forecasting.gnn_regional import GATForecaster
    from forecasting.hybrid_corrector import HybridCorrector

    DATA = os.path.join(BASE, "data")
    print("\n[1] Loading test dataset …")
    test_ds = SpainElectricityDataset(split="test", year_start=2015, data_dir=DATA)
    N         = test_ds.num_nodes
    node_names = test_ds.node_names
    seq_len    = test_ds.seq_length
    pred_h     = test_ds.pred_horizon
    ei         = test_ds.edge_index
    ew         = test_ds.edge_weight
    print(f"  Samples={len(test_ds)}  Nodes={N}  Horizon={pred_h}")

    # ── Load all 3 GAT models ─────────────────────────────────────────────────
    print("\n[2] Loading GAT model checkpoints …")
    MODELS = {
        "asymmetric": "best_gat_forecaster_asymmetric.pth",
        "holiday":    "best_gat_forecaster_holiday.pth",
        "weekend":    "best_gat_forecaster_weekend.pth",
    }
    gat_models = {}
    for name, fname in MODELS.items():
        path = os.path.join(DATA, "models", fname)
        ck   = torch.load(path, map_location="cpu", weights_only=False)
        in_ch = ck.get("in_channels", 16)
        hid   = ck.get("hidden_channels", 64)
        heads = ck.get("heads", 4)
        drop  = ck.get("dropout", 0.2)
        # Detect pred_horizon from fc2.weight shape
        fc2_w = ck["state_dict"]["fc2.weight"]   # (out_size, hidden)
        out_size = fc2_w.shape[0]                  # pred_h * 2  OR  pred_h
        if out_size % 2 == 0 and out_size // 2 == pred_h:
            actual_ph, out_ch = pred_h, 2
        else:
            actual_ph, out_ch = out_size, 1   # specialist: 24 steps, 1 output
        horiz_stored = ck.get("pred_horizon", actual_ph)
        m = GATForecaster(in_ch, hid, out_ch, actual_ph, heads=heads, dropout=drop)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        if out_ch == 1:
            m = LegacyGATWrapper(m)
        gat_models[name] = m
        print(f"  {name:12s}: in={in_ch} hid={hid} heads={heads} ph={actual_ph} out={out_ch}  ep={ck.get('epoch','?')}  val={ck.get('best_val_loss',0):.4f}")

    # ── Load HybridCorrector GBT models ──────────────────────────────────────
    print("\n[3] Loading HybridCorrector (19 HistGBT models) …")
    hc_path = os.path.join(DATA, "models", "hybrid_corrector.pkl")
    with open(hc_path, "rb") as f:
        hc_data = pickle.load(f)
    gbt_correctors = hc_data["correctors"]   # dict {node_idx: HistGBTR}
    print(f"  Loaded {len(gbt_correctors)} node correctors from {hc_path}")

    # Reconstruct a minimal HybridCorrector for inference
    hc = HybridCorrector(
        model_asym    = gat_models["asymmetric"],
        model_hol     = gat_models["holiday"],
        dataset       = test_ds,
        model_weekend = gat_models["weekend"],
    )
    hc.correctors = gbt_correctors   # inject saved GBT correctors

    # ── Run full inference ────────────────────────────────────────────────────
    print("\n[4] Running full pipeline inference on test set …")
    mean0 = test_ds.scaler.mean[:, 0]
    std0  = test_ds.scaler.std[:,  0]

    all_corrected_mw  = []   # (T, N, 24)
    all_target_mw     = []   # (T, N, 24)
    ti = test_ds.time_idx

    t0 = time.time()
    bs = 1   # HybridCorrector.predict() works sample-by-sample (squeeze(0))
    for s, batch in enumerate(DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)):
        bx, by, yh, yw = batch   # (1,N,24,17), (1,N,24,2), (1,N,24), (1,N,24)

        # Determine target date for adaptive blending
        t_start = s + seq_len
        target_date = ti[t_start] if t_start < len(ti) else None

        # Run HybridCorrector (handles all 3 GATs + GBT correction)
        corrected_scaled = hc.predict(bx, ei, ew, target_date=target_date,
                                       y_holiday=yh, y_weekend=yw)  # (N, 24)

        # Inverse-scale
        corrected_mw = corrected_scaled * std0[:, None] + mean0[:, None]   # (N, 24)
        target_demand_scaled = by[0, :, :, 0].numpy()                      # (N, 24)
        target_mw = target_demand_scaled * std0[:, None] + mean0[:, None]  # (N, 24)

        all_corrected_mw.append(corrected_mw)
        all_target_mw.append(target_mw)

        if (s + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Sample {s+1:5d}/{len(test_ds)} ({elapsed:.0f}s  {(s+1)/elapsed:.0f} samples/sec)", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {len(test_ds)} samples in {elapsed:.1f}s ({len(test_ds)/elapsed:.0f}/sec)")

    pred_mw   = np.stack(all_corrected_mw, axis=0)   # (T, N, 24)
    target_mw = np.stack(all_target_mw,    axis=0)   # (T, N, 24)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[5] Computing per-node and overall metrics …")
    p_all = pred_mw.ravel()
    t_all = target_mw.ravel()

    node_metrics = {}
    for i, name in enumerate(node_names):
        p = pred_mw[:, i, :].ravel()
        t = target_mw[:, i, :].ravel()
        mae  = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        mape = float(np.mean(np.abs((p-t) / (np.abs(t) + 1e-8))) * 100)
        r2   = float(1 - np.sum((p-t)**2) / (np.sum((t-t.mean())**2) + 1e-8))
        node_metrics[name] = {
            "MAE_MW":   round(mae,  2),
            "RMSE_MW":  round(rmse, 2),
            "MAPE_pct": round(mape, 3),
            "R2":       round(r2,   4),
            "mean_demand_mw": round(float(np.mean(t)), 2),
        }
        print(f"  {name:40s}: MAE={mae:7.1f} MW  MAPE={mape:5.2f}%  R²={r2:.4f}")

    overall = {
        "MAE_MW":   round(float(np.mean(np.abs(p_all-t_all))), 2),
        "RMSE_MW":  round(float(np.sqrt(np.mean((p_all-t_all)**2))), 2),
        "MAPE_pct": round(float(np.mean(np.abs((p_all-t_all)/(np.abs(t_all)+1e-8)))*100), 3),
        "R2":       round(float(1 - np.sum((p_all-t_all)**2)/(np.sum((t_all-t_all.mean())**2)+1e-8)), 4),
        "n_test_samples": len(test_ds),
        "n_nodes":        N,
        "inference_sec":  round(elapsed, 1),
    }

    # Horizon breakdown
    horizon_metrics = {}
    for h in range(24):
        ph = pred_mw[:,:,h].ravel(); th = target_mw[:,:,h].ravel()
        horizon_metrics[f"h{h+1:02d}"] = {
            "MAE_MW":   round(float(np.mean(np.abs(ph-th))), 2),
            "MAPE_pct": round(float(np.mean(np.abs((ph-th)/(np.abs(th)+1e-8)))*100), 3),
        }

    # Peak vs normal
    p90 = np.percentile(t_all, 90)
    pk  = t_all >= p90
    peak_mae  = float(np.mean(np.abs((p_all-t_all)[pk])))
    norm_mae  = float(np.mean(np.abs((p_all-t_all)[~pk])))
    peak_mape = float(np.mean(np.abs(((p_all-t_all)/(t_all+1e-8))[pk]))*100)

    # Weekend / holiday
    data_mat = test_ds.data_matrix
    wknd_mask = np.array([ti[s+seq_len].dayofweek >= 5 if (s+seq_len) < len(ti) else False
                           for s in range(len(test_ds))])
    hol_mask  = np.array([bool(data_mat[s+seq_len, 0, 14] > 0.5) if (s+seq_len)<len(ti) and data_mat.shape[2]>14 else False
                           for s in range(len(test_ds))])
    weekday_mask = ~wknd_mask & ~hol_mask

    def subset_m(mask1d):
        m_full = np.broadcast_to(mask1d[:,None,None], pred_mw.shape).ravel()
        pp = p_all[m_full]; tt = t_all[m_full]
        if len(pp) == 0: return {"n": 0}
        return {"MAE_MW": round(float(np.mean(np.abs(pp-tt))),2),
                "MAPE_pct": round(float(np.mean(np.abs((pp-tt)/(np.abs(tt)+1e-8)))*100),3),
                "n": int(m_full.sum())}

    regime = {
        "peak_demand_top10pct": {"MAE_MW": round(peak_mae,2), "MAPE_pct": round(peak_mape,3)},
        "normal_hours":         {"MAE_MW": round(norm_mae,2)},
        "weekday":  subset_m(weekday_mask),
        "weekend":  subset_m(wknd_mask),
        "holiday":  subset_m(hol_mask),
    }

    hr()
    print("  FULL PIPELINE RESULTS (GAT × 3 + HybridCorrector):")
    for k, v in overall.items():
        print(f"    {k}: {v}")
    print(f"\n  Peak MAE: {peak_mae:.1f} MW | Normal MAE: {norm_mae:.1f} MW")
    print(f"  Weekend MAPE: {regime['weekend'].get('MAPE_pct','N/A')}% | Holiday MAPE: {regime['holiday'].get('MAPE_pct','N/A')}%")

    results = {
        "model":      "Full Pipeline (GATx3 + HybridCorrector)",
        "pipeline":   {"asymmetric": MODELS["asymmetric"], "holiday": MODELS["holiday"],
                       "weekend": MODELS["weekend"], "corrector": "hybrid_corrector.pkl"},
        "overall":    overall,
        "per_node":   node_metrics,
        "horizon":    horizon_metrics,
        "regime":     regime,
    }
    save_json(results, "stage1b_full_pipeline_results.json")

    # Save arrays — these become the definitive paper numbers
    np.save(os.path.join(RESULTS_DIR, "pred_mw.npy"),   pred_mw)
    np.save(os.path.join(RESULTS_DIR, "target_mw.npy"), target_mw)
    print("\n  Arrays saved. These are now the reference for all downstream figures.")
    return results


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\n  Wall time: {(time.time()-t0)/60:.1f} min")
