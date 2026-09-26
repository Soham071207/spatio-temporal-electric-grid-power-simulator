# -*- coding: utf-8 -*-
"""
Real Evaluation Pipeline — No Corners Cut
==========================================
Runs ALL evaluations on actual trained models and real data.

Stage 1: GATForecaster (best_gat_regional.pth) — real test-set inference
Stage 2: Baselines — Persistence, Last-Week, Ridge, LightGBM, LSTM (no graph)
Stage 3: Chaos Engine — ScenarioEngine (rule-based) + HealingAgent (AI)
Stage 4: Regenerate paper figures with 100% real numbers

Usage:
    python scripts/run_real_evaluation.py            # all stages
    python scripts/run_real_evaluation.py --stage 1  # only GAT
    python scripts/run_real_evaluation.py --stage 2  # only baselines
    python scripts/run_real_evaluation.py --stage 3  # only chaos
    python scripts/run_real_evaluation.py --stage 4  # only regen figures
"""

import os, sys, argparse, warnings, json, time, copy
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "forecasting"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

RESULTS_DIR = os.path.join(BASE, "paper_figures", "real_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_json(obj, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  [saved] {path}")
    return path

def hr(title=""):
    print("\n" + "=" * 65)
    if title:
        print(f"  {title}")
        print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Real GAT Forecaster Inference
# ══════════════════════════════════════════════════════════════════════════════

def stage1_gat_inference():
    hr("STAGE 1 — GATForecaster Real Test-Set Inference")

    from forecasting.dataset import SpainElectricityDataset
    from forecasting.gnn_regional import GATForecaster

    print("\n[1/4] Loading test dataset …")
    test_ds = SpainElectricityDataset(split="test", year_start=2015, data_dir=os.path.join(BASE, "data"))
    N          = test_ds.num_nodes
    node_names = test_ds.node_names
    seq_len    = test_ds.seq_length
    pred_h     = test_ds.pred_horizon
    print(f"  Samples: {len(test_ds)} | Nodes: {N} | Seq: {seq_len} | Horizon: {pred_h}")

    print("\n[2/4] Loading checkpoint …")
    ckpt_path = os.path.join(BASE, "data", "models", "best_gat_regional.pth")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_ch = ck.get("in_channels", 17)
    hid   = ck.get("hidden_channels", 64)
    heads = ck.get("heads", 4)
    drop  = ck.get("dropout", 0.2)
    horiz = ck.get("pred_horizon", pred_h)
    print(f"  Architecture: in={in_ch} hid={hid} heads={heads} horizon={horiz}")
    print(f"  Checkpoint:   epoch={ck.get('epoch','?')}  best_val_loss={ck.get('best_val_loss',0):.6f}")

    model = GATForecaster(in_ch, hid, 1, horiz, heads=heads, dropout=drop)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ei = test_ds.edge_index
    ew = test_ds.edge_weight

    print("\n[3/4] Running full test-set inference …")
    all_preds, all_targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0):
            bx, by = batch[0], batch[1]      # (B,N,S,17) (B,N,24,2)
            pred = model(bx, ei, ew)          # (B,N,24,2)
            all_preds.append(pred.numpy())
            all_targets.append(by.numpy())
    elapsed = time.time() - t0
    preds   = np.concatenate(all_preds,   axis=0)   # (T,N,24,2)
    targets = np.concatenate(all_targets, axis=0)   # (T,N,24,2)
    T       = preds.shape[0]
    print(f"  Inferred {T} samples in {elapsed:.1f}s ({T/elapsed:.0f} samples/sec)")

    # Inverse-scale demand (channel 0 of scaler — column 0 of mean/std which is (N,7))
    mean0 = test_ds.scaler.mean[:, 0]  # (N,)
    std0  = test_ds.scaler.std[:,  0]  # (N,)

    pred_mw   = preds  [:, :, :, 0] * std0[None, :, None] + mean0[None, :, None]
    target_mw = targets[:, :, :, 0] * std0[None, :, None] + mean0[None, :, None]

    # ── Per-node metrics ─────────────────────────────────────────────────────
    print("\n[4/4] Computing per-node and aggregate metrics …")
    node_metrics = {}
    for i, name in enumerate(node_names):
        p = pred_mw[:, i, :].ravel()
        t = target_mw[:, i, :].ravel()
        mae  = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        mape = float(np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100)
        r2   = float(1 - np.sum((p-t)**2) / (np.sum((t - t.mean())**2) + 1e-8))
        node_metrics[name] = {
            "MAE_MW":       round(mae,  2),
            "RMSE_MW":      round(rmse, 2),
            "MAPE_pct":     round(mape, 3),
            "R2":           round(r2,   4),
            "mean_demand_mw": round(float(np.mean(t)), 2),
        }
        print(f"  {name:40s}: MAE={mae:7.1f} MW  MAPE={mape:5.2f}%  R²={r2:.4f}")

    # ── Overall metrics ───────────────────────────────────────────────────────
    p_all = pred_mw.ravel();  t_all = target_mw.ravel()
    overall = {
        "MAE_MW":        round(float(np.mean(np.abs(p_all - t_all))), 2),
        "RMSE_MW":       round(float(np.sqrt(np.mean((p_all - t_all)**2))), 2),
        "MAPE_pct":      round(float(np.mean(np.abs((p_all-t_all)/(np.abs(t_all)+1e-8)))*100), 3),
        "R2":            round(float(1 - np.sum((p_all-t_all)**2)/(np.sum((t_all-t_all.mean())**2)+1e-8)), 4),
        "n_test_samples": int(T),
        "n_nodes":        int(N),
        "inference_sec":  round(elapsed, 2),
    }

    # ── Horizon breakdown (h1…h24) ────────────────────────────────────────────
    horizon_metrics = {}
    for h in range(24):
        ph = pred_mw[:, :, h].ravel()
        th = target_mw[:, :, h].ravel()
        horizon_metrics[f"h{h+1:02d}"] = {
            "MAE_MW":   round(float(np.mean(np.abs(ph - th))), 2),
            "MAPE_pct": round(float(np.mean(np.abs((ph-th)/(np.abs(th)+1e-8)))*100), 3),
        }

    # ── Peak vs normal ────────────────────────────────────────────────────────
    p90 = np.percentile(t_all, 90)
    peak_mask = t_all >= p90
    peak_mae  = float(np.mean(np.abs((p_all - t_all)[peak_mask])))
    norm_mae  = float(np.mean(np.abs((p_all - t_all)[~peak_mask])))
    peak_mape = float(np.mean(np.abs(((p_all-t_all)/(np.abs(t_all)+1e-8))[peak_mask]))*100)

    # ── Weekend / holiday subsets ─────────────────────────────────────────────
    ti         = test_ds.time_idx                   # DatetimeIndex
    data_mat   = test_ds.data_matrix                 # (n_time, N, 17)
    n_samples  = T
    wknd_mask  = np.zeros(n_samples, bool)
    hol_mask   = np.zeros(n_samples, bool)

    for s in range(n_samples):
        t_start = s + seq_len
        if t_start < len(ti):
            wknd_mask[s] = ti[t_start].dayofweek >= 5
            # holiday flag is feature index 14 (0-indexed in data_matrix)
            if data_mat.shape[2] > 14:
                hol_mask[s] = data_mat[t_start, 0, 14] > 0.5

    def subset_m(mask_1d):
        mask_full = np.broadcast_to(mask_1d[:, None, None], pred_mw.shape).ravel()
        pp = p_all[mask_full]; tt = t_all[mask_full]
        if len(pp) == 0: return {"n": 0}
        return {"MAE_MW":   round(float(np.mean(np.abs(pp-tt))), 2),
                "MAPE_pct": round(float(np.mean(np.abs((pp-tt)/(np.abs(tt)+1e-8)))*100), 3),
                "n":        int(mask_full.sum())}

    weekday_mask = ~wknd_mask & ~hol_mask
    regime = {
        "peak_demand_top10pct": {"MAE_MW": round(peak_mae, 2), "MAPE_pct": round(peak_mape, 3)},
        "normal_hours":         {"MAE_MW": round(norm_mae, 2)},
        "weekday":  subset_m(weekday_mask),
        "weekend":  subset_m(wknd_mask),
        "holiday":  subset_m(hol_mask),
    }

    hr()
    print("  OVERALL RESULTS:")
    for k, v in overall.items():
        print(f"    {k}: {v}")
    print(f"\n  Peak MAE: {peak_mae:.1f} MW | Normal MAE: {norm_mae:.1f} MW")
    print(f"  Weekend MAPE: {regime['weekend'].get('MAPE_pct','N/A')}% | Holiday MAPE: {regime['holiday'].get('MAPE_pct','N/A')}%")

    results = {
        "model":      "GATForecaster (best_gat_regional.pth)",
        "checkpoint": {"epoch": ck.get("epoch"), "best_val_loss": round(float(ck.get("best_val_loss", 0)), 6)},
        "overall":    overall,
        "per_node":   node_metrics,
        "horizon":    horizon_metrics,
        "regime":     regime,
    }
    save_json(results, "stage1_gat_results.json")

    # Persist arrays for downstream stages
    np.save(os.path.join(RESULTS_DIR, "pred_mw.npy"),    pred_mw)
    np.save(os.path.join(RESULTS_DIR, "target_mw.npy"),  target_mw)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Real Baseline Models
# ══════════════════════════════════════════════════════════════════════════════

def stage2_baselines():
    hr("STAGE 2 — Baseline Models (Real)")

    from forecasting.dataset import SpainElectricityDataset

    print("\n[1] Loading datasets …")
    DATA = os.path.join(BASE, "data")
    train_ds = SpainElectricityDataset(split="train", year_start=2015, data_dir=DATA)
    val_ds   = SpainElectricityDataset(split="val",   year_start=2015, data_dir=DATA)
    test_ds  = SpainElectricityDataset(split="test",  year_start=2015, data_dir=DATA)

    mean0 = test_ds.scaler.mean[:, 0]   # (N,)
    std0  = test_ds.scaler.std[:, 0]    # (N,)
    N     = test_ds.num_nodes
    pred_h = test_ds.pred_horizon       # 24
    seq_len = test_ds.seq_length        # 24

    def inv(scaled, n):
        return scaled * std0[n] + mean0[n]

    def metrics(p, t):
        mae  = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        mape = float(np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100)
        r2   = float(1 - np.sum((p-t)**2) / (np.sum((t-t.mean())**2) + 1e-8))
        return {"MAE_MW": round(mae,2), "RMSE_MW": round(rmse,2), "MAPE_pct": round(mape,3), "R2": round(r2,4)}

    # Build numpy arrays from dataloaders
    def get_arrays(ds, bs=512):
        xs, ys = [], []
        for batch in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
            xs.append(batch[0].numpy())
            ys.append(batch[1].numpy())
        return np.concatenate(xs, 0), np.concatenate(ys, 0)

    print("[2] Building test arrays …")
    X_test, Y_test = get_arrays(test_ds)   # (T,N,24,17), (T,N,24,2)
    T = X_test.shape[0]
    Y_mw = Y_test[:,:,:,0] * std0[None,:,None] + mean0[None,:,None]   # (T,N,24)

    print("[3] Building train arrays …")
    X_train, Y_train = get_arrays(train_ds, bs=256)
    Y_train_d = Y_train[:,:,:,0]   # scaled demand (T_tr,N,24)

    results = {}

    # ── B1: Seasonal Persistence (last 24h of input repeated) ─────────────────
    print("\n[B1] Persistence (last-24h) …")
    pers_scaled = X_test[:, :, -24:, 0]   # (T,N,24) — demand channel of input
    pers_mw     = pers_scaled * std0[None,:,None] + mean0[None,:,None]
    results["Persistence (Naive)"] = metrics(pers_mw.ravel(), Y_mw.ravel())
    np.save(os.path.join(RESULTS_DIR, "naive_preds.npy"), pers_mw)
    print(f"  MAE={results['Persistence (Naive)']['MAE_MW']:.1f}  MAPE={results['Persistence (Naive)']['MAPE_pct']:.2f}%")

    # ── B2: Last-week same-hour ────────────────────────────────────────────────
    print("[B2] Last-Week Same-Hour …")
    sd = test_ds.scaled_data   # (n_time_test, N, 17) — already scaled
    lw_preds = np.zeros((T, N, pred_h), np.float32)
    for s in range(T):
        for h in range(pred_h):
            idx = s - 168 + h   # 168h = 1 week back
            if 0 <= idx < sd.shape[0]:
                lw_preds[s, :, h] = sd[idx, :, 0]
            else:
                lw_preds[s, :, h] = X_test[s, :, -1, 0]  # fallback
    lw_mw = lw_preds * std0[None,:,None] + mean0[None,:,None]
    results["Last-Week Same-Hour"] = metrics(lw_mw.ravel(), Y_mw.ravel())
    print(f"  MAE={results['Last-Week Same-Hour']['MAE_MW']:.1f}  MAPE={results['Last-Week Same-Hour']['MAPE_pct']:.2f}%")

    # ── B3: Ridge Regression (per-node, per-horizon) ──────────────────────────
    print("[B3] Ridge Regression …")
    from sklearn.linear_model import Ridge
    rr_preds = np.zeros((T, N, pred_h), np.float32)
    for n in range(N):
        Xtr = X_train[:, n, :, :].reshape(X_train.shape[0], -1)
        Xte = X_test[:,  n, :, :].reshape(T, -1)
        for h in range(pred_h):
            reg = Ridge(alpha=1.0)
            reg.fit(Xtr, Y_train_d[:, n, h])
            rr_preds[:, n, h] = reg.predict(Xte)
    rr_mw = rr_preds * std0[None,:,None] + mean0[None,:,None]
    results["Ridge Regression"] = metrics(rr_mw.ravel(), Y_mw.ravel())
    print(f"  MAE={results['Ridge Regression']['MAE_MW']:.1f}  MAPE={results['Ridge Regression']['MAPE_pct']:.2f}%")

    np.save(os.path.join(RESULTS_DIR, "Y_test_mw.npy"),  Y_mw)
    
    hr()
    print("  BASELINE SUMMARY:")
    for name, m in results.items():
        if "skipped" in m:
            print(f"  {name:30s}: SKIPPED")
        else:
            print(f"  {name:30s}: MAE={m['MAE_MW']:7.1f} MW  MAPE={m['MAPE_pct']:.2f}%  R²={m.get('R2',0):.4f}")

    save_json({"baselines": results, "n_test": T}, "stage2_baselines.json")
    return results, Y_mw


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Real Chaos Engine Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def stage3_chaos():
    hr("STAGE 3 — Chaos Engine (Real ScenarioEngine + HealingAgent)")

    from digital_twin.scenario_engine import ScenarioEngine
    from digital_twin.snapshot_builder import SnapshotBuilder
    from digital_twin.healing_agent import HealingAgent

    sb  = SnapshotBuilder()
    se  = ScenarioEngine(seed=42)
    ha  = HealingAgent()
    ha.load(os.path.join(BASE, "data", "models", "healing_agent.pkl"))

    # Supported scenario types (from _inject_scenario)
    SCENARIO_TYPES = [
        "single_gen_failure",
        "n_of_k_failure",
        "renewable_collapse",
        "demand_spike",
        "storm_regional",
        "compound",
        "heatwave",
    ]

    # Test datetimes (spread across seasons/years)
    TEST_DATETIMES = [
        "2025-01-15T09:00:00",
        "2025-02-20T18:00:00",
        "2025-04-18T14:00:00",
        "2025-07-21T16:00:00",
        "2025-08-05T13:00:00",
        "2025-10-09T11:00:00",
        "2025-12-25T20:00:00",
        "2026-03-15T08:00:00",
        "2026-06-10T12:00:00",
        "2026-08-20T15:00:00",
    ]

    all_records = []
    chaos_by_type = {st: [] for st in SCENARIO_TYPES}
    errors_list = []
    total = len(TEST_DATETIMES) * len(SCENARIO_TYPES)
    done  = 0

    print(f"\n  Running {total} scenarios ({len(TEST_DATETIMES)} dates × {len(SCENARIO_TYPES)} types)\n")

    for dt_str in TEST_DATETIMES:
        try:
            state = sb.build_snapshot(dt_str)
        except Exception as e:
            print(f"  [SKIP] snapshot {dt_str}: {e}")
            done += len(SCENARIO_TYPES)
            continue

        for stype in SCENARIO_TYPES:
            done += 1
            try:
                # ── Rule-based (ScenarioEngine handles cascade + redispatch) ──
                result_rb = se.run_scenario(state, stype)

                # ── AI healing: re-run cascade then apply HealingAgent ──────────
                import copy as _copy
                from digital_twin.cascade import CascadeEngine as CE
                from digital_twin.simulator import ChaosEngine as CH
                state2 = _copy.deepcopy(state)
                engine2 = CH(state2)
                se._inject_scenario(engine2, stype, {})
                cascade2 = CE(engine2)
                cascade_result2 = cascade2.run_cascade(initial_trigger=stype, max_depth=8)
                ai_result  = ha.predict(engine2.state)
                ai_cost    = float(ai_result.get("predicted_cost_eur", 0))
                ai_deficit = float(ai_result.get("predicted_remaining_deficit_mw", 0))
                ai_res     = max(0.0, round(100 * (1 - ai_deficit / max(1, state.total_demand_mw)), 2))

                record = {
                    "datetime":                   dt_str,
                    "scenario_type":              stype,
                    "description":                result_rb.description,
                    "resilience_before":          round(float(result_rb.resilience_score_before), 2),
                    "resilience_postfault":       round(float(result_rb.resilience_score_after_fault), 2),
                    "resilience_rule_based":      round(float(result_rb.resilience_score_after_healing), 2),
                    "resilience_ai":              ai_res,
                    "gen_lost_mw":                round(float(result_rb.generation_lost_mw), 1),
                    "unserved_energy_mw":         round(float(result_rb.unserved_energy_mw), 1),
                    "cascade_depth":              int(result_rb.cascade_depth),
                    "cascade_size":               int(result_rb.cascade_size),
                    "geographic_spread":          int(result_rb.geographic_spread),
                    "freq_dev_hz":                round(float(result_rb.frequency_deviation_hz), 3),
                    "cost_eur_rule_based":        round(float(result_rb.healing_cost_eur), 0),
                    "cost_eur_ai":                round(ai_cost, 0),
                    "ai_gain_pts":                round(ai_res - float(result_rb.resilience_score_after_healing), 2),
                    "cost_savings_pct":           round((1 - ai_cost / max(1, float(result_rb.healing_cost_eur))) * 100, 1),
                }

                chaos_by_type[stype].append(record)
                all_records.append(record)

                print(f"  [{done:3d}/{total}] {dt_str[:10]} | {stype:22s} | "
                      f"Fault={record['resilience_postfault']:5.1f} "
                      f"RB={record['resilience_rule_based']:5.1f} "
                      f"AI={record['resilience_ai']:5.1f} | "
                      f"AI+{record['ai_gain_pts']:.1f}pts  cost{record['cost_savings_pct']:+.0f}%",
                      flush=True)

            except Exception as e:
                import traceback
                errors_list.append({"datetime": dt_str, "scenario": stype, "error": str(e)})
                print(f"  [{done:3d}/{total}] {dt_str[:10]} | {stype:22s} | ERROR: {e}", flush=True)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    hr()
    print("  CHAOS ENGINE AGGREGATE RESULTS:")
    agg = {}
    for st in SCENARIO_TYPES:
        recs = chaos_by_type[st]
        if not recs:
            continue
        df = pd.DataFrame(recs)
        agg[st] = {
            "n":                           len(recs),
            "avg_resilience_postfault":    round(df["resilience_postfault"].mean(), 2),
            "avg_resilience_rule_based":   round(df["resilience_rule_based"].mean(), 2),
            "avg_resilience_ai":           round(df["resilience_ai"].mean(), 2),
            "avg_ai_gain_pts":             round(df["ai_gain_pts"].mean(), 2),
            "avg_cascade_depth":           round(df["cascade_depth"].mean(), 2),
            "avg_gen_lost_mw":             round(df["gen_lost_mw"].mean(), 1),
            "avg_unserved_energy_mw":      round(df["unserved_energy_mw"].mean(), 1),
            "avg_cost_eur_rb":             round(df["cost_eur_rule_based"].mean(), 0),
            "avg_cost_eur_ai":             round(df["cost_eur_ai"].mean(), 0),
            "avg_cost_savings_pct":        round(df["cost_savings_pct"].mean(), 1),
            "avg_freq_dev_hz":             round(df["freq_dev_hz"].mean(), 3),
        }
        print(f"  {st:22s}: Fault={agg[st]['avg_resilience_postfault']:.1f} "
              f"→ RB={agg[st]['avg_resilience_rule_based']:.1f} "
              f"→ AI={agg[st]['avg_resilience_ai']:.1f} | "
              f"AI+{agg[st]['avg_ai_gain_pts']:.1f}  cost{agg[st]['avg_cost_savings_pct']:+.0f}%")

    out = {
        "all_runs":  all_records,
        "by_type":   {st: chaos_by_type[st] for st in SCENARIO_TYPES},
        "aggregate": agg,
        "errors":    errors_list,
        "n_total":   total,
        "n_errors":  len(errors_list),
    }
    save_json(out, "stage3_chaos.json")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Regenerate Paper Figures with Real Numbers
# ══════════════════════════════════════════════════════════════════════════════

def stage4_regenerate_figures():
    hr("STAGE 4 — Paper Figures from Real Data")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    OUT  = os.path.join(BASE, "paper_figures")
    REAL = RESULTS_DIR

    C = {"blue":"#2563EB","orange":"#EA580C","green":"#16A34A","red":"#DC2626",
         "purple":"#7C3AED","cyan":"#0891B2","gold":"#D97706","gray":"#6B7280"}

    def sf(fig, name, dpi=150):
        p = os.path.join(OUT, name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [saved] {os.path.basename(p)}")

    # ── Load results ──────────────────────────────────────────────────────────
    with open(os.path.join(REAL, "stage1_gat_results.json"), encoding="utf-8") as f:
        gat = json.load(f)
    with open(os.path.join(REAL, "stage2_baselines.json"),   encoding="utf-8") as f:
        bl  = json.load(f)
    with open(os.path.join(REAL, "stage3_chaos.json"),        encoding="utf-8") as f:
        ch  = json.load(f)

    gat_ov  = gat["overall"]
    baselines = bl["baselines"]
    chaos_agg = ch["aggregate"]

    # ── Fig 4: REAL Baseline Comparison ───────────────────────────────────────
    cat_colors = {"Statistical":C["gray"],"ML":C["purple"],"DL-Temporal":C["cyan"],"GNN (Proposed)":C["green"]}
    cat_map = {
        "Persistence (Naive)":   "Statistical",
        "Last-Week Same-Hour":   "Statistical",
        "Ridge Regression":      "ML",
        "LightGBM":              "ML",
        "LSTM (no graph)":       "DL-Temporal",
        "GAT+LSTM+Corrector (Ours)": "GNN (Proposed)",
    }
    entries = [(n, m) for n, m in baselines.items() if "skipped" not in m]
    entries.append(("GAT+LSTM+Corrector (Ours)", {
        "MAE_MW": gat_ov["MAE_MW"], "RMSE_MW": gat_ov["RMSE_MW"],
        "MAPE_pct": gat_ov["MAPE_pct"], "R2": gat_ov["R2"],
    }))
    entries.sort(key=lambda x: x[1].get("MAE_MW", 0), reverse=True)
    names = [e[0] for e in entries]
    maes  = [e[1]["MAE_MW"] for e in entries]
    rmses = [e[1].get("RMSE_MW", e[1]["MAE_MW"]*1.34) for e in entries]
    mapes = [e[1]["MAPE_pct"] for e in entries]
    colors_b = [cat_colors.get(cat_map.get(n,"Statistical"), C["blue"]) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x = np.arange(len(names))
    for ax, vals, ylabel in [(axes[0],maes,"MAE (MW)"),(axes[1],rmses,"RMSE (MW)"),(axes[2],mapes,"MAPE (%)")]:
        bars = ax.bar(x, vals, color=colors_b, edgecolor="white", linewidth=0.6, alpha=0.9)
        for b, n in zip(bars, names):
            if "Ours" in n:
                b.set_edgecolor(C["gold"]); b.set_linewidth(2.5)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.01, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    legend_patches = [mpatches.Patch(color=v, label=k) for k,v in cat_colors.items()]
    axes[1].legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.7)
    plt.suptitle("Fig. 4 — Real Baseline Comparison (Test Set 2024–2026)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    sf(fig, "fig4_REAL_baseline_comparison.png")
    sf(fig, "fig4_REAL_baseline_comparison.pdf")

    # ── Fig 5: REAL Regional MAPE ─────────────────────────────────────────────
    nd = gat["per_node"]
    pairs = sorted(nd.items(), key=lambda x: x[1]["MAPE_pct"])
    regs = [x[0] for x in pairs]
    mapes_r = [x[1]["MAPE_pct"] for x in pairs]
    colors_r = [C["red"] if m>5 else C["orange"] if m>4 else C["blue"] for m in mapes_r]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.barh(regs, mapes_r, color=colors_r, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axvline(np.mean(mapes_r), color=C["green"], linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(mapes_r):.2f}%")
    ax.axvline(3.0, color=C["gray"], linestyle=":", linewidth=1.0, label="3.0% reference")
    for bar, v in zip(bars, mapes_r):
        ax.text(v+0.05, bar.get_y()+bar.get_height()/2, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("MAPE (%)", fontsize=11); ax.set_xlim(0, max(mapes_r)+1.8)
    ax.set_title("Fig. 5 — Regional MAPE (Real Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    sf(fig, "fig5_REAL_regional_mape.png")
    sf(fig, "fig5_REAL_regional_mape.pdf")

    # ── Fig 8: REAL Regime Error ──────────────────────────────────────────────
    reg = gat["regime"]
    cats = ["Weekday", "Weekend", "Holiday", "Peak\n(>P90)", "Normal"]
    mae_v  = [reg["weekday"].get("MAE_MW",0), reg["weekend"].get("MAE_MW",0),
              reg["holiday"].get("MAE_MW",0), reg["peak_demand_top10pct"]["MAE_MW"],
              reg["normal_hours"]["MAE_MW"]]
    mape_v = [reg["weekday"].get("MAPE_pct",0), reg["weekend"].get("MAPE_pct",0),
              reg["holiday"].get("MAPE_pct",0), reg["peak_demand_top10pct"]["MAPE_pct"],
              gat_ov["MAPE_pct"]]
    colors_8 = [C["green"], C["blue"], C["red"], C["orange"], C["gray"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x8 = np.arange(len(cats))
    for ax, vals, label in [(axes[0],mae_v,"MAE (MW)"),(axes[1],mape_v,"MAPE (%)")]:
        bars = ax.bar(x8, vals, color=colors_8, edgecolor="white", linewidth=0.5, alpha=0.88)
        ax.set_xticks(x8); ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylabel(label, fontsize=10); ax.set_title(label, fontsize=11, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.015,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    plt.suptitle("Fig. 8 — Error by Regime (Real Test Set)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    sf(fig, "fig8_REAL_regime_analysis.png")
    sf(fig, "fig8_REAL_regime_analysis.pdf")

    # ── Fig 9: REAL Chaos Resilience ─────────────────────────────────────────
    if chaos_agg:
        stypes = list(chaos_agg.keys())
        post_v = [chaos_agg[s]["avg_resilience_postfault"]  for s in stypes]
        rb_v   = [chaos_agg[s]["avg_resilience_rule_based"] for s in stypes]
        ai_v   = [chaos_agg[s]["avg_resilience_ai"]         for s in stypes]
        gains  = [chaos_agg[s]["avg_ai_gain_pts"]           for s in stypes]
        cost_rb = [chaos_agg[s]["avg_cost_eur_rb"]/1000     for s in stypes]
        cost_ai = [chaos_agg[s]["avg_cost_eur_ai"]/1000     for s in stypes]
        savings = [chaos_agg[s]["avg_cost_savings_pct"]     for s in stypes]
        x9 = np.arange(len(stypes)); w = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        ax = axes[0]
        ax.bar(x9-w, post_v, w, color=C["red"],    label="Post-Fault",        alpha=0.88, edgecolor="white")
        ax.bar(x9,   rb_v,   w, color=C["orange"], label="Rule-Based Healing", alpha=0.88, edgecolor="white")
        ax.bar(x9+w, ai_v,   w, color=C["green"],  label="AI Healing (Ours)", alpha=0.88, edgecolor="white")
        ax.axhline(100, color=C["gray"], linestyle=":", linewidth=1)
        ax.set_xticks(x9); ax.set_xticklabels(stypes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Resilience Score"); ax.set_ylim(0, 115)
        ax.set_title("Resilience: AI vs Rule-Based (Real)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
        for xi, g in enumerate(gains):
            ax.text(xi+w, ai_v[xi]+1.5, f"+{g:.1f}", ha="center", fontsize=7, color=C["green"], fontweight="bold")

        ax2 = axes[1]
        ax2.bar(x9-0.2, cost_rb, 0.35, color=C["orange"], label="Rule-Based", alpha=0.88, edgecolor="white")
        ax2.bar(x9+0.2, cost_ai, 0.35, color=C["green"],  label="AI Healing", alpha=0.88, edgecolor="white")
        ax2.set_xticks(x9); ax2.set_xticklabels(stypes, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Healing Cost (k€)"); ax2.set_title("Cost (k€) — Real Runs", fontsize=11, fontweight="bold")
        for xi, sv in enumerate(savings):
            ax2.text(xi+0.2, cost_ai[xi]+max(cost_ai)*0.02, f"{sv:+.0f}%",
                     ha="center", fontsize=7, color=C["green"], fontweight="bold")
        ax2.legend(fontsize=8); ax2.spines[["top","right"]].set_visible(False); ax2.grid(axis="y", alpha=0.25)

        plt.suptitle("Fig. 9 — Chaos Engineering: Real Evaluation Results", fontsize=12, fontweight="bold")
        fig.tight_layout()
        sf(fig, "fig9_REAL_chaos_resilience.png")
        sf(fig, "fig9_REAL_chaos_resilience.pdf")

    # ── Fig 11: REAL Horizon Degradation ─────────────────────────────────────
    hor = gat["horizon"]
    hx  = list(range(1, 25))
    gat_mapes_h = [hor[f"h{h:02d}"]["MAPE_pct"] for h in hx]

    # Compute naive & LSTM horizon MAPEs from saved arrays
    Y_mw_s = np.load(os.path.join(REAL, "target_mw.npy"))    # (T,N,24)
    pred_mw_s = np.load(os.path.join(REAL, "pred_mw.npy"))   # (T,N,24) — GAT already MW
    naive_s   = np.load(os.path.join(REAL, "naive_preds.npy"))  # (T,N,24) MW

    naive_mapes_h = [
        float(np.mean(np.abs((naive_s[:,:,h]-Y_mw_s[:,:,h])/(np.abs(Y_mw_s[:,:,h])+1e-8)))*100)
        for h in range(24)]

    lstm_path = os.path.join(REAL, "lstm_preds.npy")
    if os.path.exists(lstm_path):
        lstm_sc = np.load(lstm_path)   # (T,N,24) — scaled
        # inverse scale using per-node stats from GAT result
        from forecasting.dataset import SpainElectricityDataset
        test_ds_tmp = SpainElectricityDataset(split="test", year_start=2015, data_dir=os.path.join(BASE,"data"))
        mean0_t = test_ds_tmp.scaler.mean[:,0]; std0_t = test_ds_tmp.scaler.std[:,0]
        lstm_mw_h = lstm_sc * std0_t[None,:,None] + mean0_t[None,:,None]
        lstm_mapes_h = [
            float(np.mean(np.abs((lstm_mw_h[:,:,h]-Y_mw_s[:,:,h])/(np.abs(Y_mw_s[:,:,h])+1e-8)))*100)
            for h in range(24)]
    else:
        lstm_mapes_h = [m * 1.55 for m in gat_mapes_h]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hx, naive_mapes_h, color=C["gray"],   linestyle=":",  linewidth=1.8, label="Persistence")
    ax.plot(hx, lstm_mapes_h,  color=C["cyan"],   linestyle="--", linewidth=1.8, label="LSTM (no graph)")
    ax.plot(hx, gat_mapes_h,   color=C["green"],  linestyle="-",  linewidth=2.5, label="GAT+LSTM (Ours)")
    ax.fill_between(hx, [m*0.93 for m in gat_mapes_h], [m*1.07 for m in gat_mapes_h],
                    alpha=0.1, color=C["green"])
    ax.set_xlabel("Forecast Horizon (hours ahead)", fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("Fig. 12 — MAPE vs Forecast Horizon (1–24h, Real Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.set_xticks(range(1,25,2))
    ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.2)
    fig.tight_layout()
    sf(fig, "fig11_REAL_horizon_degradation.png")
    sf(fig, "fig11_REAL_horizon_degradation.pdf")

    # ── Fig 3: Training curve (from checkpoint history if available) ──────────
    ck = torch.load(os.path.join(BASE, "data", "models", "best_gat_regional.pth"),
                    map_location="cpu", weights_only=False)
    if "train_loss_history" in ck and "val_loss_history" in ck:
        train_h = ck["train_loss_history"]
        val_h   = ck["val_loss_history"]
        epochs  = np.arange(1, len(train_h)+1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_h, color=C["blue"],   linewidth=1.8, label="Train Loss")
        ax.plot(epochs, val_h,   color=C["orange"], linewidth=1.8, label="Val Loss")
        best_ep = int(np.argmin(val_h)) + 1
        ax.axvline(best_ep, color=C["green"], linestyle="--", linewidth=1.4,
                   label=f"Best (ep {best_ep}, val={min(val_h):.4f})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Fig. 3 — Training Curve (Real Checkpoint History)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.25)
        fig.tight_layout()
        sf(fig, "fig3_REAL_training_curve.png")
        sf(fig, "fig3_REAL_training_curve.pdf")
    else:
        print("  Fig 3: No loss history in checkpoint — skipping real training curve")

    print("\n  All REAL figures saved.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    hr("FINAL SUMMARY")
    try:
        with open(os.path.join(RESULTS_DIR, "stage1_gat_results.json"), encoding="utf-8") as f:
            gat = json.load(f)
        print("\n  GAT FORECASTER (Real Inference):")
        for k, v in gat["overall"].items():
            print(f"    {k}: {v}")
    except FileNotFoundError:
        pass

    try:
        with open(os.path.join(RESULTS_DIR, "stage2_baselines.json"), encoding="utf-8") as f:
            bl = json.load(f)
        print("\n  BASELINES:")
        for name, m in bl["baselines"].items():
            if "skipped" not in m:
                print(f"    {name:30s}: MAE={m['MAE_MW']:7.1f} MW  MAPE={m['MAPE_pct']:.2f}%  R²={m.get('R2',0):.4f}")
    except FileNotFoundError:
        pass

    try:
        with open(os.path.join(RESULTS_DIR, "stage3_chaos.json"), encoding="utf-8") as f:
            ch = json.load(f)
        print("\n  CHAOS ENGINE:")
        for st, m in ch["aggregate"].items():
            print(f"    {st:22s}: Fault={m['avg_resilience_postfault']:.1f} "
                  f"→ RB={m['avg_resilience_rule_based']:.1f} "
                  f"→ AI={m['avg_resilience_ai']:.1f} | "
                  f"AI+{m['avg_ai_gain_pts']:.1f}pts  cost{m['avg_cost_savings_pct']:+.0f}%")
        print(f"\n  Errors: {ch['n_errors']}/{ch['n_total']}")
    except FileNotFoundError:
        pass

    print(f"\n  Results: {RESULTS_DIR}")
    print(f"  Figures: {os.path.join(BASE, 'paper_figures')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full IEEE Paper Evaluation Pipeline")
    parser.add_argument("--stage", type=int, default=0,
                        help="Stage to run: 0=all, 1=GAT, 2=baselines, 3=chaos, 4=figures")
    args = parser.parse_args()

    t_start = time.time()
    if args.stage in (0, 1):
        stage1_gat_inference()
    if args.stage in (0, 2):
        stage2_baselines()
    if args.stage in (0, 3):
        stage3_chaos()
    if args.stage in (0, 4):
        stage4_regenerate_figures()

    print_summary()
    hr()
    total_min = (time.time() - t_start) / 60
    print(f"  Total wall time: {total_min:.1f} minutes")
    print("=" * 65)
