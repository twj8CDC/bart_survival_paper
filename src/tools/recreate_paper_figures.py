#!/usr/bin/env python3
"""
Publication Figure Recreation Script: Option A.
Recreates the exact manuscript figures (Figures 1, 2, and 3) comparing:
  - p-BART (New 2026 Refactored)
  - p-BART (Old Historical)
  - r-BART (OG from R::BART package)
  - CPHM (Cox Proportional Hazards) / KM (Kaplan-Meier)
Preserves the exact grayscale shading, layout, percentile ticks, and aesthetics of the published paper.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLRUNS_DIR = BASE_DIR / "archive" / "mlruns" if (BASE_DIR / "archive" / "mlruns").exists() else BASE_DIR / "mlruns"
MLRUNS2_DIR = BASE_DIR / "archive" / "mlruns2" if (BASE_DIR / "archive" / "mlruns2").exists() else BASE_DIR / "mlruns2"
COMPLEX3_DIR = BASE_DIR / "archive" / "complex3" if (BASE_DIR / "archive" / "complex3").exists() else BASE_DIR / "complex3"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_original_complex12():
    """Loads historical 1,000-replicate data for Complex 1 & 2 from MLflow."""
    artf_dir = MLRUNS2_DIR / "539750630783382144" / "15831a8d36b34ec89107c0fd2695b786" / "artifacts"
    
    # 7 quantiles: [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.01] -> [:, 1:-1] selects [0.90, 0.75, 0.50, 0.25, 0.10]
    data = {}
    for scen in ["cmplx1", "cmplx2"]:
        for model in ["p", "r", "c"]:
            fname = f"{scen}_400_met_{model}.json"
            with open(artf_dir / fname) as f:
                d = json.load(f)
            data[f"{scen}_{model}_rmse"] = np.array(d["rmse"])[:, 1:-1]
            data[f"{scen}_{model}_bias"] = np.array(d["bias"])[:, 1:-1]
            
    return data


def load_original_complex3():
    """Loads historical out-of-sample hold-out test cohort data for Complex 3."""
    csv_path = COMPLEX3_DIR / "complex1_N_400_tst.csv"
    df = pd.read_csv(csv_path)
    return df


def load_new_benchmark_data(experiment_name="suite_400_og_comparison"):
    """Loads new 2026 benchmark data with per-interval metrics."""
    suite_dir = RESULTS_DIR / experiment_name
    new_data = {}
    
    if not suite_dir.exists():
        print(f"Warning: {suite_dir} does not exist yet. New model entries will be omitted.")
        return new_data
        
    for scen_dir in suite_dir.glob("*"):
        if not scen_dir.is_dir():
            continue
        rep_file = scen_dir / "replicate_metrics.json"
        if not rep_file.exists():
            continue
            
        with open(rep_file) as f:
            reps = json.load(f)
            
        # Parse into per-scenario, per-model arrays (both pooled and per-N)
        scen_id = reps[0].get("scenario_id")
        for r in reps:
            m = r.get("model")
            n_val = r.get("n")
            keys = [f"{scen_id}_{m}"]
            if n_val is not None:
                keys.append(f"{scen_id}_N{n_val}_{m}")
            for key_base in keys:
                if "rmse_per_interval" in r and r["rmse_per_interval"] is not None:
                    new_data.setdefault(f"{key_base}_rmse", []).append(r["rmse_per_interval"])
                if "bias_per_interval" in r and r["bias_per_interval"] is not None:
                    new_data.setdefault(f"{key_base}_bias", []).append(r["bias_per_interval"])
                if "test_rmse_per_interval" in r and r["test_rmse_per_interval"] is not None:
                    new_data.setdefault(f"{key_base}_test_rmse", []).append(r["test_rmse_per_interval"])
                if "test_bias_per_interval" in r and r["test_bias_per_interval"] is not None:
                    new_data.setdefault(f"{key_base}_test_bias", []).append(r["test_bias_per_interval"])
                if "coverage" in r and r["coverage"] is not None:
                    new_data.setdefault(f"{key_base}_coverage", []).append(r["coverage"])
                if "rmse" in r and r["rmse"] is not None:
                    new_data.setdefault(f"{key_base}_scalar_rmse", []).append(r["rmse"])
                if "bias" in r and r["bias"] is not None:
                    new_data.setdefault(f"{key_base}_scalar_bias", []).append(r["bias"])
                
    for k in new_data:
        new_data[k] = np.array(new_data[k])
        
    return new_data


def recreate_fig2_complex12(og_data, new_data=None):
    """
    Recreates Figure 2 of the manuscript (Complex 1 & 2, PH vs. Non-PH).
    Plots 4 comparison groups: [p-BART (New), p-BART (Old), r-BART (OG), CPHM]
    across 5 time percentiles: [10, 25, 50, 75, 90].
    """
    lbls = ["10", "25", "50", "75", "90"]
    pos = np.array([1, 4, 7, 10, 13])
    
    # Offsets for 4 models
    # p-BART (New): -0.9, p-BART (Old): -0.3, r-BART (OG): +0.3, CPHM: +0.9
    offsets = [-0.9, -0.3, 0.3, 0.9]
    widths = 0.45
    
    scenarios = [
        ("cmplx1", "Complex 1 (Proportional Hazards)", "fig2_recreated_complex1_prop.png"),
        ("cmplx2", "Complex 2 (Non-Proportional Hazards)", "fig2_recreated_complex2_nonprop.png")
    ]
    
    for prefix, title_desc, fname in scenarios:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), dpi=300)
        
        # Data extractions
        rmse_old_p = og_data[f"{prefix}_p_rmse"] # (1000, 5)
        rmse_og_r = og_data[f"{prefix}_r_rmse"]   # (1000, 5)
        rmse_cphm = og_data[f"{prefix}_c_rmse"]   # (1000, 5)
        
        bias_old_p = og_data[f"{prefix}_p_bias"]
        bias_og_r = og_data[f"{prefix}_r_bias"]
        bias_cphm = og_data[f"{prefix}_c_bias"]
        
        scen_key = "complex_1" if prefix == "cmplx1" else "complex_2"
        has_new = (new_data and f"{scen_key}_pbart_rmse" in new_data and len(new_data[f"{scen_key}_pbart_rmse"]) > 0)
        if has_new:
            # Sliced to match [:, 1:-1]
            rmse_new_p = new_data[f"{scen_key}_pbart_rmse"][:, 1:-1] # type: ignore
            bias_new_p = new_data[f"{scen_key}_pbart_bias"][:, 1:-1] # type: ignore
        
        # -------------------- PANEL 1: RMSE --------------------
        ax = axes[0]
        # p-BART New (2026)
        if has_new:
            ax.boxplot(
                rmse_new_p, positions=pos + offsets[0], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111", "alpha": 1.0},
                capprops={"color": "#111111", "alpha": 1.0},
                flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
                label="p-BART (New 2026)"
            )
            
        # p-BART Old
        ax.boxplot(
            rmse_old_p, positions=pos + offsets[1], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#555555", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "alpha": 0.75},
            capprops={"color": "#555555", "alpha": 0.75},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="p-BART (Old OG)"
        )
        
        # r-BART OG
        ax.boxplot(
            rmse_og_r, positions=pos + offsets[2], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#888888", "alpha": 0.50},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#888888", "alpha": 0.50},
            capprops={"color": "#888888", "alpha": 0.50},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="r-BART (OG)"
        )
        
        # CPHM
        ax.boxplot(
            rmse_cphm, positions=pos + offsets[3], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#cccccc", "alpha": 0.35},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#777777", "alpha": 0.35},
            capprops={"color": "#777777", "alpha": 0.35},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="CPHM"
        )
        
        ax.set_xticks(pos)
        ax.set_xticklabels(lbls, fontsize=11)
        ax.set_xlabel("Survival Time Percentile", fontsize=11, fontweight="bold")
        ax.set_ylabel("Root Mean Squared Error (a-RMSE)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 0.25)
        ax.set_title(f"A. Calibration Error (RMSE) - {title_desc}", fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        
        # -------------------- PANEL 2: BIAS --------------------
        ax = axes[1]
        if has_new:
            ax.boxplot(
                bias_new_p, positions=pos + offsets[0], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111", "alpha": 1.0},
                capprops={"color": "#111111", "alpha": 1.0},
                flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
                label="p-BART (New 2026)"
            )
            
        ax.boxplot(
            bias_old_p, positions=pos + offsets[1], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#555555", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "alpha": 0.75},
            capprops={"color": "#555555", "alpha": 0.75},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="p-BART (Old OG)"
        )
        
        ax.boxplot(
            bias_og_r, positions=pos + offsets[2], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#888888", "alpha": 0.50},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#888888", "alpha": 0.50},
            capprops={"color": "#888888", "alpha": 0.50},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="r-BART (OG)"
        )
        
        ax.boxplot(
            bias_cphm, positions=pos + offsets[3], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#cccccc", "alpha": 0.35},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#777777", "alpha": 0.35},
            capprops={"color": "#777777", "alpha": 0.35},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="CPHM"
        )
        
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(lbls, fontsize=11)
        ax.set_xlabel("Survival Time Percentile", fontsize=11, fontweight="bold")
        ax.set_ylabel("Signed Calibration Bias", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.15, 0.15)
        ax.set_title(f"B. Signed Bias - {title_desc}", fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        
        plt.tight_layout()
        out_path = FIGURES_DIR / fname
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")


def recreate_fig3_complex3(og_df_n400, new_data=None):
    """
    Recreates Figure 3 of the manuscript (Complex 3 Hold-out Out-of-Sample Test Cohort).
    Plots [p-BART (New), p-BART (Old), r-BART (OG), CPHM] across deciles [0.1, 0.3, 0.5, 0.7, 0.9] for N=400.
    """
    deciles = [0.1, 0.3, 0.5, 0.7, 0.9]
    centers = np.array([1.5, 4.0, 6.5, 9.0, 11.5])
    
    offsets = [-0.65, -0.22, 0.22, 0.65]
    widths = 0.36
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), dpi=300)
    
    has_new = (new_data and "complex_3_pbart_test_rmse" in new_data and len(new_data["complex_3_pbart_test_rmse"]) > 0)
    
    # Map deciles to indices in new test_rmse_per_interval
    # time_quantiles in Complex 3: [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    # Indices for [0.1, 0.3, 0.5, 0.7, 0.9] are: [1, 3, 5, 7, 9]
    decile_indices = {0.1: 1, 0.3: 3, 0.5: 5, 0.7: 7, 0.9: 9}
    
    # -------------------- PANEL 1: Hold-out RMSE --------------------
    ax = axes[0]
    for d_idx, d_val in enumerate(deciles):
        pos_base = centers[d_idx]
        sub_og = og_df_n400[og_df_n400["deciles"] == d_val]
        
        # p-BART New
        if has_new:
            col_idx = decile_indices[d_val]
            new_vals = new_data["complex_3_pbart_test_rmse"][:, col_idx] # type: ignore
            ax.boxplot(
                new_vals, positions=[pos_base + offsets[0]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111"},
                capprops={"color": "#111111"},
                flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
                label="p-BART (New 2026)" if d_idx == 0 else None
            )
            
        # p-BART Old
        ax.boxplot(
            sub_og["p_rmse"].dropna(), positions=[pos_base + offsets[1]], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#555555", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="p-BART (Old OG)" if d_idx == 0 else None
        )
        
        # r-BART OG
        ax.boxplot(
            sub_og["r_rmse"].dropna(), positions=[pos_base + offsets[2]], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#888888", "alpha": 0.50},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#888888"},
            capprops={"color": "#888888"},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="r-BART (OG)" if d_idx == 0 else None
        )
        
        # CPHM
        ax.boxplot(
            sub_og["c_rmse"].dropna(), positions=[pos_base + offsets[3]], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#cccccc", "alpha": 0.35},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#777777"},
            capprops={"color": "#777777"},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="CPHM" if d_idx == 0 else None
        )
        
    ax.set_xticks(centers)
    ax.set_xticklabels([str(d) for d in deciles], fontsize=11)
    ax.set_xlabel("Time Decile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample a-RMSE", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.001, 0.25)
    ax.set_title("A. Hold-out Test Cohort Error (RMSE) - Complex 3 (N=400)", fontsize=12, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    
    # -------------------- PANEL 2: Hold-out Bias --------------------
    ax = axes[1]
    for d_idx, d_val in enumerate(deciles):
        pos_base = centers[d_idx]
        sub_og = og_df_n400[og_df_n400["deciles"] == d_val]
        
        if has_new:
            col_idx = decile_indices[d_val]
            new_vals = new_data["complex_3_pbart_test_bias"][:, col_idx] # type: ignore
            ax.boxplot(
                new_vals, positions=[pos_base + offsets[0]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111"},
                capprops={"color": "#111111"},
                flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
                label="p-BART (New 2026)" if d_idx == 0 else None
            )
            
        ax.boxplot(
            sub_og["p_bias"].dropna(), positions=[pos_base + offsets[1]], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#555555", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="p-BART (Old OG)" if d_idx == 0 else None
        )
        
        ax.boxplot(
            sub_og["r_bias"].dropna(), positions=[pos_base + offsets[2]], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#888888", "alpha": 0.50},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#888888"},
            capprops={"color": "#888888"},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="r-BART (OG)" if d_idx == 0 else None
        )
        
        ax.boxplot(
            sub_og["c_bias"].dropna(), positions=[pos_base + offsets[3]], widths=widths, patch_artist=True,
            boxprops={"facecolor": "#cccccc", "alpha": 0.35},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "#777777"},
            capprops={"color": "#777777"},
            flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
            label="CPHM" if d_idx == 0 else None
        )
        
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(d) for d in deciles], fontsize=11)
    ax.set_xlabel("Time Decile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample Signed Bias", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.075, 0.075)
    ax.set_title("B. Hold-out Test Cohort Signed Bias - Complex 3 (N=400)", fontsize=12, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_recreated_complex3_holdout.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def recreate_fig1_simple(new_data=None):
    """
    Recreates Figure 1 of the manuscript (Simple 1 & Simple 2).
    Strip/jitter plots comparing p-BART (New 2026), p-BART (Old), r-BART, and KM across RMSE, Bias, and Coverage.
    """
    artf_dir = MLRUNS_DIR / "489610385670369716" / "04d8e19fba604aae852bc2bb9d992ee2" / "artifacts"
    
    def sctr_plt(center, size, scale=0.015):
        return np.random.normal(center, scale, size)
        
    configs = [
        ("Simple 1 (One-Sample)", ["1_1", "1_2"], ["simple_1_1", "simple_1_2"], "fig1_recreated_simple1.png", (0, 0.05)),
        ("Simple 2 (Two-Sample)", ["2_1", "2_2"], ["simple_2_1", "simple_2_2"], "fig1_recreated_simple2.png", (0, 0.10))
    ]
    
    for scen_name, prefixes, new_scen_keys, fname, ylim_rmse in configs:
        rmse_p, rmse_r, rmse_k = [], [], []
        bias_p, bias_r, bias_k = [], [], []
        cov_p, cov_r, cov_k = [], [], []
        
        for pref in prefixes:
            with open(artf_dir / f"{pref}_400_met_p.json") as f:
                p = json.load(f)
            with open(artf_dir / f"{pref}_400_met_r.json") as f:
                r = json.load(f)
            with open(artf_dir / f"{pref}_400_met_k.json") as f:
                k = json.load(f)
                
            rmse_p.extend(p["rmse"])
            rmse_r.extend(r["rmse"])
            rmse_k.extend(k["rmse"])
            
            bias_p.extend(p["bias"])
            bias_r.extend(r["bias"])
            bias_k.extend(k["bias"])
            
            cov_p.extend(p.get("cov_ci", p.get("cov", [])))
            cov_r.extend(r.get("cov_ci", r.get("cov", [])))
            cov_k.extend(k.get("cov_ci", k.get("cov", [])))
            
        rmse_p = np.array(rmse_p)
        rmse_r = np.array(rmse_r)
        rmse_k = np.array(rmse_k)
        bias_p = np.array(bias_p)
        bias_r = np.array(bias_r)
        bias_k = np.array(bias_k)
        cov_p = np.array(cov_p)
        cov_r = np.array(cov_r)
        cov_k = np.array(cov_k)
        
        # Check for new 2026 p-BART data
        rmse_new_p, bias_new_p, cov_new_p = [], [], []
        if new_data:
            for sk in new_scen_keys:
                key_rmse = f"{sk}_pbart_rmse"
                key_bias = f"{sk}_pbart_bias"
                key_cov = f"{sk}_pbart_coverage"
                if key_rmse in new_data and len(new_data[key_rmse]) > 0:
                    mean_rmse_inv = np.mean(new_data[key_rmse], axis=0)
                    rmse_new_p.extend(mean_rmse_inv)
                if key_bias in new_data and len(new_data[key_bias]) > 0:
                    mean_bias_inv = np.mean(new_data[key_bias], axis=0)
                    bias_new_p.extend(mean_bias_inv)
                if key_cov in new_data and len(new_data[key_cov]) > 0:
                    cov_vals = new_data[key_cov]
                    mean_cov = float(np.mean(cov_vals))
                    cov_new_p.extend([mean_cov] * 5)
                    
        has_new = (len(rmse_new_p) > 0)
        if has_new:
            rmse_new_p = np.array(rmse_new_p)
            bias_new_p = np.array(bias_new_p)
            cov_new_p = np.array(cov_new_p)
            x_val = [0.85, 1.10, 1.35, 1.60]
            labels = ["p-BART\n(New 2026)", "p-BART\n(Old OG)", "r-BART\n(OG)", "KM"]
            xlim = (0.70, 1.75)
        else:
            x_val = [1.0, 1.25, 1.5]
            labels = ["p-BART", "r-BART", "KM"]
            xlim = (0.85, 1.65)
            
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.5), dpi=300)
        
        # 1. RMSE
        ax = axes[0]
        if has_new:
            ax.plot(sctr_plt(x_val[0], len(rmse_new_p)), rmse_new_p, "D", color="#111111", alpha=0.75, markersize=5.5, label="p-BART (New 2026)")
            ax.plot(sctr_plt(x_val[1], len(rmse_p)), rmse_p, "s", color="#555555", alpha=0.6, markersize=5.5, label="p-BART (Old OG)")
            ax.plot(sctr_plt(x_val[2], len(rmse_r)), rmse_r, "^", color="#888888", alpha=0.6, markersize=5.5, label="r-BART (OG)")
            ax.plot(sctr_plt(x_val[3], len(rmse_k)), rmse_k, "o", color="#aaaaaa", alpha=0.6, markersize=5.5, label="KM")
        else:
            ax.plot(sctr_plt(x_val[0], len(rmse_p)), rmse_p, "s", color="#555555", alpha=0.6, label="p-BART")
            ax.plot(sctr_plt(x_val[1], len(rmse_r)), rmse_r, "^", color="#888888", alpha=0.6, label="r-BART")
            ax.plot(sctr_plt(x_val[2], len(rmse_k)), rmse_k, "o", color="#aaaaaa", alpha=0.6, label="KM")
        ax.set_xticks(x_val)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Root Mean Squared Error (a-RMSE)", fontsize=11, fontweight="bold")
        ax.set_ylim(ylim_rmse)
        ax.set_xlim(xlim)
        ax.set_title(f"A. Calibration Error (RMSE)\n{scen_name}", fontsize=11, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 2. Bias
        ax = axes[1]
        if has_new:
            ax.plot(sctr_plt(x_val[0], len(bias_new_p)), bias_new_p, "D", color="#111111", alpha=0.75, markersize=5.5)
            ax.plot(sctr_plt(x_val[1], len(bias_p)), bias_p, "s", color="#555555", alpha=0.6, markersize=5.5)
            ax.plot(sctr_plt(x_val[2], len(bias_r)), bias_r, "^", color="#888888", alpha=0.6, markersize=5.5)
            ax.plot(sctr_plt(x_val[3], len(bias_k)), bias_k, "o", color="#aaaaaa", alpha=0.6, markersize=5.5)
        else:
            ax.plot(sctr_plt(x_val[0], len(bias_p)), bias_p, "s", color="#555555", alpha=0.6)
            ax.plot(sctr_plt(x_val[1], len(bias_r)), bias_r, "^", color="#888888", alpha=0.6)
            ax.plot(sctr_plt(x_val[2], len(bias_k)), bias_k, "o", color="#aaaaaa", alpha=0.6)
        ax.axhline(0, color="black", alpha=0.4, linestyle="--", linewidth=1.0)
        ax.set_xticks(x_val)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Signed Calibration Bias", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.06, 0.06)
        ax.set_xlim(xlim)
        ax.set_title(f"B. Signed Bias\n{scen_name}", fontsize=11, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 3. Coverage
        ax = axes[2]
        if has_new:
            ax.plot(sctr_plt(x_val[0], len(cov_new_p)), cov_new_p, "D", color="#111111", alpha=0.75, markersize=5.5)
            ax.plot(sctr_plt(x_val[1], len(cov_p)), cov_p, "s", color="#555555", alpha=0.6, markersize=5.5)
            ax.plot(sctr_plt(x_val[2], len(cov_r)), cov_r, "^", color="#888888", alpha=0.6, markersize=5.5)
            ax.plot(sctr_plt(x_val[3], len(cov_k)), cov_k, "o", color="#aaaaaa", alpha=0.6, markersize=5.5)
        else:
            ax.plot(sctr_plt(x_val[0], len(cov_p)), cov_p, "s", color="#555555", alpha=0.6)
            ax.plot(sctr_plt(x_val[1], len(cov_r)), cov_r, "^", color="#888888", alpha=0.6)
            ax.plot(sctr_plt(x_val[2], len(cov_k)), cov_k, "o", color="#aaaaaa", alpha=0.6)
        ax.axhline(0.95, color="black", alpha=0.4, linestyle="--", linewidth=1.0, label="Nominal 95%")
        ax.set_xticks(x_val)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Empirical 95% Coverage", fontsize=11, fontweight="bold")
        ax.set_ylim(0.40, 1.02)
        ax.set_xlim(xlim)
        ax.set_title(f"C. 95% Credible / Confidence Coverage\n{scen_name}", fontsize=11, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
def recreate_fig1_simple_new_only(new_data):
    """
    Recreates Figure 1 using ONLY the new benchmark runs (p-BART, r-BART, KM).
    Preserves exact manuscript layout, jitter, and grayscale aesthetics.
    """
    def sctr_plt(center, size, scale=0.015):
        return np.random.normal(center, scale, size)
        
    configs = [
        ("Simple 1 (One-Sample)", ["simple_1_1", "simple_1_2"], "fig1_paper_simple1.png", (0, 0.05)),
        ("Simple 2 (Two-Sample)", ["simple_2_1", "simple_2_2"], "fig1_paper_simple2.png", (0, 0.10))
    ]
    
    x_val = [1.0, 1.25, 1.5]
    labels = ["p-BART", "r-BART", "KM"]
    xlim = (0.85, 1.65)
    
    for scen_name, scen_keys, fname, ylim_rmse in configs:
        rmse_p, rmse_r, rmse_k = [], [], []
        bias_p, bias_r, bias_k = [], [], []
        cov_p, cov_r, cov_k = [], [], []
        
        for sk in scen_keys:
            # Extract scalar metrics if available, or mean across intervals
            if f"{sk}_pbart_scalar_rmse" in new_data:
                rmse_p.extend(new_data[f"{sk}_pbart_scalar_rmse"])
                bias_p.extend(new_data[f"{sk}_pbart_scalar_bias"])
                cov_p.extend(new_data[f"{sk}_pbart_coverage"])
            elif f"{sk}_pbart_rmse" in new_data:
                rmse_p.extend(np.mean(new_data[f"{sk}_pbart_rmse"], axis=1))
                bias_p.extend(np.mean(new_data[f"{sk}_pbart_bias"], axis=1))
                cov_p.extend(new_data[f"{sk}_pbart_coverage"])
                
            if f"{sk}_rbart_scalar_rmse" in new_data:
                rmse_r.extend(new_data[f"{sk}_rbart_scalar_rmse"])
                bias_r.extend(new_data[f"{sk}_rbart_scalar_bias"])
                cov_r.extend(new_data[f"{sk}_rbart_coverage"])
            elif f"{sk}_rbart_rmse" in new_data:
                rmse_r.extend(np.mean(new_data[f"{sk}_rbart_rmse"], axis=1))
                bias_r.extend(np.mean(new_data[f"{sk}_rbart_bias"], axis=1))
                cov_r.extend(new_data[f"{sk}_rbart_coverage"])
                
            if f"{sk}_km_scalar_rmse" in new_data:
                rmse_k.extend(new_data[f"{sk}_km_scalar_rmse"])
                bias_k.extend(new_data[f"{sk}_km_scalar_bias"])
                cov_k.extend(new_data[f"{sk}_km_coverage"])
            elif f"{sk}_km_rmse" in new_data:
                rmse_k.extend(np.mean(new_data[f"{sk}_km_rmse"], axis=1))
                bias_k.extend(np.mean(new_data[f"{sk}_km_bias"], axis=1))
                cov_k.extend(new_data[f"{sk}_km_coverage"])
                
        rmse_p, rmse_r, rmse_k = np.array(rmse_p), np.array(rmse_r), np.array(rmse_k)
        bias_p, bias_r, bias_k = np.array(bias_p), np.array(bias_r), np.array(bias_k)
        cov_p, cov_r, cov_k = np.array(cov_p), np.array(cov_r), np.array(cov_k)
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.5), dpi=300)
        
        # 1. RMSE
        ax = axes[0]
        if len(rmse_p) > 0:
            ax.plot(sctr_plt(x_val[0], len(rmse_p)), rmse_p, "s", color="#111111", alpha=0.75, markersize=5.5, label="p-BART")
        if len(rmse_r) > 0:
            ax.plot(sctr_plt(x_val[1], len(rmse_r)), rmse_r, "^", color="#666666", alpha=0.75, markersize=5.5, label="r-BART")
        if len(rmse_k) > 0:
            ax.plot(sctr_plt(x_val[2], len(rmse_k)), rmse_k, "o", color="#aaaaaa", alpha=0.75, markersize=5.5, label="KM")
        ax.set_xticks(x_val)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Root Mean Squared Error (a-RMSE)", fontsize=11, fontweight="bold")
        ax.set_ylim(ylim_rmse)
        ax.set_xlim(xlim)
        ax.set_title(f"A. Calibration Error (RMSE)\n{scen_name}", fontsize=11, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 2. Bias
        ax = axes[1]
        if len(bias_p) > 0:
            ax.plot(sctr_plt(x_val[0], len(bias_p)), bias_p, "s", color="#111111", alpha=0.75, markersize=5.5)
        if len(bias_r) > 0:
            ax.plot(sctr_plt(x_val[1], len(bias_r)), bias_r, "^", color="#666666", alpha=0.75, markersize=5.5)
        if len(bias_k) > 0:
            ax.plot(sctr_plt(x_val[2], len(bias_k)), bias_k, "o", color="#aaaaaa", alpha=0.75, markersize=5.5)
        ax.axhline(0, color="black", alpha=0.4, linestyle="--", linewidth=1.0)
        ax.set_xticks(x_val)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Signed Calibration Bias", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.06, 0.06)
        ax.set_xlim(xlim)
        ax.set_title(f"B. Signed Bias\n{scen_name}", fontsize=11, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 3. Coverage
        ax = axes[2]
        if len(cov_p) > 0:
            ax.plot(sctr_plt(x_val[0], len(cov_p)), cov_p, "s", color="#111111", alpha=0.75, markersize=5.5)
        if len(cov_r) > 0:
            ax.plot(sctr_plt(x_val[1], len(cov_r)), cov_r, "^", color="#666666", alpha=0.75, markersize=5.5)
        if len(cov_k) > 0:
            ax.plot(sctr_plt(x_val[2], len(cov_k)), cov_k, "o", color="#aaaaaa", alpha=0.75, markersize=5.5)
        ax.axhline(0.95, color="black", alpha=0.4, linestyle="--", linewidth=1.0, label="Nominal 95%")
        ax.set_xticks(x_val)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Empirical 95% Coverage", fontsize=11, fontweight="bold")
        ax.set_ylim(0.40, 1.02)
        ax.set_xlim(xlim)
        ax.set_title(f"C. 95% Credible / Confidence Coverage\n{scen_name}", fontsize=11, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        out_path = FIGURES_DIR / fname
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")


def recreate_fig2_complex12_new_only(new_data):
    """
    Recreates Figure 2 using ONLY the new benchmark runs (p-BART, r-BART, CPHM).
    Plots side-by-side boxplots across 5 time percentiles: [10, 25, 50, 75, 90].
    """
    lbls = ["10", "25", "50", "75", "90"]
    pos = np.array([1, 4, 7, 10, 13])
    offsets = [-0.6, 0.0, 0.6]
    widths = 0.5
    
    scenarios = [
        ("complex_1", "Complex 1 (Proportional Hazards)", "fig2_paper_complex1_prop.png"),
        ("complex_2", "Complex 2 (Non-Proportional Hazards)", "fig2_paper_complex2_nonprop.png")
    ]
    
    for scen_key, title_desc, fname in scenarios:
        has_p = f"{scen_key}_pbart_rmse" in new_data and len(new_data[f"{scen_key}_pbart_rmse"]) > 0
        has_r = f"{scen_key}_rbart_rmse" in new_data and len(new_data[f"{scen_key}_rbart_rmse"]) > 0
        has_c = f"{scen_key}_coxph_rmse" in new_data and len(new_data[f"{scen_key}_coxph_rmse"]) > 0
        
        if not (has_p or has_r or has_c):
            continue
            
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), dpi=300)
        
        # -------------------- PANEL 1: RMSE --------------------
        ax = axes[0]
        if has_p:
            rmse_p = new_data[f"{scen_key}_pbart_rmse"][:, 1:-1]
            ax.boxplot(
                rmse_p, positions=pos + offsets[0], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111", "alpha": 1.0},
                capprops={"color": "#111111", "alpha": 1.0},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="p-BART"
            )
        if has_r:
            rmse_r = new_data[f"{scen_key}_rbart_rmse"][:, 1:-1]
            ax.boxplot(
                rmse_r, positions=pos + offsets[1], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#777777", "alpha": 0.70},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#777777", "alpha": 0.70},
                capprops={"color": "#777777", "alpha": 0.70},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="r-BART"
            )
        if has_c:
            rmse_c = new_data[f"{scen_key}_coxph_rmse"][:, 1:-1]
            ax.boxplot(
                rmse_c, positions=pos + offsets[2], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#cccccc", "alpha": 0.50},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#999999", "alpha": 0.50},
                capprops={"color": "#999999", "alpha": 0.50},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="CPHM"
            )
            
        ax.set_xticks(pos)
        ax.set_xticklabels(lbls, fontsize=11)
        ax.set_xlabel("Survival Time Percentile", fontsize=11, fontweight="bold")
        ax.set_ylabel("Root Mean Squared Error (a-RMSE)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 0.25)
        ax.set_title(f"A. Calibration Error (RMSE) - {title_desc}", fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        
        # -------------------- PANEL 2: BIAS --------------------
        ax = axes[1]
        if has_p:
            bias_p = new_data[f"{scen_key}_pbart_bias"][:, 1:-1]
            ax.boxplot(
                bias_p, positions=pos + offsets[0], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111", "alpha": 1.0},
                capprops={"color": "#111111", "alpha": 1.0},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="p-BART"
            )
        if has_r:
            bias_r = new_data[f"{scen_key}_rbart_bias"][:, 1:-1]
            ax.boxplot(
                bias_r, positions=pos + offsets[1], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#777777", "alpha": 0.70},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#777777", "alpha": 0.70},
                capprops={"color": "#777777", "alpha": 0.70},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="r-BART"
            )
        if has_c:
            bias_c = new_data[f"{scen_key}_coxph_bias"][:, 1:-1]
            ax.boxplot(
                bias_c, positions=pos + offsets[2], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#cccccc", "alpha": 0.50},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#999999", "alpha": 0.50},
                capprops={"color": "#999999", "alpha": 0.50},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="CPHM"
            )
            
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(lbls, fontsize=11)
        ax.set_xlabel("Survival Time Percentile", fontsize=11, fontweight="bold")
        ax.set_ylabel("Signed Calibration Bias", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.15, 0.15)
        ax.set_title(f"B. Signed Bias - {title_desc}", fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        
        plt.tight_layout()
        out_path = FIGURES_DIR / fname
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")


def recreate_fig3_complex3_new_only(new_data):
    """
    Recreates Figure 3 using ONLY the new benchmark runs (p-BART, r-BART, CPHM).
    Plots side-by-side boxplots across deciles [0.1, 0.3, 0.5, 0.7, 0.9] for hold-out test cohort.
    """
    deciles = [0.1, 0.3, 0.5, 0.7, 0.9]
    centers = np.array([1.5, 4.0, 6.5, 9.0, 11.5])
    offsets = [-0.5, 0.0, 0.5]
    widths = 0.42
    decile_indices = {0.1: 1, 0.3: 3, 0.5: 5, 0.7: 7, 0.9: 9}
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), dpi=300)
    
    p_key = "complex_3_N400_pbart" if "complex_3_N400_pbart_test_rmse" in new_data else "complex_3_pbart"
    r_key = "complex_3_N400_rbart" if "complex_3_N400_rbart_test_rmse" in new_data else "complex_3_rbart"
    c_key = "complex_3_N400_coxph" if "complex_3_N400_coxph_test_rmse" in new_data else "complex_3_coxph"

    has_p = f"{p_key}_test_rmse" in new_data and len(new_data[f"{p_key}_test_rmse"]) > 0
    has_r = f"{r_key}_test_rmse" in new_data and len(new_data[f"{r_key}_test_rmse"]) > 0
    has_c = f"{c_key}_test_rmse" in new_data and len(new_data[f"{c_key}_test_rmse"]) > 0
    
    # -------------------- PANEL 1: Hold-out RMSE --------------------
    ax = axes[0]
    for d_idx, d_val in enumerate(deciles):
        pos_base = centers[d_idx]
        col_idx = decile_indices[d_val]
        
        if has_p:
            vals_p = new_data[f"{p_key}_test_rmse"][:, col_idx]
            ax.boxplot(
                vals_p, positions=[pos_base + offsets[0]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111"},
                capprops={"color": "#111111"},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="p-BART" if d_idx == 0 else None
            )
        if has_r:
            vals_r = new_data[f"{r_key}_test_rmse"][:, col_idx]
            ax.boxplot(
                vals_r, positions=[pos_base + offsets[1]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#777777", "alpha": 0.70},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#777777"},
                capprops={"color": "#777777"},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="r-BART" if d_idx == 0 else None
            )
        if has_c:
            vals_c = new_data[f"{c_key}_test_rmse"][:, col_idx]
            ax.boxplot(
                vals_c, positions=[pos_base + offsets[2]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#cccccc", "alpha": 0.50},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#999999"},
                capprops={"color": "#999999"},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="CPHM" if d_idx == 0 else None
            )
            
    ax.set_xticks(centers)
    ax.set_xticklabels([str(d) for d in deciles], fontsize=11)
    ax.set_xlabel("Time Decile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample a-RMSE", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.001, 0.25)
    ax.set_title("A. Hold-out Test Cohort Error (RMSE) - Complex 3 (N=400)", fontsize=12, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    
    # -------------------- PANEL 2: Hold-out Bias --------------------
    ax = axes[1]
    for d_idx, d_val in enumerate(deciles):
        pos_base = centers[d_idx]
        col_idx = decile_indices[d_val]
        
        if has_p:
            vals_p = new_data[f"{p_key}_test_bias"][:, col_idx]
            ax.boxplot(
                vals_p, positions=[pos_base + offsets[0]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#111111", "alpha": 1.0},
                medianprops={"color": "white", "linewidth": 1.2},
                whiskerprops={"color": "#111111"},
                capprops={"color": "#111111"},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="p-BART" if d_idx == 0 else None
            )
        if has_r:
            vals_r = new_data[f"{r_key}_test_bias"][:, col_idx]
            ax.boxplot(
                vals_r, positions=[pos_base + offsets[1]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#777777", "alpha": 0.70},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#777777"},
                capprops={"color": "#777777"},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="r-BART" if d_idx == 0 else None
            )
        if has_c:
            vals_c = new_data[f"{c_key}_test_bias"][:, col_idx]
            ax.boxplot(
                vals_c, positions=[pos_base + offsets[2]], widths=widths, patch_artist=True,
                boxprops={"facecolor": "#cccccc", "alpha": 0.50},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#999999"},
                capprops={"color": "#999999"},
                flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
                label="CPHM" if d_idx == 0 else None
            )
            
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(d) for d in deciles], fontsize=11)
    ax.set_xlabel("Time Decile", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Sample Signed Bias", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.075, 0.075)
    ax.set_title("B. Hold-out Test Cohort Signed Bias - Complex 3 (N=400)", fontsize=12, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_paper_complex3_holdout.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recreate publication figures")
    parser.add_argument("--experiment-name", type=str, default="suite_400_10iter_final", help="Benchmark experiment directory")
    args = parser.parse_args()
    
    print("\n" + "=" * 75)
    print(f"RECREATING PUBLICATION FIGURES (BENCHMARK: {args.experiment_name})")
    print("=" * 75)
    
    # 1. Load new benchmark data
    new_data = load_new_benchmark_data(experiment_name=args.experiment_name)
    
    # 2. Generate New-Runs-Only Paper Figures (Exact original paper layout & grayscale styling)
    print("\nGenerating Figures from NEW RUNS ONLY (p-BART, r-BART, CPHM/KM)...")
    recreate_fig1_simple_new_only(new_data)
    recreate_fig2_complex12_new_only(new_data)
    recreate_fig3_complex3_new_only(new_data)
    
    # 3. Load historical original runs for 4-model comparison (Option A)
    print("\nGenerating 4-Model Comparison Figures (Option A)...")
    og_c12 = load_original_complex12()
    og_c3 = load_original_complex3()
    recreate_fig1_simple(new_data)
    recreate_fig2_complex12(og_c12, new_data)
    recreate_fig3_complex3(og_c3, new_data)
    
    print("\nSUCCESS: All publication figures generated cleanly in figures/!")
    
    # Copy to artifact directory for UI display
    art_dir = Path("/Users/jacobtiegs/.gemini/antigravity/brain/70b28f79-9f31-4ee7-92bf-f24b5be801e4")
    if art_dir.exists():
        import shutil
        for p in FIGURES_DIR.glob("*.png"):
            shutil.copy2(p, art_dir / p.name)
        print(f"Copied figures to: {art_dir}")
        
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
