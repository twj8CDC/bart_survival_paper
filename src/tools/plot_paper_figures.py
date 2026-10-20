"""
CLI & Script for generating publication-quality figures and summary tables
from validation benchmark outputs.
Saves plots into `figures/` and prints formatted tables.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

def load_data(experiment_name=None):
    if experiment_name:
        suite_dir = RESULTS_DIR / experiment_name
    else:
        # Find latest directory in results
        dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and "historical" not in d.name]
        if not dirs:
            raise FileNotFoundError("No experiment runs found.")
        suite_dir = sorted(dirs, key=lambda p: p.stat().st_mtime)[-1]
        
    print(f"Loading results from: {suite_dir}")
    reps_list = []
    causal_list = []
    
    for scen_dir in sorted(suite_dir.glob("*")):
        if not scen_dir.is_dir():
            continue
        rep_file = scen_dir / "replicate_metrics.json"
        if rep_file.exists():
            with open(rep_file) as f:
                reps_list.extend(json.load(f))
        causal_file = scen_dir / "causal_effects.json"
        if causal_file.exists():
            with open(causal_file) as f:
                causal_list.extend(json.load(f))
                
    df_reps = pd.DataFrame(reps_list)
    df_causal = pd.DataFrame(causal_list)
    return df_reps, df_causal

def plot_fig1_simple(df_reps):
    simple_mask = df_reps['scenario_id'].str.startswith('simple')
    df_simple = df_reps[simple_mask].copy()
    if df_simple.empty:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    
    # 1. RMSE
    sns.barplot(
        data=df_simple, x='scenario_id', y='rmse', hue='model',
        palette={'pbart': '#1f77b4', 'km': '#2ca02c', 'rbart': '#ff7f0e'},
        ax=axes[0], errorbar='se', capsize=0.1
    )
    axes[0].set_title("Root Mean Squared Error (a-RMSE)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("a-RMSE", fontsize=10)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=25)
    
    # 2. Bias
    sns.barplot(
        data=df_simple, x='scenario_id', y='bias', hue='model',
        palette={'pbart': '#1f77b4', 'km': '#2ca02c', 'rbart': '#ff7f0e'},
        ax=axes[1], errorbar='se', capsize=0.1
    )
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title("Signed Calibration Bias", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Mean Signed Bias", fontsize=10)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=25)
    
    # 3. Coverage
    cov_df = df_simple.dropna(subset=['coverage'])
    sns.barplot(
        data=cov_df, x='scenario_id', y='coverage', hue='model',
        palette={'pbart': '#1f77b4', 'km': '#2ca02c', 'rbart': '#ff7f0e'},
        ax=axes[2], errorbar='se', capsize=0.1
    )
    axes[2].axhline(0.95, color='crimson', linestyle=':', linewidth=1.2, label='Nominal 95%')
    axes[2].set_title("95% Credible / Confidence Coverage", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Empirical Coverage", fontsize=10)
    axes[2].set_ylim(0.70, 1.02)
    axes[2].set_xlabel("")
    axes[2].tick_params(axis='x', rotation=25)
    axes[2].legend(loc='lower right')
    
    plt.tight_layout()
    out = FIGURES_DIR / "fig1_simple_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_fig2_complex(df_reps):
    complex_mask = df_reps['scenario_id'].isin(['complex_1', 'complex_2'])
    df_complex = df_reps[complex_mask].copy()
    if df_complex.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    sns.boxplot(
        data=df_complex, x='scenario_id', y='rmse', hue='model',
        palette={'pbart': '#1f77b4', 'coxph': '#d62728', 'rbart': '#ff7f0e'},
        ax=axes[0], width=0.45
    )
    axes[0].set_title("Calibration Error (PH vs. Non-PH)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("a-RMSE", fontsize=10)
    axes[0].set_xticklabels(["Complex 1 (PH)", "Complex 2 (Non-PH)"])
    axes[0].set_xlabel("")
    
    sns.boxplot(
        data=df_complex, x='scenario_id', y='bias', hue='model',
        palette={'pbart': '#1f77b4', 'coxph': '#d62728', 'rbart': '#ff7f0e'},
        ax=axes[1], width=0.45
    )
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title("Signed Calibration Bias", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Signed Bias", fontsize=10)
    axes[1].set_xticklabels(["Complex 1 (PH)", "Complex 2 (Non-PH)"])
    axes[1].set_xlabel("")
    
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_complex_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_fig3_complex3(df_reps):
    c3_mask = df_reps['scenario_id'] == 'complex_3'
    df_c3 = df_reps[c3_mask].copy()
    if df_c3.empty or 'test_rmse' not in df_c3.columns:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    
    sns.barplot(
        data=df_c3, x='model', y='rmse',
        palette={'pbart': '#1f77b4', 'coxph': '#d62728'},
        ax=axes[0], errorbar='se', capsize=0.1
    )
    axes[0].set_title("Train Cohort Calibration a-RMSE", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("a-RMSE", fontsize=10)
    
    sns.barplot(
        data=df_c3.dropna(subset=['test_rmse']), x='model', y='test_rmse',
        palette={'pbart': '#1f77b4', 'coxph': '#d62728'},
        ax=axes[1], errorbar='se', capsize=0.1
    )
    axes[1].set_title("Hold-out Test Cohort a-RMSE", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Hold-out a-RMSE", fontsize=10)
    
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_complex3_nonlinear_generalization.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_fig4_causal(df_causal):
    if df_causal.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    
    c1_causal = df_causal[df_causal['scenario_id'] == 'complex_1'].copy()
    if not c1_causal.empty:
        sns.barplot(
            data=c1_causal, x='variable', y='hazard_ratio_weighted_mean', hue='model',
            palette={'pbart': '#1f77b4', 'coxph': '#d62728'},
            ax=axes[0], errorbar='se', capsize=0.1
        )
        for i, var in enumerate(['x0', 'x6']):
            t_row = c1_causal[c1_causal['variable'] == var]['true_hr'].dropna()
            if len(t_row) > 0:
                t_val = t_row.iloc[0]
                axes[0].plot([i - 0.35, i + 0.35], [t_val, t_val], color='gold', linewidth=2.5, linestyle='--', label='True HR' if i == 0 else "")
        axes[0].set_title("Complex 1: Causal HR Estimation (x0 vs x6)", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("Event-Weighted Summary HR", fontsize=10)
        axes[0].legend()
        
    c3_causal = df_causal[df_causal['scenario_id'] == 'complex_3'].copy()
    if not c3_causal.empty:
        sns.barplot(
            data=c3_causal, x='variable', y='hazard_ratio_weighted_mean', hue='model',
            palette={'pbart': '#1f77b4', 'coxph': '#d62728'},
            ax=axes[1], errorbar='se', capsize=0.1
        )
        for i, var in enumerate(['x0', 'x2', 'x4']):
            t_row = c3_causal[c3_causal['variable'] == var]['true_hr'].dropna()
            if len(t_row) > 0:
                t_val = t_row.iloc[0]
                axes[1].plot([i - 0.35, i + 0.35], [t_val, t_val], color='gold', linewidth=2.5, linestyle='--', label='True HR' if i == 0 else "")
        axes[1].set_title("Complex 3: Causal Contrasts (U-Shape & Linear)", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Event-Weighted Summary HR", fontsize=10)
        axes[1].legend()
        
    plt.tight_layout()
    out = FIGURES_DIR / "fig4_causal_effect_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_fig5_runtime(df_reps):
    time_df = df_reps.dropna(subset=['total_time_seconds']).copy()
    if time_df.empty:
        return
        
    plt.figure(figsize=(9, 4.2), dpi=300)
    sns.barplot(
        data=time_df, x='scenario_id', y='total_time_seconds', hue='model',
        palette={'pbart': '#1f77b4', 'coxph': '#d62728', 'km': '#2ca02c', 'rbart': '#ff7f0e'},
        errorbar='se', capsize=0.1
    )
    plt.title("Wall-Clock Execution Time per Replicate", fontsize=11, fontweight='bold')
    plt.ylabel("Wall-Clock Time (s, Log Scale)", fontsize=10)
    plt.xlabel("")
    plt.yscale('log')
    plt.xticks(rotation=25)
    plt.tight_layout()
    out = FIGURES_DIR / "fig5_runtime_scaling.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

def print_summary_table(df_reps):
    summary_table = df_reps.groupby(['scenario_id', 'model']).agg(
        RMSE_mean=('rmse', 'mean'),
        RMSE_std=('rmse', 'std'),
        Bias_mean=('bias', 'mean'),
        Coverage=('coverage', 'mean'),
        Interval_Length=('interval_length', 'mean'),
        Duration_s=('total_time_seconds', 'mean')
    ).reset_index()
    print("\n" + "="*80)
    print("MANUSCRIPT REVISION SUMMARY TABLE")
    print("="*80)
    print(summary_table.round(4).to_string(index=False))
    print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="suite_5iter_fast")
    args = parser.parse_args()
    
    df_reps, df_causal = load_data(args.experiment_name)
    plot_fig1_simple(df_reps)
    plot_fig2_complex(df_reps)
    plot_fig3_complex3(df_reps)
    plot_fig4_causal(df_causal)
    plot_fig5_runtime(df_reps)
    print_summary_table(df_reps)

if __name__ == "__main__":
    main()

