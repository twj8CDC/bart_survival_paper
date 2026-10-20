#!/usr/bin/env python3
"""
Consolidates Multi-N Validation Benchmark results into a publication-ready Excel workbook:
multi_N_100iter_benchmark_results.xlsx

Tabs created:
1. 'Multi-N Benchmark (p-BART)': Comprehensive 15-tier benchmark table
2. 'Complex 3 Multi-Model Comparison': Head-to-head comparison of p-BART, r-BART, and Cox PH across N in [400, 1000, 4000]
3. 'Metric Dictionary': Formal definitions, formulas, and targets
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
SUITE_DIR = RESULTS_DIR / "suite_multi_N_100iter_standard"
OUT_XLSX = SUITE_DIR / "multi_N_100iter_benchmark_results.xlsx"


def load_json(p: Path):
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def safe_mean(arr, decimals=4):
    if arr is None or len(arr) == 0:
        return None
    a = np.asarray(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    return round(float(np.mean(a)), decimals)


def safe_std(arr, decimals=4):
    if arr is None or len(arr) == 0:
        return None
    a = np.asarray(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    return round(float(np.std(a)), decimals)


def style_table(ws, header_rows=2):
    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    
    subgroup_fill = PatternFill(start_color="E8EEF5", end_color="E8EEF5", fill_type="solid")
    subgroup_font = Font(name="Calibri", size=10, bold=True, color="1B365D")
    
    thin_side = Side(border_style="thin", color="D0D5DD")
    thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    thick_bottom = Border(bottom=Side(border_style="medium", color="1B365D"), left=thin_side, right=thin_side, top=thin_side)

    for col in range(1, ws.max_column + 1):
        c1 = ws.cell(row=1, column=col)
        c1.fill = header_fill
        c1.font = header_font
        c1.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c1.border = thin_border
        
        if header_rows > 1:
            c2 = ws.cell(row=2, column=col)
            c2.fill = subgroup_fill
            c2.font = subgroup_font
            c2.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c2.border = thick_bottom

    for row in range(header_rows + 1, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=9.5)
            val = cell.value
            if col in (1, 2):
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif isinstance(val, (int, float)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
                if isinstance(val, float):
                    if abs(val) < 0.001 and val != 0:
                        cell.number_format = "0.00000"
                    elif "coverage" in str(ws.cell(row=header_rows, column=col).value).lower() or "%" in str(ws.cell(row=header_rows, column=col).value):
                        cell.number_format = "0.0%" if val <= 1.0 else "0.00"
                    else:
                        cell.number_format = "0.0000"
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)


def generate_workbook():
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default sheet

    # -------------------------------------------------------------
    # TAB 1: Multi-N Benchmark Summary (All 15 tiers, p-BART focus)
    # -------------------------------------------------------------
    ws1 = wb.create_sheet(title="Multi-N Benchmark (p-BART)")
    
    title_banner = "BART-SURVIVAL MULTI-N VALIDATION BENCHMARK (100 ITERS STANDARD, 10 ITERS UPDATED COMPLEX 3)"
    ws1.append([title_banner])
    ws1.append(["Standard Scientific Preset (50 trees, 1,000 draws, 250 tune, thin=4 -> 1,000 final draws) evaluated across all 15 sample size tiers (N=200, 400, 1000, 4000)."])
    
    headers1 = [
        "Scenario", "N", "Iterations", "Fit Time (s)", "Total Time (s)",
        "a-RMSE (Train)", "a-RMSE SD", "Signed Bias", "95% Coverage",
        "Interval Length", "Test a-RMSE (Holdout)", "R-hat Mean", "R-hat Max",
        "ESS Bulk Mean", "Causal Accuracy"
    ]
    ws1.append(headers1)

    summary_rows = []
    for folder in sorted(SUITE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        rep_file = folder / "replicate_metrics.json"
        if not rep_file.exists():
            continue
        reps = load_json(rep_file)
        pb_reps = [r for r in reps if r.get("model") == "pbart"]
        if not pb_reps:
            continue
            
        scen_id = pb_reps[0]["scenario_id"]
        n = pb_reps[0]["n"]
        df_pb = pd.DataFrame(pb_reps)
        
        c_file = folder / "causal_effects.json"
        causal_info = "N/A"
        if c_file.exists():
            c_data = load_json(c_file)
            c_df = pd.DataFrame([r for r in c_data if r.get("model") == "pbart"])
            if not c_df.empty and "variable" in c_df:
                contrasts = []
                for v, grp in c_df.groupby("variable"):
                    m_hr = grp["hazard_ratio_weighted_mean"].mean() if "hazard_ratio_weighted_mean" in grp else None
                    t_hr = grp["true_hr"].iloc[0] if "true_hr" in grp else None
                    cov_hr = grp["coverage_hr"].mean() if "coverage_hr" in grp else None
                    if m_hr and t_hr:
                        contrasts.append(f"{v}: HR={m_hr:.3f} (True={t_hr:.3f}, Cov={cov_hr*100:.0f}%)")
                if contrasts:
                    causal_info = " | ".join(contrasts)
                    
        row = [
            scen_id,
            n,
            len(df_pb),
            safe_mean(df_pb["fit_time_seconds"], 2),
            safe_mean(df_pb["total_time_seconds"], 2),
            safe_mean(df_pb["rmse"], 4),
            safe_std(df_pb["rmse"], 4),
            safe_mean(df_pb["bias"], 4),
            f"{safe_mean(df_pb['coverage'], 3)*100:.1f}%" if df_pb["coverage"].notna().any() else "N/A",
            safe_mean(df_pb["interval_length"], 4),
            safe_mean(df_pb["test_rmse"], 4) if "test_rmse" in df_pb and df_pb["test_rmse"].notna().any() else "N/A",
            safe_mean(df_pb["rhat_mean"], 4) if "rhat_mean" in df_pb and df_pb["rhat_mean"].notna().any() else "N/A",
            safe_mean(df_pb["rhat_max"], 4) if "rhat_max" in df_pb and df_pb["rhat_max"].notna().any() else "N/A",
            round(df_pb["ess_bulk_mean"].mean(), 0) if "ess_bulk_mean" in df_pb and df_pb["ess_bulk_mean"].notna().any() else "N/A",
            causal_info
        ]
        summary_rows.append(row)

    # Sort by scenario and N
    summary_rows.sort(key=lambda x: (x[0], x[1]))
    for r in summary_rows:
        ws1.append(r)

    # Apply styling
    ws1.merge_cells("A1:O1")
    ws1.merge_cells("A2:O2")
    ws1["A1"].font = Font(name="Calibri", size=13, bold=True, color="1B365D")
    ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws1["A2"].font = Font(name="Calibri", size=9.5, italic=True, color="555555")
    ws1["A2"].alignment = Alignment(horizontal="center", vertical="center")

    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    thin_side = Side(border_style="thin", color="D0D5DD")
    thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    
    for col in range(1, 16):
        c = ws1.cell(row=3, column=col)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin_border

    for row in range(4, ws1.max_row + 1):
        for col in range(1, 16):
            cell = ws1.cell(row=row, column=col)
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=9.5)
            if col in (1,):
                cell.alignment = Alignment(horizontal="left", vertical="center")
                cell.font = Font(name="Calibri", size=9.5, bold=True)
            elif col in (15,):
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws1.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws1.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 55)

    # -------------------------------------------------------------
    # TAB 2: Complex 3 Multi-Model Comparison (p-BART vs r-BART vs CoxPH)
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Complex 3 Multi-Model Benchmark")
    
    title_banner2 = "COMPLEX 3 BENCHMARK: p-BART vs. r-BART vs. COX PROPORTIONAL HAZARDS"
    ws2.append([title_banner2])
    ws2.append(["Evaluated on updated manuscript DGP: alpha(X) = 1 + 2*x0 + x0*x1 + 0.1*x1*x4^2, lambda(X) = 50 + 100*x0 - 100*x0*x1 + 10*x2 + 20*x3^2 + 20*x3*x4."])
    
    # 2-level grouped headers
    row_g1 = [
        "CONFIGURATION", "", "",
        "TRAINING CALIBRATION", "", "",
        "OUT-OF-SAMPLE GENERALIZATION", "", "",
        "UNCERTAINTY & INFERENCE", "",
        "COMPUTATIONAL EFFICIENCY", ""
    ]
    row_g2 = [
        "Sample Size (N)", "Model Estimator", "Iterations",
        "Train a-RMSE", "Train a-RMSE SD", "Signed Bias",
        "Test a-RMSE (Holdout)", "Test a-RMSE SD", "Test Signed Bias",
        "95% Coverage", "Credible Width",
        "Fit Time (s)", "Total Time (s)"
    ]
    ws2.append(row_g1)
    ws2.append(row_g2)
    
    ws2.merge_cells("A1:M1")
    ws2.merge_cells("A2:M2")
    ws2.merge_cells("A3:C3")
    ws2.merge_cells("D3:F3")
    ws2.merge_cells("G3:I3")
    ws2.merge_cells("J3:K3")
    ws2.merge_cells("L3:M3")
    
    ws2["A1"].font = Font(name="Calibri", size=13, bold=True, color="1B365D")
    ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws2["A2"].font = Font(name="Calibri", size=9.5, italic=True, color="555555")
    ws2["A2"].alignment = Alignment(horizontal="center", vertical="center")
    
    c3_rows = []
    for n_val in [400, 1000, 4000]:
        folder = SUITE_DIR / f"complex_3_N{n_val}_standard"
        if not folder.exists():
            continue
        rep_file = folder / "replicate_metrics.json"
        if not rep_file.exists():
            continue
        reps = load_json(rep_file)
        df_all = pd.DataFrame(reps)
        
        for m_key, m_label in [("pbart", "p-BART (2026)"), ("rbart", "r-BART (R::BART)"), ("coxph", "Cox PH (lifelines)")]:
            df_m = df_all[df_all["model"] == m_key]
            if df_m.empty:
                continue
            row = [
                n_val,
                m_label,
                len(df_m),
                safe_mean(df_m["rmse"], 4),
                safe_std(df_m["rmse"], 4),
                safe_mean(df_m["bias"], 4),
                safe_mean(df_m["test_rmse"], 4) if "test_rmse" in df_m and df_m["test_rmse"].notna().any() else "N/A",
                safe_std(df_m["test_rmse"], 4) if "test_rmse" in df_m and df_m["test_rmse"].notna().any() else "N/A",
                safe_mean(df_m["test_bias"], 4) if "test_bias" in df_m and df_m["test_bias"].notna().any() else "N/A",
                f"{safe_mean(df_m['coverage'], 3)*100:.1f}%" if df_m["coverage"].notna().any() else "N/A",
                safe_mean(df_m["interval_length"], 4) if df_m["interval_length"].notna().any() else "N/A",
                safe_mean(df_m["fit_time_seconds"], 2),
                safe_mean(df_m["total_time_seconds"], 2)
            ]
            c3_rows.append(row)

    for r in c3_rows:
        ws2.append(r)

    # Style Tab 2
    for col in range(1, 14):
        c1 = ws2.cell(row=3, column=col)
        c1.fill = header_fill
        c1.font = header_font
        c1.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c1.border = thin_border
        
        c2 = ws2.cell(row=4, column=col)
        c2.fill = PatternFill(start_color="E8EEF5", end_color="E8EEF5", fill_type="solid")
        c2.font = Font(name="Calibri", size=9.5, bold=True, color="1B365D")
        c2.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c2.border = thin_border

    for row in range(5, ws2.max_row + 1):
        for col in range(1, 14):
            cell = ws2.cell(row=row, column=col)
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=9.5)
            if col in (2,):
                cell.alignment = Alignment(horizontal="left", vertical="center")
                cell.font = Font(name="Calibri", size=9.5, bold=True)
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws2.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws2.column_dimensions[col_letter].width = max(max_len + 3, 14)

    # -------------------------------------------------------------
    # TAB 3: Metric Dictionary
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Metric Dictionary")
    ws3.append(["Metric Term", "Target Value / Ideal", "Mathematical Formulation", "Clinical / Methodological Interpretation"])
    
    dictionary_data = [
        ("Overall Survival RMSE (a-RMSE)", "0.0000 (Minimization)", "sqrt( 1/(N*K) * sum_{i,k} [S_hat(t_k|X_i) - S(t_k|X_i)]^2 )", "Root mean squared error measuring global calibration accuracy across all individuals and follow-up time intervals."),
        ("Overall Signed Calibration Bias", "0.0000 (Unbiased)", "1/(N*K) * sum_{i,k} [S_hat(t_k|X_i) - S(t_k|X_i)]", "Signed average divergence. Positive bias indicates overestimating true survival probability; negative indicates underestimation."),
        ("Empirical 95% Credible / Confidence Coverage", "0.9500 (95.0%)", "1/(N*K) * sum_{i,k} 1{ L_{0.025}(t_k|X_i) <= S(t_k|X_i) <= U_{0.975}(t_k|X_i) }", "Evaluates epistemic uncertainty quantification. Reflects whether the 95% posterior credible interval encloses true survival 95% of the time."),
        ("95% Credible Interval Width (Precision)", "Narrower is superior at nominal coverage", "1/(N*K) * sum_{i,k} [U_{0.975}(t_k|X_i) - L_{0.025}(t_k|X_i)]", "Mean length of the posterior uncertainty band. Measures statistical efficiency and epistemic certainty."),
        ("Hold-out Test RMSE & Bias", "0.0000 RMSE / 0.0000 Bias", "Evaluated on independent unseen test cohort N=400 with pre-specified interval cutpoints", "Measures generalization to unseen patients, guarding against tree overfitting on non-linear response surfaces."),
        ("Marginal Causal Hazard Ratio (G-Computation)", "Exact Analytical True HR", "exp( sum_k w_k * ln[ log(1 - lambda_k(1)) / log(1 - lambda_k(0)) ] )", "Population average causal contrast for counterfactual treatment assignment. Estimated via posterior predictive counterfactual integration."),
        ("Gelman-Rubin Convergence (R-hat)", "R-hat <= 1.02 across all chains", "Potential scale reduction factor comparing between-chain to within-chain variance across 4 parallel chains", "Confirms adequate MCMC stationarity and absence of multi-modal chain trapping."),
        ("Effective Sample Size (ESS Bulk)", "ESS >= 400", "Bulk effective sample size accounting for autocorrelation", "Guarantees sufficient independent posterior draws for reliable quantile and credible interval estimation.")
    ]
    
    for row in dictionary_data:
        ws3.append(list(row))
        
    for col in range(1, 5):
        c = ws3.cell(row=1, column=col)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin_border
        
    for row in range(2, ws3.max_row + 1):
        for col in range(1, 5):
            cell = ws3.cell(row=row, column=col)
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=9.5)
            if col == 1:
                cell.font = Font(name="Calibri", size=9.5, bold=True)
                cell.alignment = Alignment(horizontal="left", vertical="center")
            elif col == 2:
                cell.font = Font(name="Calibri", size=9.5, bold=True, color="004D40")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col == 3:
                cell.font = Font(name="Courier New", size=8.5)
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    ws3.column_dimensions["A"].width = 38
    ws3.column_dimensions["B"].width = 24
    ws3.column_dimensions["C"].width = 46
    ws3.column_dimensions["D"].width = 58

    # Save workbook
    wb.save(OUT_XLSX)
    print(f"Successfully generated: {OUT_XLSX}")
    
    # Also mirror to root
    root_copy = BASE_DIR.parent / "multi_N_100iter_benchmark_results.xlsx"
    wb.save(root_copy)
    print(f"Mirrored copy saved to: {root_copy}")


if __name__ == "__main__":
    generate_workbook()

