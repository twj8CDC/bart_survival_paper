#!/usr/bin/env python3
"""
Master Benchmark CLI Runner for BART Survival Validation Suite.
Faithfully executes manuscript scenarios, benchmarks p-BART vs. CoxPH/KM/r-BART,
extracts survival curve calibration and causal estimands, and logs structured outputs.
"""

import sys
import os

# Cap underlying BLAS/OpenMP thread pools to 1 per worker process to prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
import gc
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure src is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config.scenarios import SCENARIOS
from config.models import MODEL_CONFIGS
from validation.dgp import generate_scenario_data
from validation.metrics import compute_interval_metrics
from validation.runners.pbart_runner import run_pbart
from validation.runners.cox_runner import run_coxph
from validation.runners.km_runner import run_km
from validation.runners.rbart_runner import run_rbart
from validation.tracker import BenchmarkTracker


def run_single_scenario(
    scenario_id: str,
    n: int,
    iters: int,
    preset: str,
    seed_offset: int,
    skip_r: bool,
    tracker: BenchmarkTracker
):
    scenario_cfg = SCENARIOS[scenario_id]
    model_cfg = MODEL_CONFIGS[preset]
    category = scenario_cfg.get("category", "simple")
    causal_specs = scenario_cfg.get("causal_contrasts", [])
    
    print("\n" + "=" * 75)
    print(f"RUNNING SCENARIO: {scenario_cfg['name']} | N={n} | Preset={preset} | Iterations={iters}")
    print("=" * 75)
    
    tracker.log_run_start(scenario_id, n, preset, scenario_cfg, model_cfg)
    
    replicate_records, causal_records = tracker.load_replicates(scenario_id, n, preset)
    completed_seeds = set(r["seed"] for r in replicate_records if r.get("model") == "pbart")
    if completed_seeds:
        print(f"Resuming {scenario_id}: Found {len(completed_seeds)} already completed replicates.")
    
    start_seed = n * seed_offset
    
    for rep_idx in range(iters):
        current_seed = start_seed + rep_idx
        if current_seed in completed_seeds:
            continue
        print(f"\n--- [Replicate {rep_idx + 1}/{iters}] Seed {current_seed} ---")
        
        # 1. Generate Training Data
        train_data = generate_scenario_data(scenario_cfg, n=n, seed=current_seed)
        
        # 2. Generate Independent Test Data (if scenario evaluates test cohort, e.g. Complex 3)
        test_data = None
        if "test_sample_size" in scenario_cfg:
            test_n = scenario_cfg["test_sample_size"]
            test_data = generate_scenario_data(scenario_cfg, n=test_n, seed=current_seed + 10000)
            
        true_sv_train = train_data["sv_true_eval"]
        
        # -------------------------------------------------------------
        # Model 1: p-BART (SurvBART)
        # -------------------------------------------------------------
        print("  -> Running p-BART (SurvBART)...", end=" ", flush=True)
        pbart_res = run_pbart(
            train_data=train_data,
            model_params=model_cfg,
            causal_specs=causal_specs,
            test_data=test_data,
            progressbar=False,
            random_seed=current_seed
        )
        pbart_metrics = compute_interval_metrics(
            true_sv=true_sv_train,
            pred_mean=pbart_res["sv_mean"],
            pred_ci_lower=pbart_res["ci_lower"],
            pred_ci_upper=pbart_res["ci_upper"],
            time_points=train_data["eval_times"]
        )
        print(f"DONE in {pbart_res['timings']['total_seconds']:.2f}s | Train RMSE: {pbart_metrics['overall_rmse']:.4f}")
        
        pb_rec = {
            "scenario_id": scenario_id,
            "n": n,
            "preset": preset,
            "seed": current_seed,
            "model": "pbart",
            "rmse": pbart_metrics["overall_rmse"],
            "bias": pbart_metrics["overall_bias"],
            "coverage": pbart_metrics.get("overall_coverage"),
            "interval_length": pbart_metrics.get("overall_interval_length"),
            "rmse_per_interval": pbart_metrics["rmse_per_interval"],
            "bias_per_interval": pbart_metrics["bias_per_interval"],
            "fit_time_seconds": pbart_res["timings"]["fit_seconds"],
            "total_time_seconds": pbart_res["timings"]["total_seconds"],
            "rhat_mean": pbart_res.get("diagnostics", {}).get("rhat_mean"),
            "rhat_max": pbart_res.get("diagnostics", {}).get("rhat_max"),
            "ess_bulk": pbart_res.get("diagnostics", {}).get("ess_mean"),
            "ess_min": pbart_res.get("diagnostics", {}).get("ess_min"),
        }
        
        # Evaluate test metrics if available
        if test_data and pbart_res.get("test_results"):
            test_metrics = compute_interval_metrics(
                true_sv=test_data["sv_true_eval"],
                pred_mean=pbart_res["test_results"]["sv_mean"],
                pred_ci_lower=pbart_res["test_results"]["ci_lower"],
                pred_ci_upper=pbart_res["test_results"]["ci_upper"]
            )
            pb_rec["test_rmse"] = test_metrics["overall_rmse"]
            pb_rec["test_bias"] = test_metrics["overall_bias"]
            pb_rec["test_coverage"] = test_metrics.get("overall_coverage")
            pb_rec["test_rmse_per_interval"] = test_metrics["rmse_per_interval"]
            pb_rec["test_bias_per_interval"] = test_metrics["bias_per_interval"]
            print(f"     [Test Cohort] RMSE: {test_metrics['overall_rmse']:.4f}")
            
        strat = train_data.get("stratification", {})
        has_strat = ("idx_top25" in strat and "true_sv_top25" in strat)
        
        if has_strat and pbart_res.get("sv_draws") is not None:
            sv_draws = pbart_res["sv_draws"]
            d_top = sv_draws[:, strat["idx_top25"], :].mean(axis=1)
            d_bot = sv_draws[:, strat["idx_bottom25"], :].mean(axis=1)
            top_m = d_top.mean(axis=0)
            bot_m = d_bot.mean(axis=0)
            top_ci = np.quantile(d_top, [0.025, 0.975], axis=0)
            bot_ci = np.quantile(d_bot, [0.025, 0.975], axis=0)
            
            pb_rec["rmse_top25"] = float(np.sqrt(np.mean((top_m - strat["true_sv_top25"])**2)))
            pb_rec["bias_top25"] = float(np.mean(top_m - strat["true_sv_top25"]))
            pb_rec["coverage_top25"] = float(np.mean((top_ci[0] <= strat["true_sv_top25"]) & (strat["true_sv_top25"] <= top_ci[1])))
            
            pb_rec["rmse_bot25"] = float(np.sqrt(np.mean((bot_m - strat["true_sv_bot25"])**2)))
            pb_rec["bias_bot25"] = float(np.mean(bot_m - strat["true_sv_bot25"]))
            pb_rec["coverage_bot25"] = float(np.mean((bot_ci[0] <= strat["true_sv_bot25"]) & (strat["true_sv_bot25"] <= bot_ci[1])))
            
            ci_min = [pbart_res["ci_lower"][strat["idx_min"]], pbart_res["ci_upper"][strat["idx_min"]]]
            pb_rec["coverage_min"] = float(np.mean((ci_min[0] <= strat["true_sv_min"]) & (strat["true_sv_min"] <= ci_min[1])))
            ci_max = [pbart_res["ci_lower"][strat["idx_max"]], pbart_res["ci_upper"][strat["idx_max"]]]
            pb_rec["coverage_max"] = float(np.mean((ci_max[0] <= strat["true_sv_max"]) & (strat["true_sv_max"] <= ci_max[1])))
            
        replicate_records.append(pb_rec)
        
        # Record Causal Effects (p-BART)
        true_causal_data = train_data.get("true_causal", {})
        for var_name, eff_dict in pbart_res.get("causal_effects", {}).items():
            hr_curve = eff_dict.get("hazard_ratio_curve_mean")
            if isinstance(hr_curve, np.ndarray):
                hr_curve = hr_curve.tolist()
            drr_curve = eff_dict.get("discrete_risk_ratio_mean")
            if isinstance(drr_curve, np.ndarray):
                drr_curve = drr_curve.tolist()
            rd_curve = eff_dict.get("risk_difference_mean")
            if isinstance(rd_curve, np.ndarray):
                rd_curve = rd_curve.tolist()
                
            tc = true_causal_data.get(var_name, {})
            true_hr_val = tc.get("hazard_ratio_weighted_mean", eff_dict.get("true_hr"))
            true_rd_val = tc.get("true_rd_at_t50")
            hr_ci_l = eff_dict.get("hazard_ratio_weighted_ci", [None, None])[0] if eff_dict.get("hazard_ratio_weighted_ci") else None
            hr_ci_u = eff_dict.get("hazard_ratio_weighted_ci", [None, None])[1] if eff_dict.get("hazard_ratio_weighted_ci") else None
            hr_w = eff_dict.get("hazard_ratio_ci_width")
            if hr_w is None and hr_ci_l is not None and hr_ci_u is not None:
                hr_w = float(hr_ci_u - hr_ci_l)
            cov_hr = 1 if (hr_ci_l is not None and hr_ci_u is not None and true_hr_val is not None and hr_ci_l <= true_hr_val <= hr_ci_u) else 0

            rd_ci_l = eff_dict.get("risk_difference_t50_ci_lower")
            rd_ci_u = eff_dict.get("risk_difference_t50_ci_upper")
            rd_w = eff_dict.get("risk_difference_t50_ci_width")
            if rd_w is None and rd_ci_l is not None and rd_ci_u is not None:
                rd_w = float(rd_ci_u - rd_ci_l)
            cov_rd = 1 if (rd_ci_l is not None and rd_ci_u is not None and true_rd_val is not None and rd_ci_l <= true_rd_val <= rd_ci_u) else 0
                
            causal_records.append({
                "scenario_id": scenario_id,
                "n": n,
                "preset": preset,
                "seed": current_seed,
                "model": "pbart",
                "variable": var_name,
                "hazard_ratio_weighted_mean": eff_dict.get("hazard_ratio_weighted_mean"),
                "hazard_ratio_ci_lower": hr_ci_l,
                "hazard_ratio_ci_upper": hr_ci_u,
                "hazard_ratio_ci_width": hr_w,
                "coverage_hr": cov_hr,
                "hazard_ratio_unweighted_mean": eff_dict.get("hazard_ratio_unweighted_mean"),
                "risk_difference_t50_mean": eff_dict.get("risk_difference_t50_mean"),
                "risk_difference_t50_ci_lower": rd_ci_l,
                "risk_difference_t50_ci_upper": rd_ci_u,
                "risk_difference_t50_ci_width": rd_w,
                "coverage_rd_t50": cov_rd,
                "true_rd_at_t50": true_rd_val,
                "t50_time": eff_dict.get("t50_time"),
                "horizon_risk_difference": eff_dict.get("risk_difference_horizon_mean"),
                "hazard_ratio_curve": hr_curve,
                "discrete_risk_ratio_curve": drr_curve,
                "risk_difference_curve": rd_curve,
                "true_hr": true_hr_val,
                "true_hr_unweighted": tc.get("hazard_ratio_unweighted_mean"),
                "true_horizon_risk_difference": tc.get("horizon_risk_difference"),
                "true_hazard_ratio_curve": tc.get("hazard_ratio_curve"),
                "true_risk_difference_curve": tc.get("risk_difference_curve"),
                "true_discrete_risk_ratio_curve": tc.get("discrete_risk_ratio_curve"),
            })
            
        # -------------------------------------------------------------
        # Model 2: Cox Proportional Hazards (for Complex & Causal scenarios)
        # -------------------------------------------------------------
        if category in ["complex_binary", "complex_continuous"] or len(causal_specs) > 0:
            print("  -> Running Cox Proportional Hazards...", end=" ", flush=True)
            cox_res = run_coxph(
                train_data=train_data,
                causal_specs=causal_specs,
                test_data=test_data
            )
            cox_metrics = compute_interval_metrics(
                true_sv=true_sv_train,
                pred_mean=cox_res["sv_mean"]
            )
            print(f"DONE in {cox_res['timings']['total_seconds']:.2f}s | Train RMSE: {cox_metrics['overall_rmse']:.4f}")
            
            cox_rec = {
                "scenario_id": scenario_id,
                "n": n,
                "preset": preset,
                "seed": current_seed,
                "model": "coxph",
                "rmse": cox_metrics["overall_rmse"],
                "bias": cox_metrics["overall_bias"],
                "coverage": cox_metrics.get("overall_coverage"),
                "interval_length": cox_metrics.get("overall_interval_length"),
                "rmse_per_interval": cox_metrics["rmse_per_interval"],
                "bias_per_interval": cox_metrics["bias_per_interval"],
                "fit_time_seconds": cox_res["timings"]["fit_seconds"],
                "total_time_seconds": cox_res["timings"]["total_seconds"],
            }
            if has_strat:
                c_top_m = cox_res["sv_mean"][strat["idx_top25"], :].mean(axis=0)
                c_bot_m = cox_res["sv_mean"][strat["idx_bottom25"], :].mean(axis=0)
                cox_rec["rmse_top25"] = float(np.sqrt(np.mean((c_top_m - strat["true_sv_top25"])**2)))
                cox_rec["bias_top25"] = float(np.mean(c_top_m - strat["true_sv_top25"]))
                cox_rec["rmse_bot25"] = float(np.sqrt(np.mean((c_bot_m - strat["true_sv_bot25"])**2)))
                cox_rec["bias_bot25"] = float(np.mean(c_bot_m - strat["true_sv_bot25"]))
                
            if test_data and cox_res.get("test_results"):
                c_test_mets = compute_interval_metrics(
                    true_sv=test_data["sv_true_eval"],
                    pred_mean=cox_res["test_results"]["sv_mean"]
                )
                cox_rec["test_rmse"] = c_test_mets["overall_rmse"]
                cox_rec["test_bias"] = c_test_mets["overall_bias"]
                cox_rec["test_rmse_per_interval"] = c_test_mets["rmse_per_interval"]
                cox_rec["test_bias_per_interval"] = c_test_mets["bias_per_interval"]
                print(f"     [Test Cohort] RMSE: {c_test_mets['overall_rmse']:.4f}")
                
            replicate_records.append(cox_rec)
            
            for var_name, eff_dict in cox_res.get("causal_effects", {}).items():
                tc = true_causal_data.get(var_name, {})
                true_hr_val = tc.get("hazard_ratio_weighted_mean", eff_dict.get("true_hr"))
                true_rd_val = tc.get("true_rd_at_t50")
                hr_ci_l = eff_dict.get("hazard_ratio_weighted_ci", [None, None])[0] if eff_dict.get("hazard_ratio_weighted_ci") else None
                hr_ci_u = eff_dict.get("hazard_ratio_weighted_ci", [None, None])[1] if eff_dict.get("hazard_ratio_weighted_ci") else None
                hr_w = eff_dict.get("hazard_ratio_ci_width")
                if hr_w is None and hr_ci_l is not None and hr_ci_u is not None:
                    hr_w = float(hr_ci_u - hr_ci_l)
                cov_hr = 1 if (hr_ci_l is not None and hr_ci_u is not None and true_hr_val is not None and hr_ci_l <= true_hr_val <= hr_ci_u) else 0

                rd_ci_l = eff_dict.get("risk_difference_t50_ci_lower")
                rd_ci_u = eff_dict.get("risk_difference_t50_ci_upper")
                rd_w = eff_dict.get("risk_difference_t50_ci_width")
                if rd_w is None and rd_ci_l is not None and rd_ci_u is not None:
                    rd_w = float(rd_ci_u - rd_ci_l)
                cov_rd = 1 if (rd_ci_l is not None and rd_ci_u is not None and true_rd_val is not None and rd_ci_l <= true_rd_val <= rd_ci_u) else 0
                
                causal_records.append({
                    "scenario_id": scenario_id,
                    "n": n,
                    "preset": preset,
                    "seed": current_seed,
                    "model": "coxph",
                    "variable": var_name,
                    "hazard_ratio_weighted_mean": eff_dict.get("hazard_ratio_weighted_mean"),
                    "hazard_ratio_ci_lower": hr_ci_l,
                    "hazard_ratio_ci_upper": hr_ci_u,
                    "hazard_ratio_ci_width": hr_w,
                    "coverage_hr": cov_hr,
                    "risk_difference_t50_mean": eff_dict.get("risk_difference_t50_mean"),
                    "risk_difference_t50_ci_lower": rd_ci_l,
                    "risk_difference_t50_ci_upper": rd_ci_u,
                    "risk_difference_t50_ci_width": rd_w,
                    "coverage_rd_t50": cov_rd,
                    "true_rd_at_t50": true_rd_val,
                    "t50_time": eff_dict.get("t50_time"),
                    "true_hr": true_hr_val,
                    "true_hr_unweighted": tc.get("hazard_ratio_unweighted_mean"),
                    "true_horizon_risk_difference": tc.get("horizon_risk_difference"),
                })
                
        # -------------------------------------------------------------
        # Model 3: Kaplan-Meier (for Simple 1 & 2)
        # -------------------------------------------------------------
        if category == "simple":
            print("  -> Running Kaplan-Meier...", end=" ", flush=True)
            is_two_sample = ("2_1" in scenario_id or "2_2" in scenario_id)
            km_res = run_km(train_data, is_two_sample=is_two_sample)
            km_metrics = compute_interval_metrics(
                true_sv=true_sv_train,
                pred_mean=km_res["sv_mean"],
                pred_ci_lower=km_res["ci_lower"],
                pred_ci_upper=km_res["ci_upper"]
            )
            print(f"DONE in {km_res['timings']['total_seconds']:.2f}s | Train RMSE: {km_metrics['overall_rmse']:.4f}")
            
            km_rec = {
                "scenario_id": scenario_id,
                "n": n,
                "preset": preset,
                "seed": current_seed,
                "model": "km",
                "rmse": km_metrics["overall_rmse"],
                "bias": km_metrics["overall_bias"],
                "coverage": km_metrics.get("overall_coverage"),
                "interval_length": km_metrics.get("overall_interval_length"),
                "rmse_per_interval": km_metrics.get("rmse_per_interval"),
                "bias_per_interval": km_metrics.get("bias_per_interval"),
                "total_time_seconds": km_res["timings"]["total_seconds"],
            }
            if has_strat:
                k_top_m = km_res["sv_mean"][strat["idx_top25"], :].mean(axis=0)
                k_bot_m = km_res["sv_mean"][strat["idx_bottom25"], :].mean(axis=0)
                km_rec["rmse_top25"] = float(np.sqrt(np.mean((k_top_m - strat["true_sv_top25"])**2)))
                km_rec["bias_top25"] = float(np.mean(k_top_m - strat["true_sv_top25"]))
                km_rec["rmse_bot25"] = float(np.sqrt(np.mean((k_bot_m - strat["true_sv_bot25"])**2)))
                km_rec["bias_bot25"] = float(np.mean(k_bot_m - strat["true_sv_bot25"]))
            replicate_records.append(km_rec)
            
        # -------------------------------------------------------------
        # Model 4: r-BART (BART package in R)
        # -------------------------------------------------------------
        if not skip_r:
            print("  -> Running r-BART (R::BART)...", end=" ", flush=True)
            rbart_res = run_rbart(
                train_data=train_data,
                skip_r=False,
                scenario_id=scenario_id,
                test_data=test_data,
                causal_specs=causal_specs
            )
            if not rbart_res.get("skipped"):
                rb_metrics = compute_interval_metrics(
                    true_sv=true_sv_train,
                    pred_mean=rbart_res["sv_mean"],
                    pred_ci_lower=rbart_res["ci_lower"],
                    pred_ci_upper=rbart_res["ci_upper"]
                )
                print(f"DONE in {rbart_res['timings']['total_seconds']:.2f}s | Train RMSE: {rb_metrics['overall_rmse']:.4f}")
                rb_rec = {
                    "scenario_id": scenario_id,
                    "n": n,
                    "preset": preset,
                    "seed": current_seed,
                    "model": "rbart",
                    "rmse": rb_metrics["overall_rmse"],
                    "bias": rb_metrics["overall_bias"],
                    "coverage": rb_metrics.get("overall_coverage"),
                    "interval_length": rb_metrics.get("overall_interval_length"),
                    "rmse_per_interval": rb_metrics.get("rmse_per_interval"),
                    "bias_per_interval": rb_metrics.get("bias_per_interval"),
                    "fit_time_seconds": rbart_res["timings"].get("fit_seconds"),
                    "total_time_seconds": rbart_res["timings"]["total_seconds"],
                    "rhat_mean": rbart_res.get("diagnostics", {}).get("rhat_mean"),
                    "rhat_max": rbart_res.get("diagnostics", {}).get("rhat_max"),
                    "ess_bulk": rbart_res.get("diagnostics", {}).get("ess_mean"),
                    "ess_min": rbart_res.get("diagnostics", {}).get("ess_min"),
                }
                if has_strat and rbart_res.get("stratification_results"):
                    st = rbart_res["stratification_results"]
                    r_top_m = st["top25_mean"]
                    r_top_ci = [st["top25_ci_lower"], st["top25_ci_upper"]]
                    r_bot_m = st["bot25_mean"]
                    r_bot_ci = [st["bot25_ci_lower"], st["bot25_ci_upper"]]
                    
                    rb_rec["rmse_top25"] = float(np.sqrt(np.mean((r_top_m - strat["true_sv_top25"])**2)))
                    rb_rec["bias_top25"] = float(np.mean(r_top_m - strat["true_sv_top25"]))
                    rb_rec["coverage_top25"] = float(np.mean((r_top_ci[0] <= strat["true_sv_top25"]) & (strat["true_sv_top25"] <= r_top_ci[1])))
                    
                    rb_rec["rmse_bot25"] = float(np.sqrt(np.mean((r_bot_m - strat["true_sv_bot25"])**2)))
                    rb_rec["bias_bot25"] = float(np.mean(r_bot_m - strat["true_sv_bot25"]))
                    rb_rec["coverage_bot25"] = float(np.mean((r_bot_ci[0] <= strat["true_sv_bot25"]) & (strat["true_sv_bot25"] <= r_bot_ci[1])))
                    
                    r_ci_min = [rbart_res["ci_lower"][strat["idx_min"]], rbart_res["ci_upper"][strat["idx_min"]]]
                    rb_rec["coverage_min"] = float(np.mean((r_ci_min[0] <= strat["true_sv_min"]) & (strat["true_sv_min"] <= r_ci_min[1])))
                    r_ci_max = [rbart_res["ci_lower"][strat["idx_max"]], rbart_res["ci_upper"][strat["idx_max"]]]
                    rb_rec["coverage_max"] = float(np.mean((r_ci_max[0] <= strat["true_sv_max"]) & (strat["true_sv_max"] <= r_ci_max[1])))
                    
                if test_data and rbart_res.get("test_results"):
                    rb_test_mets = compute_interval_metrics(
                        true_sv=test_data["sv_true_eval"],
                        pred_mean=rbart_res["test_results"]["sv_mean"],
                        pred_ci_lower=rbart_res["test_results"].get("ci_lower"),
                        pred_ci_upper=rbart_res["test_results"].get("ci_upper")
                    )
                    rb_rec["test_rmse"] = rb_test_mets["overall_rmse"]
                    rb_rec["test_bias"] = rb_test_mets["overall_bias"]
                    rb_rec["test_coverage"] = rb_test_mets.get("overall_coverage")
                    rb_rec["test_rmse_per_interval"] = rb_test_mets.get("rmse_per_interval")
                    rb_rec["test_bias_per_interval"] = rb_test_mets.get("bias_per_interval")
                replicate_records.append(rb_rec)
                
                # Record Causal Effects (r-BART)
                for var_name, eff_dict in rbart_res.get("causal_effects", {}).items():
                    tc = true_causal_data.get(var_name, {})
                    true_hr_val = tc.get("hazard_ratio_weighted_mean")
                    true_rd_val = tc.get("true_rd_at_t50")
                    hr_ci_l = eff_dict.get("hazard_ratio_weighted_ci", [None, None])[0] if eff_dict.get("hazard_ratio_weighted_ci") else None
                    hr_ci_u = eff_dict.get("hazard_ratio_weighted_ci", [None, None])[1] if eff_dict.get("hazard_ratio_weighted_ci") else None
                    hr_w = eff_dict.get("hazard_ratio_ci_width")
                    if hr_w is None and hr_ci_l is not None and hr_ci_u is not None:
                        hr_w = float(hr_ci_u - hr_ci_l)
                    cov_hr = 1 if (hr_ci_l is not None and hr_ci_u is not None and true_hr_val is not None and hr_ci_l <= true_hr_val <= hr_ci_u) else 0

                    rd_ci_l = eff_dict.get("risk_difference_t50_ci_lower")
                    rd_ci_u = eff_dict.get("risk_difference_t50_ci_upper")
                    rd_w = eff_dict.get("risk_difference_t50_ci_width")
                    if rd_w is None and rd_ci_l is not None and rd_ci_u is not None:
                        rd_w = float(rd_ci_u - rd_ci_l)
                    cov_rd = 1 if (rd_ci_l is not None and rd_ci_u is not None and true_rd_val is not None and rd_ci_l <= true_rd_val <= rd_ci_u) else 0
                    
                    causal_records.append({
                        "scenario_id": scenario_id,
                        "n": n,
                        "preset": preset,
                        "seed": current_seed,
                        "model": "rbart",
                        "variable": var_name,
                        "hazard_ratio_weighted_mean": eff_dict.get("hazard_ratio_weighted_mean"),
                        "hazard_ratio_ci_lower": hr_ci_l,
                        "hazard_ratio_ci_upper": hr_ci_u,
                        "hazard_ratio_ci_width": hr_w,
                        "coverage_hr": cov_hr,
                        "risk_difference_t50_mean": eff_dict.get("risk_difference_t50_mean"),
                        "risk_difference_t50_ci_lower": rd_ci_l,
                        "risk_difference_t50_ci_upper": rd_ci_u,
                        "risk_difference_t50_ci_width": rd_w,
                        "coverage_rd_t50": cov_rd,
                        "true_rd_at_t50": true_rd_val,
                        "t50_time": eff_dict.get("t50_time"),
                        "true_hr": true_hr_val,
                    })
            else:
                print("SKIPPED (Using Historical Lookup)")
                
        # Checkpoint after each replicate
        tracker.log_replicates(scenario_id, n, preset, replicate_records, causal_records)
        
        # Explicit garbage collection to prevent memory accumulation across MCMC runs
        del train_data
        if test_data is not None:
            del test_data
        if "pbart_res" in locals():
            del pbart_res
        if "cox_res" in locals():
            del cox_res
        if "km_res" in locals():
            del km_res
        gc.collect()
                
    # Final log and summary for scenario
    tracker.log_replicates(scenario_id, n, preset, replicate_records, causal_records)
    print(f"\nSUCCESS: Scenario {scenario_id} completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Master Validation Runner for BART Survival")
    parser.add_argument("--scenarios", nargs="+", default=["all"], help="Scenario keys (e.g. simple_1_1 complex_1) or 'all'")
    parser.add_argument("--n-samples", nargs="+", type=int, default=None, help="Sample sizes to test")
    parser.add_argument("--iters", type=int, default=5, help="Number of Monte Carlo iterations")
    parser.add_argument("--preset", choices=["fast", "standard", "production", "prod_tune250", "high_precision"], default="standard", help="Model hyperparameter preset")
    parser.add_argument("--seed-offset", type=int, default=99, help="Additive seed offset for reproducibility")
    parser.add_argument("--skip-r", action="store_true", help="Skip live Rscript BART execution (use historical baseline)")
    parser.add_argument("--experiment-name", type=str, default="validation_run_v2", help="Folder name for tracking outputs")
    
    args = parser.parse_args()
    
    if "all" in args.scenarios:
        target_scenarios = list(SCENARIOS.keys())
    else:
        target_scenarios = [s for s in args.scenarios if s in SCENARIOS]
        
    tracker = BenchmarkTracker(experiment_name=args.experiment_name)
    
    print("\n" + "#" * 75)
    print("BART-SURVIVAL MASTER VALIDATION SUITE")
    print(f"Scenarios:    {target_scenarios}")
    print(f"Preset:       {args.preset}")
    print(f"Iterations:   {args.iters}")
    print(f"Skip R-BART:  {args.skip_r}")
    print(f"Output Dir:   {tracker.base_dir}")
    print("#" * 75 + "\n")
    
    total_start_time = time.time()
    
    for scen_id in target_scenarios:
        scen_cfg = SCENARIOS[scen_id]
        if args.n_samples:
            sample_sizes = args.n_samples
        else:
            sample_sizes = [scen_cfg["default_sample_sizes"][0]] # run first sample size by default for fast verification
            
        for n in sample_sizes:
            run_single_scenario(
                scenario_id=scen_id,
                n=n,
                iters=args.iters,
                preset=args.preset,
                seed_offset=args.seed_offset,
                skip_r=args.skip_r,
                tracker=tracker
            )
            
    total_elapsed = time.time() - total_start_time
    print("\n" + "#" * 75)
    print(f"ALL BENCHMARKS FINISHED CLEANLY IN {total_elapsed / 60:.2f} MINUTES")
    print("#" * 75)

if __name__ == "__main__":
    main()
