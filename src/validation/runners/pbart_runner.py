"""
p-BART (SurvBART) Model Runner.
Fits SurvBART, predicts cumulative survival S(t), captures feature importances,
and extracts Causal Treatment Effects (HR, RR, RD curves and summaries) with runtime tracking.
"""

import time
import gc
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from bart_survival import SurvBART, format_survival_data, format_prediction_data, compute_survival_curves

def run_pbart(
    train_data: Dict[str, Any],
    model_params: Dict[str, Any],
    causal_specs: Optional[List[Dict[str, Any]]] = None,
    test_data: Optional[Dict[str, Any]] = None,
    progressbar: bool = False,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes SurvBART fitting, out-of-sample predictions, and causal estimation.
    """
    start_total_time = time.time()
    
    x_train = train_data["x_mat"]
    event_train = train_data["event_dict"]
    eval_times = train_data["eval_times"]
    
    # 1. Format pooled person-time dataset
    # We pass time_scale=None because event_train["t_event"] is already mapped to discrete interval times
    trn = format_survival_data(
        y_time=event_train["t_event"],
        y_status=event_train["status"],
        x=x_train,
        time_scale=None
    )
    
    # 2. Format out-of-sample prediction grid across discrete intervals
    post_test_train = format_prediction_data(
        y_time=eval_times,
        y_status=np.zeros_like(eval_times, dtype=int),
        x=x_train,
        time_scale=None
    )
    
    # 3. Initialize SurvBART
    m_cfg = model_params.copy()
    model = SurvBART(**m_cfg)
    
    # 4. Fit Model & Track Training Time
    start_fit_time = time.time()
    feature_names = [f"x{i}" for i in range(x_train.shape[1])]
    model.fit(
        y=trn["y"],
        X=trn["x"],
        weights=trn["w"],
        coords=trn["coord"],
        predictor_names=["t"] + feature_names,
        progressbar=progressbar,
        random_seed=random_seed
    )
    fit_duration = time.time() - start_fit_time
    
    # 5. Predict Survival on Training Cohort
    start_pred_time = time.time()
    post_preds = model.predict(post_test_train["post_x"], coords=post_test_train["coords"])
    sv_dict = compute_survival_curves(post_preds, coords=post_test_train["coords"])
    
    # sv has shape: (n_draws, n_subjects, n_times)
    sv_draws = sv_dict["sv"]
    sv_mean = sv_draws.mean(axis=0)
    sv_ci = np.quantile(sv_draws, [0.025, 0.975], axis=0) # shape: (2, n_subjects, n_times)
    pred_duration = time.time() - start_pred_time
    
    # 6. Optional Feature Importance
    try:
        feat_imp = model.feature_importance(sort=True).to_dict()
    except Exception:
        feat_imp = {}
        
    # 7. Predict on Test Cohort (if provided, e.g. for Complex 3)
    test_results = None
    if test_data is not None:
        x_test = test_data["x_mat"]
        post_test_test = format_prediction_data(
            y_time=eval_times,
            y_status=np.zeros_like(eval_times, dtype=int),
            x=x_test,
            time_scale=None
        )
        preds_test = model.predict(post_test_test["post_x"], coords=post_test_test["coords"])
        sv_dict_test = compute_survival_curves(preds_test, coords=post_test_test["coords"])
        sv_test_draws = sv_dict_test["sv"]
        test_results = {
            "sv_mean": sv_test_draws.mean(axis=0),
            "ci_lower": np.quantile(sv_test_draws, 0.025, axis=0),
            "ci_upper": np.quantile(sv_test_draws, 0.975, axis=0),
        }
        
    # 8. Causal Treatment Effects Extraction
    causal_results = {}
    if causal_specs:
        for c_spec in causal_specs:
            var_idx = c_spec["variable"]
            var_name = c_spec.get("name", f"x{var_idx}")
            c_type = c_spec.get("type", "binary")
            
            try:
                if c_type == "binary":
                    vals = c_spec.get("values", [0, 1])
                    eff = model.estimate_effect(
                        X=x_train,
                        primary_col=var_idx,
                        values=vals,
                        times=eval_times,
                        audit_support=False
                    )
                elif c_type == "contrast":
                    vals = c_spec.get("values", [0.10, 0.50])
                    eff = model.estimate_effect(
                        X=x_train,
                        primary_col=var_idx,
                        values=vals,
                        times=eval_times,
                        audit_support=False
                    )
                elif c_type == "continuous_shift":
                    delta = c_spec.get("delta", 0.20)
                    eff = model.estimate_effect(
                        X=x_train,
                        primary_col=var_idx,
                        delta=delta,
                        shift=True,
                        times=eval_times,
                        audit_support=False
                    )
                else:
                    eff = {}
                    
                t50_idx = len(eval_times) // 2
                t50_time = float(eval_times[t50_idx])
                rd_draws = eff.get("risk_difference_draws")
                if rd_draws is not None:
                    rd_at_t50 = rd_draws[:, t50_idx]
                    rd_t50_m = float(rd_at_t50.mean())
                    rd_t50_ci = np.quantile(rd_at_t50, [0.025, 0.975]).tolist()
                else:
                    rd_t50_m = None
                    rd_t50_ci = [None, None]
                    
                hr_ci = eff.get("hazard_ratio_weighted_ci")
                hr_w = float(hr_ci[1] - hr_ci[0]) if (hr_ci and hr_ci[0] is not None and hr_ci[1] is not None) else None
                rd_w = float(rd_t50_ci[1] - rd_t50_ci[0]) if (rd_t50_ci[0] is not None and rd_t50_ci[1] is not None) else None

                causal_results[var_name] = {
                    "hazard_ratio_weighted_mean": eff.get("hazard_ratio_weighted_mean"),
                    "hazard_ratio_weighted_ci": hr_ci,
                    "hazard_ratio_ci_width": hr_w,
                    "hazard_ratio_unweighted_mean": eff.get("hazard_ratio_unweighted_mean"),
                    "hazard_ratio_curve_mean": eff.get("hazard_ratio_curve_mean"),
                    "discrete_risk_ratio_mean": eff.get("discrete_risk_ratio_mean"),
                    "risk_difference_mean": eff.get("risk_difference_mean"),
                    "risk_difference_horizon_mean": eff.get("risk_difference_horizon_mean"),
                    "risk_difference_t50_mean": rd_t50_m,
                    "risk_difference_t50_ci_lower": float(rd_t50_ci[0]) if rd_t50_ci[0] is not None else None,
                    "risk_difference_t50_ci_upper": float(rd_t50_ci[1]) if rd_t50_ci[1] is not None else None,
                    "risk_difference_t50_ci_width": rd_w,
                    "t50_time": t50_time,
                    "true_hr": c_spec.get("true_hr", c_spec.get("true_hr_delta", None)),
                }
            except Exception as e:
                causal_results[var_name] = {"error": str(e)}
                
    # 9. MCMC Convergence Diagnostics (R-hat and Effective Sample Size)
    diagnostics = {}
    n_chains = int(m_cfg.get("chains", 4))
    if sv_draws is not None and sv_draws.shape[0] >= n_chains and n_chains > 1:
        n_draws_total = sv_draws.shape[0]
        draws_per_chain = n_draws_total // n_chains
        if draws_per_chain >= 2:
            try:
                import arviz as az
                cohort_sv = sv_draws.mean(axis=1) # shape (n_draws_total, n_times)
                chains_mat = cohort_sv[:n_chains * draws_per_chain].reshape(n_chains, draws_per_chain, -1)
                rhat_arr = az.rhat(chains_mat)
                ess_arr = az.ess(chains_mat)
                diagnostics = {
                    "rhat_mean": float(np.nanmean(rhat_arr)),
                    "rhat_max": float(np.nanmax(rhat_arr)),
                    "ess_mean": float(np.nanmean(ess_arr)),
                    "ess_min": float(np.nanmin(ess_arr)),
                }
            except Exception:
                diagnostics = {"rhat_mean": None, "rhat_max": None, "ess_mean": None, "ess_min": None}

    total_duration = time.time() - start_total_time
    
    del model, post_test_train, trn
    gc.collect()
    
    return {
        "model": "pbart",
        "sv_mean": sv_mean,
        "ci_lower": sv_ci[0],
        "ci_upper": sv_ci[1],
        "sv_draws": sv_draws,
        "test_results": test_results,
        "causal_effects": causal_results,
        "feature_importance": feat_imp,
        "diagnostics": diagnostics,
        "timings": {
            "fit_seconds": fit_duration,
            "predict_seconds": pred_duration,
            "total_seconds": total_duration,
        }
    }

