"""
Unified Data Generating Process & Interval Discretization.
Uses bart_survival.simulation and faithful interval assignment.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from bart_survival import simulation as sm

def get_quant_times_from_survival(
    sv_true_mean: np.ndarray,
    true_times: np.ndarray,
    quantiles: List[float]
) -> np.ndarray:
    """Finds exact time points closest to requested survival distribution quantiles."""
    qnt_t = []
    for q in quantiles:
        diff = np.abs(sv_true_mean - q)
        idx = np.argmin(diff)
        qnt_t.append(int(true_times[idx]))
    return np.array(qnt_t, dtype=int)

def assign_quant_events(
    event_dict: Dict[str, np.ndarray],
    qnt_times: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Partitions continuous event times into the discrete evaluation intervals.
    Matches the grouping mechanics in _functions1.py:get_quant_events.
    """
    q = np.sort(qnt_times)
    t_raw = np.asarray(event_dict["t_event"], dtype=np.float64).ravel()
    s_raw = np.asarray(event_dict["status"], dtype=np.int32).ravel()
    
    t_out = t_raw.copy()
    s_out = s_raw.copy()
    
    for i in range(len(q)):
        if i == 0:
            msk = t_raw <= q[i]
            t_out[msk] = q[i]
        else:
            msk = (t_raw > q[i - 1]) & (t_raw <= q[i])
            t_out[msk] = q[i]
            if i == len(q) - 1:
                # Censored beyond final quantile
                tail_msk = t_raw > q[i]
                t_out[tail_msk] = q[i]
                s_out[tail_msk] = 0
                
    return {
        "t_event": t_out,
        "status": s_out,
        "t_raw": t_raw,
        "status_raw": s_raw,
    }

def generate_scenario_data(
    scenario_cfg: Dict[str, Any],
    n: int,
    seed: int,
    eval_times: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Generates synthetic survival cohort, analytically derived true survival,
    and discretized event indicators faithful to manuscript scenario specifications.
    If eval_times is provided, uses this exact evaluation grid (ensures perfect alignment
    between training and holdout test cohorts).
    """
    rng = np.random.default_rng(seed)
    
    x_vars = scenario_cfg["x_vars"]
    var_class = scenario_cfg["VAR_CLASS"]
    var_prob = scenario_cfg["VAR_PROB"]
    scale_f = scenario_cfg["scale_f"]
    shape_f = scenario_cfg["shape_f"]
    cens_scale = scenario_cfg.get("cens_scale", None)
    eos = scenario_cfg.get("eos", 180)
    
    # 1. Generate Covariates
    x_mat = sm.get_x_matrix(
        N=n,
        x_vars=x_vars,
        VAR_CLASS=var_class,
        VAR_PROB=var_prob,
        rng=rng
    )
    
    # 2. Continuous time simulation with analytical survival curves
    event_dict, sv_true, sv_scale_true = sm.simulate_survival(
        x_mat=x_mat,
        scale_f=scale_f,
        shape_f=shape_f,
        cens_scale=cens_scale,
        eos=eos,
        rng=rng
    )
    
    true_times = sv_true["true_times"]
    sv_true_mat = sv_true["sv_true"] # shape: (n_obs, len(true_times))
    
    # 3. Discretization
    category = scenario_cfg.get("category", "simple")
    
    if eval_times is not None:
        # Pre-specified evaluation grid (e.g. for test cohort aligned to train cohort)
        eval_times = np.sort(np.unique(np.asarray(eval_times, dtype=int)))
        disc_events = assign_quant_events(event_dict, eval_times)
    elif "quantiles" in scenario_cfg:
        # Simple & Complex 1/2: Quantiles on mean survival
        sv_mean = sv_true_mat.mean(axis=0)
        qnt_times = get_quant_times_from_survival(sv_mean, true_times, scenario_cfg["quantiles"])
        qnt_times = np.sort(np.unique(qnt_times))
        disc_events = assign_quant_events(event_dict, qnt_times)
        eval_times = qnt_times
    elif "time_quantiles" in scenario_cfg:
        # Complex 3: Quantiles on event times / follow-up horizon
        max_t = int(event_dict["t_event"].max())
        qnt_times = np.ceil(np.quantile(range(max(1, max_t)), scenario_cfg["time_quantiles"])).astype(int)
        qnt_times = np.sort(np.unique(qnt_times))
        disc_events = assign_quant_events(event_dict, qnt_times)
        eval_times = qnt_times
    else:
        eval_times = np.unique(event_dict["t_event"])
        disc_events = event_dict
        
    # Extract true survival evaluated at the exact discrete evaluation times
    # true_times is 1..180 (1-indexed days)
    time_indices = np.clip(eval_times - 1, 0, sv_true_mat.shape[1] - 1).astype(int)
    sv_true_eval = sv_true_mat[:, time_indices]
    
    cens_rate = float(np.mean(disc_events["status"] == 0))
    
    # Calculate empirical interval event weights
    event_counts = np.array([
        np.sum((disc_events["t_event"] == t) & (disc_events["status"].ravel() == 1))
        for t in eval_times
    ], dtype=np.float64)
    tot_ev = float(np.sum(event_counts))
    eff_weights = event_counts / tot_ev if tot_ev > 0 else np.ones(len(eval_times)) / len(eval_times)
    
    true_causal = compute_true_causal_estimands(
        scenario_cfg=scenario_cfg,
        x_mat=x_mat,
        eval_times=eval_times,
        event_weights=eff_weights
    )
    
    # Calculate risk stratification based on individual true analytical survival
    subject_true_mean_sv = np.mean(sv_true_eval, axis=1)
    q25_threshold = float(np.quantile(subject_true_mean_sv, 0.25))
    q75_threshold = float(np.quantile(subject_true_mean_sv, 0.75))
    
    idx_bottom25 = np.where(subject_true_mean_sv <= q25_threshold)[0] # High-risk cohort
    idx_top25 = np.where(subject_true_mean_sv >= q75_threshold)[0]    # Low-risk cohort
    idx_min_subject = int(np.argmin(subject_true_mean_sv))             # Extreme high-risk boundary
    idx_max_subject = int(np.argmax(subject_true_mean_sv))             # Extreme low-risk boundary
    
    true_sv_top25 = np.mean(sv_true_eval[idx_top25, :], axis=0)
    true_sv_bot25 = np.mean(sv_true_eval[idx_bottom25, :], axis=0)
    true_sv_min = sv_true_eval[idx_min_subject, :]
    true_sv_max = sv_true_eval[idx_max_subject, :]
    
    return {
        "x_mat": x_mat,
        "event_dict": disc_events,
        "event_dict_continuous": event_dict,
        "eval_times": eval_times,
        "sv_true_eval": sv_true_eval,
        "sv_true_full": sv_true_mat,
        "true_times": true_times,
        "censoring_rate": cens_rate,
        "true_causal": true_causal,
        "event_weights": eff_weights,
        "seed": seed,
        "n": n,
        "stratification": {
            "idx_top25": idx_top25,
            "idx_bottom25": idx_bottom25,
            "idx_min": idx_min_subject,
            "idx_max": idx_max_subject,
            "true_sv_top25": true_sv_top25,
            "true_sv_bot25": true_sv_bot25,
            "true_sv_min": true_sv_min,
            "true_sv_max": true_sv_max,
        }
    }


def compute_true_causal_estimands(
    scenario_cfg: Dict[str, Any],
    x_mat: np.ndarray,
    eval_times: np.ndarray,
    event_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes exact marginal counterfactual survival curves, time-varying hazard ratios,
    discrete risk ratios, and absolute risk difference curves directly from mathematical DGP equations.
    """
    scale_f = scenario_cfg.get("scale_f")
    shape_f = scenario_cfg.get("shape_f")
    causal_specs = scenario_cfg.get("causal_contrasts", [])
    if not scale_f or not shape_f or not causal_specs:
        return {}
        
    n_times = len(eval_times)
    if event_weights is None or np.sum(event_weights) == 0:
        w = np.ones(n_times, dtype=np.float64) / n_times
    else:
        w = np.asarray(event_weights, dtype=np.float64)
        w = w / w.sum()
        
    true_causal = {}
    for c_spec in causal_specs:
        var_idx = c_spec["variable"]
        var_name = c_spec.get("name", f"x{var_idx}")
        c_type = c_spec.get("type", "binary")
        
        if c_type in ["binary", "contrast"]:
            vals = c_spec.get("values", [0, 1])
            x0 = x_mat.copy(); x0[:, var_idx] = vals[0]
            x1 = x_mat.copy(); x1[:, var_idx] = vals[1]
        elif c_type == "continuous_shift":
            delta = c_spec.get("delta", 0.20)
            x0 = x_mat.copy()
            x1 = x_mat.copy(); x1[:, var_idx] = np.clip(x1[:, var_idx] + delta, 0, 1)
        else:
            continue
            
        scale_0 = eval(scale_f, {"np": np, "x_mat": x0})
        shape_0 = eval(str(shape_f), {"np": np, "x_mat": x0}) if "x_mat" in str(shape_f) else float(eval(str(shape_f)))
        sv_0 = sm.weibull_survival(scale_0, shape_0, eval_times)
        s0_mean = sv_0.mean(axis=0)
        
        scale_1 = eval(scale_f, {"np": np, "x_mat": x1})
        shape_1 = eval(str(shape_f), {"np": np, "x_mat": x1}) if "x_mat" in str(shape_f) else float(eval(str(shape_f)))
        sv_1 = sm.weibull_survival(scale_1, shape_1, eval_times)
        s1_mean = sv_1.mean(axis=0)
        
        rd_curve = s0_mean - s1_mean
        s0_prev = np.r_[1.0, s0_mean[:-1]]
        s1_prev = np.r_[1.0, s1_mean[:-1]]
        h0 = np.clip((s0_prev - s0_mean) / np.maximum(s0_prev, 1e-6), 1e-4, 1.0 - 1e-4)
        h1 = np.clip((s1_prev - s1_mean) / np.maximum(s1_prev, 1e-6), 1e-4, 1.0 - 1e-4)
        
        hr_curve = np.clip(-np.log(1.0 - h1) / -np.log(1.0 - h0), 0.02, 50.0)
        rr_curve = np.clip(h1 / h0, 0.02, 50.0)
        hr_weighted = float(np.exp(np.dot(np.log(hr_curve), w)))
        hr_unweighted = float(np.exp(np.mean(np.log(hr_curve))))
        
        t50_idx = len(eval_times) // 2
        t50_time = float(eval_times[t50_idx])
        true_rd_t50 = float(rd_curve[t50_idx])
        
        true_causal[var_name] = {
            "hazard_ratio_weighted_mean": hr_weighted,
            "hazard_ratio_unweighted_mean": hr_unweighted,
            "hazard_ratio_curve": np.round(hr_curve, 4).tolist(),
            "discrete_risk_ratio_curve": np.round(rr_curve, 4).tolist(),
            "risk_difference_curve": np.round(rd_curve, 4).tolist(),
            "horizon_risk_difference": float(rd_curve[-1]),
            "true_rd_at_t50": true_rd_t50,
            "t50_time": t50_time,
            "t50_index": t50_idx,
        }
        
    return true_causal


