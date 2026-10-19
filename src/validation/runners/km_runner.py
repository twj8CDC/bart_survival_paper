"""
Kaplan-Meier Estimator Runner using lifelines.
Supports 1-sample baseline calibration and 2-sample subgroup estimation.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

def run_km(
    train_data: Dict[str, Any],
    is_two_sample: bool = False,
    group_col: int = 0
) -> Dict[str, Any]:
    """
    Executes Kaplan-Meier estimation and interpolates S(t) to the discrete evaluation times.
    """
    start_total_time = time.time()
    
    x_train = train_data["x_mat"]
    event_train = train_data["event_dict"]
    eval_times = train_data["eval_times"]
    
    t_ev = event_train["t_event"]
    s_ev = event_train["status"]
    
    n_obs = len(t_ev)
    
    if not is_two_sample:
        kmf = KaplanMeierFitter()
        kmf.fit(t_ev, event_observed=s_ev)
        
        # Extract survival probabilities at eval_times
        sv_curve = kmf.predict(eval_times).to_numpy()
        # Shape: (n_obs, len(eval_times)) where every subject shares the population estimate
        sv_mean = np.tile(sv_curve, (n_obs, 1))
        
        # Confidence interval bounds
        ci_df = kmf.confidence_interval_survival_function_
        ci_lower_vals = np.interp(eval_times, ci_df.index, ci_df.iloc[:, 0])
        ci_upper_vals = np.interp(eval_times, ci_df.index, ci_df.iloc[:, 1])
        ci_lower = np.tile(ci_lower_vals, (n_obs, 1))
        ci_upper = np.tile(ci_upper_vals, (n_obs, 1))
    else:
        # Two-sample subgroup KM (e.g. for Simple 2_1 and 2_2)
        group = (x_train[:, group_col] > 0.5).astype(int)
        sv_mean = np.zeros((n_obs, len(eval_times)), dtype=np.float64)
        ci_lower = np.zeros((n_obs, len(eval_times)), dtype=np.float64)
        ci_upper = np.zeros((n_obs, len(eval_times)), dtype=np.float64)
        
        for g_val in [0, 1]:
            msk = (group == g_val)
            if np.sum(msk) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(t_ev[msk], event_observed=s_ev[msk])
                sv_g = kmf.predict(eval_times).to_numpy()
                sv_mean[msk, :] = sv_g
                
                ci_df = kmf.confidence_interval_survival_function_
                ci_l = np.interp(eval_times, ci_df.index, ci_df.iloc[:, 0])
                ci_u = np.interp(eval_times, ci_df.index, ci_df.iloc[:, 1])
                ci_lower[msk, :] = ci_l
                ci_upper[msk, :] = ci_u
                
    total_duration = time.time() - start_total_time
    
    return {
        "model": "km",
        "sv_mean": sv_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "timings": {
            "total_seconds": total_duration
        }
    }

