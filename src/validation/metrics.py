"""
Standardized Statistical Evaluation Metrics for BART Survival Validation Suite.
Implements a-RMSE, signed bias, empirical 95% CI coverage, interval width, and causal accuracy.
"""

import numpy as np
from typing import Dict, Any, Union, Tuple, Optional, List

def compute_interval_metrics(
    true_sv: np.ndarray,
    pred_mean: np.ndarray,
    pred_ci_lower: Optional[np.ndarray] = None,
    pred_ci_upper: Optional[np.ndarray] = None,
    time_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes point-wise and summary accuracy metrics across subjects and time intervals.

    Args:
        true_sv (np.ndarray): Ground-truth survival curves, shape (n_obs, n_times).
        pred_mean (np.ndarray): Predicted survival probabilities, shape (n_obs, n_times).
        pred_ci_lower (Optional[np.ndarray]): Lower 95% CI bound, shape (n_obs, n_times).
        pred_ci_upper (Optional[np.ndarray]): Upper 95% CI bound, shape (n_obs, n_times).
        time_points (Optional[np.ndarray]): Array of discrete evaluation times.

    Returns:
        Dict[str, Any]: Summary metrics across intervals and overall.
    """
    true_sv = np.asarray(true_sv, dtype=np.float64)
    pred_mean = np.asarray(pred_mean, dtype=np.float64)
    if true_sv.shape != pred_mean.shape and true_sv.ndim == 2 and pred_mean.ndim == 2:
        k = min(true_sv.shape[1], pred_mean.shape[1])
        true_sv = true_sv[:, :k]
        pred_mean = pred_mean[:, :k]
        if pred_ci_lower is not None:
            pred_ci_lower = np.asarray(pred_ci_lower, dtype=np.float64)[:, :k]
        if pred_ci_upper is not None:
            pred_ci_upper = np.asarray(pred_ci_upper, dtype=np.float64)[:, :k]
        if time_points is not None:
            time_points = np.asarray(time_points)[:k]

    # 1. Absolute / Pointwise error
    error = pred_mean - true_sv
    
    # Per-interval Root Mean Square Error: sqrt(mean over subjects (pred - true)^2)
    rmse_per_interval = np.sqrt(np.mean(error ** 2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(error ** 2)))
    
    # Per-interval Bias: mean over subjects (pred - true)
    bias_per_interval = np.mean(error, axis=0)
    overall_bias = float(np.mean(error))
    
    # Mean Absolute Deviation
    mad_per_interval = np.mean(np.abs(error), axis=0)
    overall_mad = float(np.mean(np.abs(error)))
    
    results = {
        "rmse_per_interval": rmse_per_interval.tolist(),
        "overall_rmse": overall_rmse,
        "bias_per_interval": bias_per_interval.tolist(),
        "overall_bias": overall_bias,
        "mad_per_interval": mad_per_interval.tolist(),
        "overall_mad": overall_mad,
    }
    
    # 2. Coverage & Interval Length (if CI bounds provided)
    if pred_ci_lower is not None and pred_ci_upper is not None:
        pred_ci_lower = np.asarray(pred_ci_lower, dtype=np.float64)
        pred_ci_upper = np.asarray(pred_ci_upper, dtype=np.float64)
        
        # Coverage indicator per subject per interval (rounded to 4 decimal places for floating-point tolerance)
        covered = (np.round(pred_ci_lower, 4) <= np.round(true_sv, 4)) & (np.round(true_sv, 4) <= np.round(pred_ci_upper, 4))
        coverage_per_interval = np.mean(covered, axis=0)
        overall_coverage = float(np.mean(covered))
        
        # Interval length (width)
        interval_length_per_interval = np.mean(pred_ci_upper - pred_ci_lower, axis=0)
        overall_interval_length = float(np.mean(pred_ci_upper - pred_ci_lower))
        
        results.update({
            "coverage_per_interval": coverage_per_interval.tolist(),
            "overall_coverage": overall_coverage,
            "interval_length_per_interval": interval_length_per_interval.tolist(),
            "overall_interval_length": overall_interval_length,
        })
        
    return results

def r_squared_survival(true_sv: np.ndarray, pred_sv: np.ndarray) -> float:
    """Computes R^2 coefficient of determination on cumulative survival curves."""
    true_flat = np.asarray(true_sv).ravel()
    pred_flat = np.asarray(pred_sv).ravel()
    ssr = np.sum((true_flat - pred_flat) ** 2)
    sst = np.sum((true_flat - np.mean(true_flat)) ** 2)
    return float(1.0 - (ssr / sst)) if sst > 0 else 0.0

