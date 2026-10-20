"""
Declarative DGP Scenario Specifications for BART Survival Validation Suite.
Faithful to manuscript DGPs with formal discretization and causal estimands.
"""

from typing import Dict, Any, List, Optional
import numpy as np

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # Scenario 1: One-Sample (Null / Baseline calibration)
    # -------------------------------------------------------------
    "simple_1_1": {
        "name": "Simple 1-Sample (20% Censoring)",
        "category": "simple",
        "type": "Simple, 1 group, 20% cens",
        "x_vars": 1,
        "VAR_CLASS": [2],
        "VAR_PROB": [1.0],
        "scale_f": "7*x_mat[:,0]",
        "shape_f": "3",
        "cens_scale": 7.0,
        "eos": 180,
        # Quantiles of survival distribution evaluated in paper
        "quantiles": [0.90, 0.75, 0.50, 0.25, 0.10],
        "default_sample_sizes": [200, 400],
        "causal_contrasts": [],
    },
    "simple_1_2": {
        "name": "Simple 1-Sample (50% Censoring)",
        "category": "simple",
        "type": "Simple, 1 group, 50% cens",
        "x_vars": 1,
        "VAR_CLASS": [2],
        "VAR_PROB": [1.0],
        "scale_f": "7*x_mat[:,0]",
        "shape_f": "3",
        "cens_scale": 1.3,
        "eos": 180,
        "quantiles": [0.90, 0.75, 0.50, 0.25, 0.10],
        "default_sample_sizes": [200, 400],
        "causal_contrasts": [],
    },
    
    # -------------------------------------------------------------
    # Scenario 2: Two-Sample (Binary Group Contrast)
    # -------------------------------------------------------------
    "simple_2_1": {
        "name": "Simple 2-Sample (20% Censoring)",
        "category": "simple",
        "type": "Simple, 2 group, 20% cens",
        "x_vars": 1,
        "VAR_CLASS": [2],
        "VAR_PROB": [0.5],
        "scale_f": "12 - .5 *x_mat[:,0]",
        "shape_f": "1.2 + 1 *x_mat[:,0]",
        "cens_scale": 5.9,
        "eos": 180,
        "quantiles": [0.90, 0.75, 0.50, 0.25, 0.10],
        "default_sample_sizes": [200, 400],
        "causal_contrasts": [
            {"variable": 0, "name": "x0", "type": "binary", "values": [0, 1]}
        ],
    },
    "simple_2_2": {
        "name": "Simple 2-Sample (50% Censoring)",
        "category": "simple",
        "type": "Simple, 2 group, 50% cens",
        "x_vars": 1,
        "VAR_CLASS": [2],
        "VAR_PROB": [0.5],
        "scale_f": "12 - .5 *x_mat[:,0]",
        "shape_f": "1.2 + 1 *x_mat[:,0]",
        "cens_scale": 1.1,
        "eos": 180,
        "quantiles": [0.90, 0.75, 0.50, 0.25, 0.10],
        "default_sample_sizes": [200, 400],
        "causal_contrasts": [
            {"variable": 0, "name": "x0", "type": "binary", "values": [0, 1]}
        ],
    },
    
    # -------------------------------------------------------------
    # Scenario 3: Complex Multivariate Binary (Proportional Hazards)
    # -------------------------------------------------------------
    "complex_1": {
        "name": "Complex Multivariate Binary (Proportional Hazards, 20% Censoring)",
        "category": "complex_binary",
        "type": "Complex, binary, prop, 20% cens",
        "x_vars": 9,
        "VAR_CLASS": [2] * 9,
        "VAR_PROB": [0.5] * 9,
        "scale_f": "np.exp(3 + 0.1*(x_mat[:,0] + x_mat[:,1] + x_mat[:,2] + x_mat[:,3] + x_mat[:,4] + x_mat[:,5]) + x_mat[:,6])",
        "shape_f": "2",
        "cens_scale": 3.5,
        "eos": 180,
        # 7 quantiles evaluated in paper
        "quantiles": [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.01],
        "default_sample_sizes": [200, 400],
        "causal_contrasts": [
            {"variable": 0, "name": "x0", "type": "binary", "values": [0, 1], "true_hr": float(np.exp(-2.0 * 0.10))},
            {"variable": 6, "name": "x6", "type": "binary", "values": [0, 1], "true_hr": float(np.exp(-2.0 * 1.00))},
        ],
    },
    
    # -------------------------------------------------------------
    # Scenario 4: Complex Multivariate Binary (Non-Proportional Hazards)
    # -------------------------------------------------------------
    "complex_2": {
        "name": "Complex Multivariate Binary (Non-Proportional Hazards, 20% Censoring)",
        "category": "complex_binary",
        "type": "Complex, binary, nonprop, 20% cens",
        "x_vars": 9,
        "VAR_CLASS": [2] * 9,
        "VAR_PROB": [0.5] * 9,
        "scale_f": "20 + 5*(x_mat[:,0] + x_mat[:,1] + x_mat[:,2] + x_mat[:,3] + x_mat[:,4] + x_mat[:,5] + 10*x_mat[:,6])",
        "shape_f": "0.7 + 1.3*x_mat[:,6]",
        "cens_scale": 3.9,
        "eos": 180,
        "quantiles": [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.01],
        "default_sample_sizes": [200, 400],
        "causal_contrasts": [
            {"variable": 0, "name": "x0", "type": "binary", "values": [0, 1]},
            {"variable": 6, "name": "x6", "type": "binary", "values": [0, 1]}, # Non-proportional shape driver
        ],
    },
    
    # -------------------------------------------------------------
    # Scenario 5: Complex Continuous (Nonlinear Interactions & Non-Proportional Shape)
    # -------------------------------------------------------------
    "complex_3": {
        "name": "Complex Multivariable Regression (Nonlinear Interactions & Non-Proportional Shape)",
        "category": "complex_continuous",
        "type": "Complex, continuous, 20% cens",
        "x_vars": 10,
        "VAR_CLASS": [2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        "VAR_PROB": [0.5, 0.5, None, None, None, None, None, None, None, None],
        "scale_f": "50 + 100*x_mat[:,0] - 100*x_mat[:,1]*x_mat[:,0] + 10*x_mat[:,2] + 20*np.power(x_mat[:,3],2) + 20 * x_mat[:,3] * x_mat[:,4]",
        "shape_f": "1 + 2*x_mat[:,0] + 1*x_mat[:,1]*x_mat[:,0] + .1*x_mat[:,1]*np.power(x_mat[:,4],2)",
        "cens_scale": 2.5,
        "eos": 180,
        "time_quantiles": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "default_sample_sizes": [400, 1000, 4000],
        "test_sample_size": 400,
        "causal_contrasts": [
            {"variable": 0, "name": "x0", "type": "binary", "values": [0, 1]},
            {"variable": 1, "name": "x1", "type": "binary", "values": [0, 1]},
            {"variable": 3, "name": "x3", "type": "continuous_shift", "delta": 0.20},
        ],
    },
}
