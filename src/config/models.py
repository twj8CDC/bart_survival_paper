"""
Model Presets & Sampler Configurations for Validation Suite.
Configures p-BART (standard, fast, high_precision) and comparator models.
"""

from typing import Dict, Any

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Standard scientific preset matching empirical threshold study
    "standard": {
        "trees": 50,
        "draws": 1000,
        "tune": 250,
        "chains": 4,
        "thin": 4,
        "thin_idata": False,
        "update_fraction": 1.0,
        "use_sparsity": False,
        "split_method": "empirical",
        "max_cutpoints": 512,
        "min_node_size": 5,
        "k": 2.0,
    },
    # Backwards-compatible alias for standard
    "production": {
        "trees": 50,
        "draws": 1000,
        "tune": 250,
        "chains": 4,
        "thin": 4,
        "thin_idata": False,
        "update_fraction": 1.0,
        "use_sparsity": False,
        "split_method": "empirical",
        "max_cutpoints": 512,
        "min_node_size": 5,
        "k": 2.0,
    },
    # Production preset with reduced tuning (250 tune draws)
    "prod_tune250": {
        "trees": 50,
        "draws": 2500,
        "tune": 250,
        "chains": 4,
        "thin": 10,
        "thin_idata": False,
        "update_fraction": 1.0,
        "use_sparsity": False,
        "split_method": "empirical",
        "max_cutpoints": 512,
        "min_node_size": 5,
        "k": 2.0,
    },
    # Fast exploration preset for rapid smoke-testing and development
    "fast": {
        "trees": 25,
        "draws": 500,
        "tune": 250,
        "chains": 4,
        "thin": 5,
        "thin_idata": False,
        "update_fraction": 1.0,
        "use_sparsity": False,
        "split_method": "empirical",
        "max_cutpoints": 512,
        "min_node_size": 5,
        "k": 2.0,
    },
    # High-precision preset for publication tables
    "high_precision": {
        "trees": 100,
        "draws": 5000,
        "tune": 2000,
        "chains": 4,
        "thin": 10,
        "thin_idata": False,
        "update_fraction": 1.0,
        "use_sparsity": False,
        "split_method": "empirical",
        "max_cutpoints": 512,
        "min_node_size": 5,
        "k": 2.0,
    },
}

