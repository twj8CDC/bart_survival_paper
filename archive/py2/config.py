# config.py

SIMULATION_CONFIG = {
    "experiment": {
        "name": "simple1",
        "run_name": "small_particles_1000iter",
        "iters": 2,
        "seed_addl": 13,
        "sample_sizes": [400],
        "plot_all": True
    },
    
    "scenarios": {
        # ---------------------------------------------------------
        # Scenario 1: Simple, 1 group, 20% censoring
        # ---------------------------------------------------------
        "simple_1_1": {
            "dgp": {
                "type": "Simple, 1 group, 20% cens",
                "x_vars": 1, 
                "VAR_CLASS": [2],
                "VAR_PROB": [1],
                "scale_f": "7*x_mat[:,0]",
                "shape_f": "3",
                "cens_scale": 7
            },
            "model": {
                "trees": 100,
                "split_rules": ["pmb.ContinuousSplitRule()", "pmb.ContinuousSplitRule()"]
            },
            "sampler": {
                "draws": 300,
                "tune": 50,
                "cores": 4,
                "chains": 4,
                "compute_convergence_checks": False,
                "pgbart": {
                    "num_particles": 5,
                    "batch": (0.1, 0.1)
                }
            }
        },
        
        # ---------------------------------------------------------
        # Scenario 2: Simple, 2 group, 50% censoring
        # ---------------------------------------------------------
        "simple_2_2": {
            "dgp": {
                "type": "Simple, 2 group, 50% cens",
                "x_vars": 1, 
                "VAR_CLASS": [2],
                "VAR_PROB": [0.5],
                "scale_f": "12 - .5 *x_mat[:,0]",
                "shape_f": "1.2 + 1 *x_mat[:,0]",
                "cens_scale": 1.1
            },
            "model": {
                "trees": 100,
                "split_rules": [
                    "pmb.ContinuousSplitRule()", 
                    "pmb.ContinuousSplitRule()"
                ]
            },
            "sampler": {
                "draws": 300,
                "tune": 50,
                "cores": 4,
                "chains": 4,
                "compute_convergence_checks": False,
                "pgbart": {
                    "num_particles": 5,
                    "batch": (0.1, 0.1)
                }
            }
        }
        # Add simple_1_2, simple_2_1, complex_1, etc. using this exact schema.
    }
}