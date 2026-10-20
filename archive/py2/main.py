import itertools
import time
from config import SIMULATION_CONFIG
from logger import SimLogger
import traceback
# import _functions1 as fn 
import _sim_iter as si 

# main.py
from logger import SimLogger
# ... (other imports)

def main():
    print("STARTING")
    exp_cfg = SIMULATION_CONFIG["experiment"]
    logger = SimLogger(experiment_name=exp_cfg["name"])
    
    try:
        logger.log_dict({
            "iters": exp_cfg["iters"], 
            "seed_addl": exp_cfg["seed_addl"]
        }, "experiment_meta.json")

        grid = itertools.product(exp_cfg["sample_sizes"], SIMULATION_CONFIG["scenarios"].items())
        
        for n, (scenario_name, config) in grid:
            print(f"=== Starting {scenario_name} | N={n} ===")
            
        
        # Run the iterator (we will build this next)
            results = si.run_replicates(
                iters=exp_cfg["iters"], 
                n=n,
                seed_addl=exp_cfg["seed_addl"],
                scenario=config["dgp"], 
                model_dict=config["model"], 
                sampler_dict=config["sampler"]
            )
            
            # # 2. Use the custom logger to save the outputs
            logger.log_dict(config, f"config_{scenario_name}_N{n}.json")
            logger.log_metrics(scenario_name, n, results["metrics"])
            logger.log_dict({"time_seconds": time.time() - strt_time}, f"time_{scenario_name}_N")

        logger.log_finish()
        
    except Exception as e:
            print("\n" + "="*50)
            print("❌ CRITICAL ERROR: Simulation Failed")
            print("="*50)
            
            # This function prints the exact error and the full stack trace
            traceback.print_exc()
            
            print("="*50)
            # The _status_RUNNING.json file will remain in the directory, 
            # letting you know this specific run folder contains a crashed run.
if __name__ == "__main__":
    main()