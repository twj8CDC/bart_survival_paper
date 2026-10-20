import numpy as np
import functions as fn # Assuming you saved the consolidated functions here

def run_replicates(iters, n, seed_addl, scenario, model_dict, sampler_dict):
    """
    Executes multiple simulation iterations for a given scenario configuration.
    Returns a structured dictionary of metrics, completely decoupled from plotting.
    
    Args:
        iters (int): Number of iterations to run.
        n (int): Sample size for this batch.
        seed_addl (int): Additive factor for reproducible seed generation.
        scenario (dict): The DGP configuration dictionary (from config.py).
        model_dict (dict): The BART model configuration dictionary.
        sampler_dict (dict): The PyMC sampler configuration dictionary.
        
    Returns:
        dict: A JSON-serializable dictionary containing performance metrics across all iterations.
    """
    # Initialize a clean structure for tracking metrics across iterations
    # You can expand these keys based on what `fn.calculate_metrics` returns
    results = {
        "seeds": [],
        "train": {"rmse_mean": [], "bias_mean": []},
        "test": {"rmse_mean": [], "bias_mean": []}
    }

    # Calculate starting and ending seed logic based on your original codebase
    strt = n * seed_addl
    end = strt + iters

    for i in range(strt, end):
        print(f"  -> Running iteration {i - strt + 1}/{iters} (Seed: {i})")
        results["seeds"].append(i)

        # 1. Execute the Data Generating Process
        # We unpack the scenario dictionary seamlessly into the generator
        x_mat, event_dict, sv_true, sv_scale_true = fn.generate_data(
            rng_seed=i, 
            n=n, 
            dgp_config=scenario
        )
        
        
        # (Optional) If you have a separate test set generation, you would call it here:
        # x_mat_test, event_dict_test, sv_true_test, _ = fn.generate_data(rng_seed=i+1000, n=n, dgp_config=scenario)

        # 2. Fit the Model and Generate Predictions
        bsm, post_preds = fn.fit_and_predict_bart(
            x_mat=x_mat, 
            event_dict=event_dict, 
            model_config=model_dict, 
            sampler_config=sampler_dict
        )

        # 3. Calculate standardized metrics
        metrics = fn.calculate_metrics(
            sv_true=sv_true, 
            posterior_preds=post_preds
        )
        
        # (Optional) Calculate test metrics if you generated test predictions
        # metrics_test = fn.calculate_metrics(sv_true=sv_true_test, posterior_preds=post_preds_test)

        # 4. Append metrics to our results dictionary safely
        # We ensure everything is cast to a standard Python float to prevent JSON serialization errors
        for key, value in metrics.items():
            # If the metric happens to be returned as an array, average it for the summary log
            if isinstance(value, np.ndarray):
                value = float(np.mean(value))
            
            # Append to the training tracker
            if key in results["train"]:
                results["train"][key].append(float(value))
            else:
                # Dynamically add new metrics if fn.calculate_metrics outputs something new
                results["train"][key] = [float(value)]

        # Note: We are explicitly dropping `intvs` and `fig` generation from this loop.
        # Plotting and deep interval analysis will happen in a separate script that reads the saved outputs.

    return {"metrics": results}