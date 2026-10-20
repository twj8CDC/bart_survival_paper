import numpy as np
import sksurv as sks
from bart_survival import surv_bart as sb 
from bart_survival import simulation as sm

# ---------------------------------------------------------
# 1. Data Generating Process (DGP)
# ---------------------------------------------------------
def generate_data(rng_seed, n, dgp_config):
    """
    Unified DGP. It takes the dgp dictionary from config.py and unpacks it 
    directly into the bart_survival simulation generator.
    """
    rng = np.random.default_rng(rng_seed)
    
    # Generate Covariates
    x_mat = sm.get_x_matrix(
        N=n,
        x_vars=dgp_config.get("x_vars", 1),
        VAR_CLASS=dgp_config.get("VAR_CLASS", [2]),
        VAR_PROB=dgp_config.get("VAR_PROB", [0.5]),
        rng=rng
    )
    
    # Generate Survival Times
    # The **dgp_config automatically maps keys like 'scale_f', 'shape_f', and 'cens_scale'
    # without us needing to hardcode them as function arguments.
    event_dict, sv_true, sv_scale_true = sm.simulate_survival(
        x_mat=x_mat,
        scale_f=dgp_config["scale_f"],
        shape_f=dgp_config["shape_f"],
        eos=dgp_config.get("eos", 180),
        cens_scale=dgp_config.get("cens_scale", None),
        time_scale=dgp_config.get("time_scale", 60), # Useful for testing discretization bias
        true_only=False,
        rng=rng
    )
    
    return x_mat, event_dict, sv_true, sv_scale_true

# ---------------------------------------------------------
# 2. Model Fitting and Prediction
# ---------------------------------------------------------
def fit_and_predict_bart(x_mat, event_dict, model_config, sampler_config, time_scale=60):
    """
    Unified wrapper to prep the data, train the BartSurvModel, and return the posterior.
    """
    # Prepare sklearn-formatted survival data
    # t_scale = sb.get_time_transform(event_dict["t_event"], time_scale=time_scale)
    # y_sk = sb.get_y_sklearn(event_dict["status"], t_scale)
    
    # Format for PyMC/BART
    trn = sb.get_surv_pre_train(y_time=event_dict["t_event"], y_status=event_dict["status"], x=x_mat, weight=None)
    
    # Initialize Model
    bsm = sb.BartSurvModel(model_config=model_config, sampler_config=sampler_config)
    
    # Fit Model
    bsm.fit(
        y=trn["y"],
        X=trn["x"],
        weights=trn["w"],
        coords=trn["coord"],
        random_seed=sampler_config.get("seed", None)
    )
    
    # Get Posterior Predictions on the training set (or pass a separate x_test if needed)
    tst = sb.get_posterior_test(y_status=event_dict["status"], y_time=event_dict["t_event"],x=x_mat)
    post = bsm.sample_posterior_predictive(X_pred=tst["post_x"], coords=tst["coords"])

    return bsm, post

# ---------------------------------------------------------
# 3. Standardized Evaluation Metrics
# ---------------------------------------------------------
def calculate_metrics(sv_true, posterior_preds):
    """
    Consolidates metric calculations (RMSE, Bias, HDI/CI bounds) 
    that were previously scattered across shared functions.
    """
    
    posterior_preds = sb.get_sv_prob(posterior_preds)
    # Extract the mean prediction and true survival matrix
    sv_pred_mean = posterior_preds["sv"].mean(axis=0) # Average across MCMC draws
    sv_true_mat = sv_true["sv_true"]
    
    # Ensure dimensions match before calculating
    # (Assuming sv_true_mat and sv_pred_mean share the same shape [N_obs, T_timepoints])
    
    # Calculate continuous metrics across the entire matrix
    bias_matrix = sv_pred_mean - sv_true_mat
    rmse_matrix = np.sqrt((bias_matrix ** 2))
    
    # Aggregate into scalar values for easy logging
    metrics = {
        "rmse_mean": float(np.mean(rmse_matrix)),
        "bias_mean": float(np.mean(bias_matrix)),
        # You can add HDI and CI coverage logic here based on your pymc posteriors
        # e.g., "hdi_95_coverage": compute_coverage(sv_true_mat, posterior_preds["hdi"])
    }
    
    return metrics