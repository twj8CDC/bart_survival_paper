"""
r-BART (BART::surv.bart) Model Runner with historical lookup fallback.
Runs Rscript subprocess with tempfile isolation, extracting survival curves,
risk-stratified subgroup marginals (top 25% / bottom 25%), and causal risk differences RD(t50).
"""

import time
import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def run_rbart(
    train_data: Dict[str, Any],
    skip_r: bool = False,
    scenario_id: Optional[str] = None,
    test_data: Optional[Dict[str, Any]] = None,
    causal_specs: Optional[List[Dict[str, Any]]] = None,
    r_script_path: str = "/usr/local/bin/Rscript"
) -> Dict[str, Any]:
    """
    Executes r-BART via Rscript. Computes survival curves, hold-out predictions,
    top/bottom 25% risk-stratified marginals, and causal RD(t50) with 95% CIs.
    """
    start_total_time = time.time()
    
    x_train = train_data["x_mat"]
    event_train = train_data["event_dict"]
    eval_times = train_data["eval_times"]
    
    t_ev = event_train["t_event"]
    s_ev = event_train["status"]
    n_obs = len(t_ev)
    
    if skip_r:
        # Load historical r-BART benchmark metrics if available
        base_dir = Path("/Users/jacobtiegs/projects/bart_val_online/bart_survival_paper_1000")
        hist_path = base_dir / "results" / "historical_rbart_baseline.json"
        hist_entry = None
        if hist_path.exists() and scenario_id:
            try:
                with open(hist_path, "r") as f:
                    all_hist = json.load(f)
                for k, v in all_hist.items():
                    if scenario_id in k:
                        hist_entry = v
                        break
            except Exception:
                pass
                
        return {
            "model": "rbart",
            "skipped": True,
            "historical_baseline": hist_entry,
            "timings": {"total_seconds": 0.0}
        }
        
    # Check if Rscript exists
    if not (os.path.exists(r_script_path) or subprocess.run(["which", "Rscript"], capture_output=True).returncode == 0):
        print(f"Warning: Rscript not found at {r_script_path}. Falling back to historical lookup.")
        return run_rbart(train_data, skip_r=True, scenario_id=scenario_id)
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_in = tmp_path / "train_data.csv"
        csv_out = tmp_path / "rbart_preds.csv"
        r_script_file = tmp_path / "run_rbart.R"
        
        # Prepare input CSV: columns [t, s, x0, x1, ...]
        col_names = ["t", "s"] + [f"x{i}" for i in range(x_train.shape[1])]
        df_in = pd.DataFrame(np.hstack([t_ev.reshape(-1, 1), s_ev.reshape(-1, 1), x_train]), columns=col_names)
        df_in.to_csv(csv_in, index=False)
        
        # Prepare test holdout CSV if present
        csv_test = tmp_path / "test_data.csv"
        has_test = (test_data is not None and "x_mat" in test_data)
        if has_test:
            df_test = pd.DataFrame(test_data["x_mat"], columns=[f"x{i}" for i in range(test_data["x_mat"].shape[1])])
            df_test.to_csv(csv_test, index=False)
            
        # Prepare Stratification indices (top 25% and bottom 25%)
        csv_strat = tmp_path / "strat_indices.csv"
        has_strat = ("stratification" in train_data and train_data["stratification"] is not None)
        if has_strat:
            s_dict = train_data["stratification"]
            max_len = max(len(s_dict["idx_top25"]), len(s_dict["idx_bottom25"]))
            s_df = pd.DataFrame({
                "top25": list(s_dict["idx_top25"]) + [-1] * (max_len - len(s_dict["idx_top25"])),
                "bot25": list(s_dict["idx_bottom25"]) + [-1] * (max_len - len(s_dict["idx_bottom25"])),
            })
            s_df.to_csv(csv_strat, index=False)
            
        # Prepare Causal Counterfactuals CSV
        csv_causal = tmp_path / "causal_contrasts.csv"
        has_causal = (causal_specs is not None and len(causal_specs) > 0 and x_train.shape[1] > 1)
        causal_var_map = {}
        if has_causal:
            causal_rows = []
            for c_spec in causal_specs:
                v_idx = c_spec["variable"]
                v_name = c_spec.get("name", f"x{v_idx}")
                causal_var_map[v_name] = v_idx
                c_type = c_spec.get("type", "binary")
                
                if c_type in ["binary", "contrast"]:
                    vals = c_spec.get("values", [0, 1])
                    # Intervention arm x1 (vals[1]) and Reference arm x0 (vals[0])
                    x1 = x_train.copy(); x1[:, v_idx] = vals[1]
                    x0 = x_train.copy(); x0[:, v_idx] = vals[0]
                elif c_type == "continuous_shift":
                    delta = c_spec.get("delta", 0.20)
                    # Modified treatment policy: intervention shifted by delta vs natural course
                    x1 = x_train.copy(); x1[:, v_idx] = np.clip(x1[:, v_idx] + delta, 0, 1)
                    x0 = x_train.copy()
                else:
                    x1 = x_train.copy(); x1[:, v_idx] = 1
                    x0 = x_train.copy(); x0[:, v_idx] = 0
                causal_rows.append(np.vstack([x1, x0]))
            if causal_rows:
                causal_mat = np.vstack(causal_rows)
                pd.DataFrame(causal_mat, columns=[f"x{i}" for i in range(x_train.shape[1])]).to_csv(csv_causal, index=False)
            
        seed_val = int(train_data.get("seed", 99))
        t50_idx = len(eval_times) // 2
        
        # R script to run BART::surv.bart
        r_code = f"""
        suppressPackageStartupMessages(library(BART))
        set.seed({seed_val})
        df <- read.csv("{csv_in}")
        times <- df$t
        delta <- df$s
        x <- as.matrix(df[, -(1:2), drop=FALSE])
        N_train <- nrow(x)
        
        # Scenario 1: One-sample null (1 feature, all constant)
        if (ncol(x) == 1 && length(unique(x[,1])) == 1) {{
            post <- mc.surv.bart(times=times, delta=delta, seed={seed_val}, mc.cores=4)
            
            sv_mu <- post$surv.test.mean
            sv_cil <- apply(post$surv.test, 2, quantile, probs=0.025)
            sv_cih <- apply(post$surv.test, 2, quantile, probs=0.975)
            out_df <- data.frame(
                grp = rep(x[1,1], length(sv_mu)),
                sv_mu = sv_mu,
                sv_cil = sv_cil,
                sv_cih = sv_cih
            )
            write.csv(out_df, "{csv_out}", row.names=FALSE)
            write.csv(as.data.frame(post$surv.test), "{tmp_path / 'sv_cohort_draws.csv'}", row.names=FALSE)
            
        # Scenario 2: Two-sample binary group (1 feature, 2 unique values)
        }} else if (ncol(x) == 1 && length(unique(x[,1])) == 2) {{
            x_test <- matrix(sort(unique(x[,1])), ncol=1)
            post <- mc.surv.bart(times=times, delta=delta, x.train=x, x.test=x_test, seed={seed_val}, mc.cores=4)
            
            sv_mu <- post$surv.test.mean
            sv_cil <- apply(post$surv.test, 2, quantile, probs=0.025)
            sv_cih <- apply(post$surv.test, 2, quantile, probs=0.975)
            n_grp <- length(sv_mu) / nrow(x_test)
            out_df <- data.frame(
                grp = rep(x_test[,1], each=n_grp),
                sv_mu = sv_mu,
                sv_cil = sv_cil,
                sv_cih = sv_cih
            )
            write.csv(out_df, "{csv_out}", row.names=FALSE)
            sv_cohort_scen2 <- 0.5 * (post$surv.test[, 1:n_grp, drop=FALSE] + post$surv.test[, (n_grp+1):(2*n_grp), drop=FALSE])
            write.csv(as.data.frame(sv_cohort_scen2), "{tmp_path / 'sv_cohort_draws.csv'}", row.names=FALSE)
            
            # Compute causal effect for binary group if requested
            if ({str(len(causal_specs) > 0 if causal_specs else False).upper()}) {{
                K <- n_grp
                k50 <- {t50_idx + 1}
                s0_mat <- post$surv.test[, 1:K, drop=FALSE]
                s1_mat <- post$surv.test[, (K+1):(2*K), drop=FALSE]
                
                s1_prev <- cbind(1.0, s1_mat[, 1:(K-1), drop=FALSE])
                s0_prev <- cbind(1.0, s0_mat[, 1:(K-1), drop=FALSE])
                h1_mat <- pmin(pmax((s1_prev - s1_mat) / pmax(s1_prev, 1e-6), 1e-4), 1 - 1e-4)
                h0_mat <- pmin(pmax((s0_prev - s0_mat) / pmax(s0_prev, 1e-6), 1e-4), 1 - 1e-4)
                
                hr_mat <- pmin(pmax(-log(1 - h1_mat) / -log(1 - h0_mat), 0.02), 50.0)
                log_hr_d <- rowMeans(log(hr_mat))
                hr_d <- exp(log_hr_d)
                
                hr_m <- exp(mean(log_hr_d))
                hr_cil <- as.numeric(quantile(hr_d, 0.025))
                hr_cih <- as.numeric(quantile(hr_d, 0.975))
                hr_w <- hr_cih - hr_cil
                
                d_rd <- s0_mat[, k50] - s1_mat[, k50]
                rd_m <- mean(d_rd)
                rd_cil <- as.numeric(quantile(d_rd, 0.025))
                rd_cih <- as.numeric(quantile(d_rd, 0.975))
                rd_w <- rd_cih - rd_cil
                
                causal_df <- data.frame(
                    contrast = 1,
                    hr_mean = hr_m,
                    hr_cil = hr_cil,
                    hr_cih = hr_cih,
                    hr_width = hr_w,
                    rd_mean = rd_m,
                    rd_cil = rd_cil,
                    rd_cih = rd_cih,
                    rd_width = rd_w
                )
                write.csv(causal_df, "{tmp_path / 'causal_out.csv'}", row.names=FALSE)
            }}
            
        # Scenario 3: Complex multi-covariate scenarios
        }} else {{
            test_exists <- file.exists("{csv_test}")
            causal_exists <- file.exists("{csv_causal}")
            
            x_combined_test <- x
            N_test <- 0
            if (test_exists) {{
                x_test_holdout <- as.matrix(read.csv("{csv_test}"))
                N_test <- nrow(x_test_holdout)
                x_combined_test <- rbind(x_combined_test, x_test_holdout)
            }}
            N_causal <- 0
            if (causal_exists) {{
                x_causal <- as.matrix(read.csv("{csv_causal}"))
                N_causal <- nrow(x_causal)
                x_combined_test <- rbind(x_combined_test, x_causal)
            }}
            
            post <- mc.surv.bart(times=times, delta=delta, x.train=x, x.test=x_combined_test, seed={seed_val}, mc.cores=4)
            
            K <- post$K
            N_total <- nrow(x_combined_test)
            
            # Extract training predictions
            sv_mu_train <- t(matrix(post$surv.test.mean[1:(N_train * K)], ncol=N_train, nrow=K))
            sv_cil_train <- t(matrix(apply(post$surv.test[, 1:(N_train * K)], 2, quantile, probs=0.025), ncol=N_train, nrow=K))
            sv_cih_train <- t(matrix(apply(post$surv.test[, 1:(N_train * K)], 2, quantile, probs=0.975), ncol=N_train, nrow=K))
            
            write.csv(as.data.frame(sv_mu_train), "{tmp_path / 'mu.csv'}", row.names=FALSE)
            write.csv(as.data.frame(sv_cil_train), "{tmp_path / 'cil.csv'}", row.names=FALSE)
            write.csv(as.data.frame(sv_cih_train), "{tmp_path / 'cih.csv'}", row.names=FALSE)
            
            # Extract cohort mean survival curve across draws for MCMC diagnostics
            sv_cohort <- matrix(0, nrow=nrow(post$surv.test), ncol=K)
            for (k in 1:K) {{
                cols_k <- ((1:N_train) - 1) * K + k
                sv_cohort[, k] <- rowMeans(post$surv.test[, cols_k, drop=FALSE])
            }}
            write.csv(as.data.frame(sv_cohort), "{tmp_path / 'sv_cohort_draws.csv'}", row.names=FALSE)
            
            # Extract test holdout predictions if present
            if (test_exists) {{
                test_start_col <- (N_train * K) + 1
                test_end_col <- (N_train + N_test) * K
                sv_mu_test <- t(matrix(post$surv.test.mean[test_start_col:test_end_col], ncol=N_test, nrow=K))
                sv_cil_test <- t(matrix(apply(post$surv.test[, test_start_col:test_end_col], 2, quantile, probs=0.025), ncol=N_test, nrow=K))
                sv_cih_test <- t(matrix(apply(post$surv.test[, test_start_col:test_end_col], 2, quantile, probs=0.975), ncol=N_test, nrow=K))
                
                write.csv(as.data.frame(sv_mu_test), "{tmp_path / 'mu_test.csv'}", row.names=FALSE)
                write.csv(as.data.frame(sv_cil_test), "{tmp_path / 'cil_test.csv'}", row.names=FALSE)
                write.csv(as.data.frame(sv_cih_test), "{tmp_path / 'cih_test.csv'}", row.names=FALSE)
            }}
            
            # Extract Subgroup Stratification (Top 25% vs Bottom 25%)
            if (file.exists("{csv_strat}")) {{
                strat_df <- read.csv("{csv_strat}")
                top_idx <- strat_df$top25[strat_df$top25 >= 0] + 1 # 1-indexed for R
                bot_idx <- strat_df$bot25[strat_df$bot25 >= 0] + 1
                
                top_mu <- numeric(K); top_cil <- numeric(K); top_cih <- numeric(K)
                bot_mu <- numeric(K); bot_cil <- numeric(K); bot_cih <- numeric(K)
                
                for (k in 1:K) {{
                    c_top <- (top_idx - 1) * K + k
                    c_bot <- (bot_idx - 1) * K + k
                    
                    d_top <- rowMeans(post$surv.test[, c_top, drop=FALSE])
                    d_bot <- rowMeans(post$surv.test[, c_bot, drop=FALSE])
                    
                    top_mu[k] <- mean(d_top)
                    top_cil[k] <- quantile(d_top, 0.025)
                    top_cih[k] <- quantile(d_top, 0.975)
                    
                    bot_mu[k] <- mean(d_bot)
                    bot_cil[k] <- quantile(d_bot, 0.025)
                    bot_cih[k] <- quantile(d_bot, 0.975)
                }}
                
                strat_out <- data.frame(
                    top25_mu = top_mu, top25_cil = top_cil, top25_cih = top_cih,
                    bot25_mu = bot_mu, bot25_cil = bot_cil, bot25_cih = bot_cih
                )
                write.csv(strat_out, "{tmp_path / 'strat_out.csv'}", row.names=FALSE)
            }}
            
            # Extract Causal RD(t50) and Marginal Hazard Ratio for each contrast
            if (causal_exists) {{
                k50 <- {t50_idx + 1} # 1-indexed interval for median time
                # Each contrast has 2 * N_train rows: N_train for x1, N_train for x0
                offset_base <- N_train + N_test
                n_contrasts <- N_causal / (2 * N_train)
                
                caus_res_list <- list()
                for (c_i in 1:n_contrasts) {{
                    c_start1 <- offset_base + (c_i - 1) * (2 * N_train) + 1
                    c_end1 <- c_start1 + N_train - 1
                    c_start0 <- c_end1 + 1
                    c_end0 <- c_start0 + N_train - 1
                    
                    # Marginal survival curves across all MCMC draws
                    nd <- nrow(post$surv.test)
                    s1_mat <- matrix(0, nrow=nd, ncol=K)
                    s0_mat <- matrix(0, nrow=nd, ncol=K)
                    for (k in 1:K) {{
                        cols1_k <- ((c_start1:c_end1) - 1) * K + k
                        cols0_k <- ((c_start0:c_end0) - 1) * K + k
                        s1_mat[, k] <- rowMeans(post$surv.test[, cols1_k, drop=FALSE])
                        s0_mat[, k] <- rowMeans(post$surv.test[, cols0_k, drop=FALSE])
                    }}
                    
                    # Interval hazards: h_k = (S_prev - S_curr) / S_prev
                    s1_prev <- cbind(1.0, s1_mat[, 1:(K-1), drop=FALSE])
                    s0_prev <- cbind(1.0, s0_mat[, 1:(K-1), drop=FALSE])
                    h1_mat <- pmin(pmax((s1_prev - s1_mat) / pmax(s1_prev, 1e-6), 1e-4), 1 - 1e-4)
                    h0_mat <- pmin(pmax((s0_prev - s0_mat) / pmax(s0_prev, 1e-6), 1e-4), 1 - 1e-4)
                    
                    # Interval Hazard Ratio via complementary log-log link: -log(1 - h1) / -log(1 - h0)
                    hr_mat <- pmin(pmax(-log(1 - h1_mat) / -log(1 - h0_mat), 0.02), 50.0)
                    
                    # Geometric mean summary HR across intervals per draw
                    log_hr_d <- rowMeans(log(hr_mat))
                    hr_d <- exp(log_hr_d)
                    
                    hr_m <- exp(mean(log_hr_d))
                    hr_cil <- as.numeric(quantile(hr_d, 0.025))
                    hr_cih <- as.numeric(quantile(hr_d, 0.975))
                    hr_w <- hr_cih - hr_cil
                    
                    # Risk difference at t50: S0(t50) - S1(t50) (excess event risk for 1 over 0)
                    d_rd <- s0_mat[, k50] - s1_mat[, k50]
                    rd_m <- mean(d_rd)
                    rd_cil <- as.numeric(quantile(d_rd, 0.025))
                    rd_cih <- as.numeric(quantile(d_rd, 0.975))
                    rd_w <- rd_cih - rd_cil
                    
                    caus_res_list[[c_i]] <- data.frame(
                        contrast = c_i,
                        hr_mean = hr_m,
                        hr_cil = hr_cil,
                        hr_cih = hr_cih,
                        hr_width = hr_w,
                        rd_mean = rd_m,
                        rd_cil = rd_cil,
                        rd_cih = rd_cih,
                        rd_width = rd_w
                    )
                }}
                write.csv(do.call(rbind, caus_res_list), "{tmp_path / 'causal_out.csv'}", row.names=FALSE)
            }}
        }}
        """
        r_script_file.write_text(r_code)
        
        # Execute R script
        start_fit_time = time.time()
        try:
            res = subprocess.run(["Rscript", str(r_script_file)], capture_output=True, text=True, timeout=300)
            fit_duration = time.time() - start_fit_time
            if res.returncode != 0:
                print(f"r-BART error: {res.stderr}")
                return run_rbart(train_data, skip_r=True, scenario_id=scenario_id)
                
            test_results = None
            strat_results = None
            causal_results = {}
            
            if csv_out.exists():
                out_df = pd.read_csv(csv_out)
                group_col = x_train[:, 0]
                unique_grps = sorted(np.unique(group_col))
                sv_mean = np.zeros((n_obs, len(eval_times)))
                ci_l = np.zeros((n_obs, len(eval_times)))
                ci_u = np.zeros((n_obs, len(eval_times)))
                
                for g_idx, g_val in enumerate(unique_grps):
                    sub_df = out_df[out_df["grp"] == g_val]
                    msk = (group_col == g_val)
                    if len(sub_df) >= len(eval_times):
                        sv_mean[msk, :] = sub_df["sv_mu"].values[:len(eval_times)]
                        ci_l[msk, :] = sub_df["sv_cil"].values[:len(eval_times)]
                        ci_u[msk, :] = sub_df["sv_cih"].values[:len(eval_times)]
            else:
                mu_f = tmp_path / "mu.csv"
                cil_f = tmp_path / "cil.csv"
                cih_f = tmp_path / "cih.csv"
                sv_mean = pd.read_csv(mu_f).to_numpy()[:, :len(eval_times)]
                ci_l = pd.read_csv(cil_f).to_numpy()[:, :len(eval_times)]
                ci_u = pd.read_csv(cih_f).to_numpy()[:, :len(eval_times)]
                
                mu_test_f = tmp_path / "mu_test.csv"
                cil_test_f = tmp_path / "cil_test.csv"
                cih_test_f = tmp_path / "cih_test.csv"
                if mu_test_f.exists():
                    test_results = {
                        "sv_mean": pd.read_csv(mu_test_f).to_numpy()[:, :len(eval_times)],
                        "ci_lower": pd.read_csv(cil_test_f).to_numpy()[:, :len(eval_times)],
                        "ci_upper": pd.read_csv(cih_test_f).to_numpy()[:, :len(eval_times)],
                    }
                    
                strat_f = tmp_path / "strat_out.csv"
                if strat_f.exists():
                    st_df = pd.read_csv(strat_f)
                    strat_results = {
                        "top25_mean": st_df["top25_mu"].values[:len(eval_times)],
                        "top25_ci_lower": st_df["top25_cil"].values[:len(eval_times)],
                        "top25_ci_upper": st_df["top25_cih"].values[:len(eval_times)],
                        "bot25_mean": st_df["bot25_mu"].values[:len(eval_times)],
                        "bot25_ci_lower": st_df["bot25_cil"].values[:len(eval_times)],
                        "bot25_ci_upper": st_df["bot25_cih"].values[:len(eval_times)],
                    }
                    
            caus_f = tmp_path / "causal_out.csv"
            if caus_f.exists() and causal_specs:
                c_df = pd.read_csv(caus_f)
                for c_idx, c_spec in enumerate(causal_specs):
                    var_idx = c_spec["variable"]
                    var_name = c_spec.get("name", f"x{var_idx}")
                    if c_idx < len(c_df):
                        row = c_df.iloc[c_idx]
                        hr_l = float(row["hr_cil"])
                        hr_u = float(row["hr_cih"])
                        rd_l = float(row["rd_cil"])
                        rd_u = float(row["rd_cih"])
                        causal_results[var_name] = {
                            "hazard_ratio_weighted_mean": float(row["hr_mean"]),
                            "hazard_ratio_weighted_ci": [hr_l, hr_u],
                            "hazard_ratio_ci_width": float(row["hr_width"]) if "hr_width" in row else (hr_u - hr_l),
                            "risk_difference_t50_mean": float(row["rd_mean"]),
                            "risk_difference_t50_ci_lower": rd_l,
                            "risk_difference_t50_ci_upper": rd_u,
                            "risk_difference_t50_ci_width": float(row["rd_width"]) if "rd_width" in row else (rd_u - rd_l),
                            "t50_time": float(eval_times[t50_idx]),
                        }
                
            # 9. MCMC Diagnostics (R-hat & ESS) for r-BART
            diag_f = tmp_path / "sv_cohort_draws.csv"
            rb_diagnostics = {}
            if diag_f.exists():
                try:
                    c_draws = pd.read_csv(diag_f).to_numpy()
                    mc_cores = 4
                    if c_draws.shape[0] >= mc_cores:
                        d_per_c = c_draws.shape[0] // mc_cores
                        if d_per_c >= 2:
                            import arviz as az
                            r_chains = c_draws[:mc_cores * d_per_c].reshape(mc_cores, d_per_c, -1)
                            rhat_arr = az.rhat(r_chains)
                            ess_arr = az.ess(r_chains)
                            rb_diagnostics = {
                                "rhat_mean": float(np.nanmean(rhat_arr)),
                                "rhat_max": float(np.nanmax(rhat_arr)),
                                "ess_mean": float(np.nanmean(ess_arr)),
                                "ess_min": float(np.nanmin(ess_arr)),
                            }
                except Exception:
                    rb_diagnostics = {"rhat_mean": None, "rhat_max": None, "ess_mean": None, "ess_min": None}
                
            total_duration = time.time() - start_total_time
            out_res = {
                "model": "rbart",
                "sv_mean": sv_mean,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "stratification_results": strat_results,
                "causal_effects": causal_results,
                "diagnostics": rb_diagnostics,
                "timings": {
                    "fit_seconds": fit_duration,
                    "total_seconds": total_duration
                }
            }
            if test_results is not None:
                out_res["test_results"] = test_results
            return out_res
        except Exception as e:
            print(f"Failed to execute Rscript: {e}. Falling back to historical lookup.")
            return run_rbart(train_data, skip_r=True, scenario_id=scenario_id)
