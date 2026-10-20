# BART Survival Analysis Validation Suite (2026 Edition)

Welcome to the modernized validation suite for **`SurvBART`** (`bart-survival`), benchmarking the Python implementation against traditional statistical methods (Kaplan-Meier, Cox Proportional Hazards) and the reference R `BART` package (`BART::surv.bart`).

## Quick Navigation

* **Comprehensive Experiment Guide & Technical Manual**: See [`VALIDATION_EXPERIMENTS_GUIDE.md`](./VALIDATION_EXPERIMENTS_GUIDE.md) for full DGP equations, metrics, mathematical derivations, and replication instructions.
* **Results Workbooks**:
  * Consolidated multi-sample benchmark: [`results/suite_multi_N_100iter_standard/multi_N_100iter_benchmark_results.xlsx`](./results/suite_multi_N_100iter_standard/multi_N_100iter_benchmark_results.xlsx)
  * Comprehensive master benchmark: [`results/master_benchmark_results.xlsx`](./results/master_benchmark_results.xlsx)
* **Publication Figures**: All recreated publication figures are generated in [`figures/`](./figures/).

---

## Benchmark Highlights (Complex 3: Generalization & PH Violation)

Under severe non-proportional Weibull hazards ($\alpha(X) = 1 + 2 x_0 + x_0 x_1 + 0.1 x_1 x_4^2$), crossing hazard curves break the fundamental assumptions of Cox Proportional Hazards. While Cox PH error remains flat at $\sim 14.3\% - 15.1\%$ regardless of sample size, both Bayesian Additive Regression Tree implementations (`p-BART` and `r-BART`) adapt to the non-linear interaction surface, reducing out-of-sample error monotonically down to $\sim 0.037$:

| Training Sample Size ($N$) | Estimator | Replicates | Train a-RMSE | Test a-RMSE (Holdout) | Test Signed Bias | 95% Coverage | Credible Interval Width | Fit Time | Speedup vs. R |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 400$** | **`p-BART (2026)`** | 10 | $0.0863$ | **$0.0854$** | $+0.0091$ | $91.7\%$ | $0.3532$ | **$4.63\,\text{s}$** | **$2.2\times$** |
| $N = 400$ | `r-BART (R::BART)` | 10 | $0.0838$ | **$0.0836$** | $+0.0087$ | $90.8\%$ | $0.3414$ | $10.09\,\text{s}$ | $1.0\times$ |
| $N = 400$ | `Cox PH (lifelines)` | 10 | $0.1483$ | **$0.1507$** | $-0.0050$ | N/A | N/A | $0.02\,\text{s}$ | — |
| **$N = 1000$** | **`p-BART (2026)`** | 10 | $0.0615$ | **$0.0607$** | $+0.0032$ | $92.1\%$ | $0.2652$ | **$6.33\,\text{s}$** | **$4.2\times$** |
| $N = 1000$ | `r-BART (R::BART)` | 10 | $0.0601$ | **$0.0597$** | $+0.0029$ | $91.2\%$ | $0.2544$ | $26.60\,\text{s}$ | $1.0\times$ |
| $N = 1000$ | `Cox PH (lifelines)` | 10 | $0.1436$ | **$0.1445$** | $-0.0061$ | N/A | N/A | $0.03\,\text{s}$ | — |
| **$N = 4000$** | **`p-BART (2026)`** | 10 | $0.0373$ | **$0.0370$** | $+0.0077$ | $90.9\%$ | $0.1477$ | **$13.02\,\text{s}$** | **$9.2\times$** |
| $N = 4000$ | `r-BART (R::BART)` | 10 | $0.0361$ | **$0.0366$** | $+0.0077$ | $91.0\%$ | $0.1407$ | $120.24\,\text{s}$ | $1.0\times$ |
| $N = 4000$ | `Cox PH (lifelines)` | 10 | $0.1418$ | **$0.1430$** | $-0.0023$ | N/A | N/A | $0.03\,\text{s}$ | — |

---

## Core Scenarios

1. **`simple_1_1` & `simple_1_2`**: One-Sample Null Baseline (20% & 50% Censoring)
2. **`simple_2_1` & `simple_2_2`**: Two-Sample Binary Exposure Calibration (20% & 50% Censoring)
3. **`complex_1`**: Multivariable Linear Proportional Hazards (9 Binary Predictors)
4. **`complex_2`**: Multivariable Severe Non-Proportional Hazards (Crossing Survival Curves)
5. **`complex_3`**: Multivariable Regression with Nonlinear Interactions, U-Shaped Effects, and Out-of-Sample Holdout Testing ($N \in [400, 1000, 4000]$)

---

## Quick Replication Commands

```bash
# 1. Run Complex 3 across all sample sizes (p-BART, r-BART, Cox PH):
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/run_benchmark.py \
    --scenarios complex_3 \
    --all-n \
    --iters 10 \
    --preset standard \
    --experiment-name suite_complex3_10iter

# 2. Compile Excel Results:
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/tools/generate_multi_N_excel.py

# 3. Re-render publication figures (Fig 1, 2, 3):
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/tools/recreate_paper_figures.py \
    --experiment-name suite_multi_N_100iter_standard
```
