# BART Survival Analysis Validation Suite: Experiment Design & Replication Guide

Comprehensive technical reference, scenario specifications, mathematical derivations, and replication instructions for the Bayesian Additive Regression Trees (BART) discrete-time survival analysis validation suite.

---

## 1. Executive Summary & Purpose

This validation suite evaluates and benchmarks **`SurvBART`**—the modern Python implementation of discrete-time Bayesian Additive Regression Trees for survival analysis (`bart-survival` package)—against:
1. **Traditional Frequentist Estimators**:
   * **Kaplan-Meier (KM)**: Univariate baseline survival estimator (via `lifelines.KaplanMeierFitter`).
   * **Cox Proportional Hazards Model (CPHM)**: Semi-parametric regression standard (via `lifelines.CoxPHFitter`).
2. **Gold-Standard Bayesian Reference**:
   * **`r-BART`**: The established R implementation (`BART::surv.bart` / `BART::mc.surv.bart`).

### Core Research Questions Evaluated:
* **Calibration & Censoring Resilience**: Can `p-BART` accurately recover underlying baseline survival distributions under mild ($20\%$) and heavy ($50\%$) administrative and random right-censoring?
* **Non-Proportional Hazards**: How robust is `p-BART` when the proportional hazards assumption is severely violated (crossing survival curves)?
* **Out-of-Sample Generalization & Overfitting**: Under complex nonlinear interactions and varying shape parameters, does `p-BART` regularize effectively and generalize to unseen test cohorts across increasing sample sizes ($N \in [400, 1000, 4000]$)?
* **Causal Counterfactual Inferences**: Can the model reliably estimate causal hazard ratios ($\text{HR}$) and marginal risk differences at median survival ($\text{RD}(t_{50})$)?
* **MCMC Convergence & Computational Scaling**: Does `p-BART` maintain valid posterior mixing ($\hat{R} \le 1.02$, $\text{ESS} \ge 400$) and scale efficiently relative to R?

---

## 2. Theoretical Framework & Data Generating Processes (DGPs)

Survival times are simulated from continuous Weibull distributions parameterized by a scale function $\lambda(X)$ and shape parameter $\alpha(X)$:
$$S(t \mid X) = \exp\left( - \left(\frac{t}{\lambda(X)}\right)^{\alpha(X)} \right), \quad t \ge 0$$
Individual continuous survival times $T_i$ are drawn via the inverse probability transform:
$$T_i = \lambda(X_i) \cdot \left( -\ln(U_i) \right)^{1 / \alpha(X_i)}, \quad U_i \sim \text{Uniform}(0, 1)$$

Censoring times $C_i$ are drawn from an exponential distribution with rate parameter calibrated by `cens_scale`:
$$C_i \sim \text{Exponential}\left(\frac{1}{\text{cens\_scale}}\right)$$
The observed follow-up time is $Y_i = \min(T_i, C_i, \text{eos})$, with event indicator $\delta_i = \mathbf{1}(T_i \le \min(C_i, \text{eos}))$. The end-of-study horizon is set to $\text{eos} = 180$ days.

---

### Scenario Matrix

| Scenario Key | Category | Sample Sizes ($N$) | Censoring | Covariates ($X$) | Scale Function $\lambda(X)$ | Shape Function $\alpha(X)$ | Core Benchmark Goal |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **`simple_1_1`** | Null Calibration | 200, 400 | 20% | 1 constant ($x_0 = 1$) | $7$ | $3$ (constant) | Baseline survival calibration |
| **`simple_1_2`** | Null Calibration | 200, 400 | 50% | 1 constant ($x_0 = 1$) | $7$ | $3$ (constant) | Robustness under heavy censoring |
| **`simple_2_1`** | Two-Sample | 200, 400 | 20% | 1 binary ($x_0 \sim \text{Bern}(0.5)$) | $12 - 0.5 x_0$ | $1.2 + 1.0 x_0$ | Binary subgroup separation |
| **`simple_2_2`** | Two-Sample | 200, 400 | 50% | 1 binary ($x_0 \sim \text{Bern}(0.5)$) | $12 - 0.5 x_0$ | $1.2 + 1.0 x_0$ | Binary separation under heavy censoring |
| **`complex_1`** | Multivariable PH | 200, 400 | 20% | 9 binary ($x_0 \dots x_8 \sim \text{Bern}(0.5)$) | $\exp(3 + 0.1 \sum_{j=0}^5 x_j + 1.0 x_6)$ | $2$ (constant) | Proportional hazards with multiple linear covariates |
| **`complex_2`** | Multivariable Non-PH | 200, 400 | 20% | 9 binary ($x_0 \dots x_8 \sim \text{Bern}(0.5)$) | $20 + 5(\sum_{j=0}^5 x_j + 10 x_6)$ | $0.7 + 1.3 x_6$ | Severe non-proportional hazards (crossing survival) |
| **`complex_3`** | Complex Nonlinear | 400, 1000, 4000 | 20% | 2 binary, 8 continuous $U(0, 1)$ | $50 + 100 x_0 - 100 x_0 x_1 + 10 x_2 + 20 x_3^2 + 20 x_3 x_4$ | $1 + 2 x_0 + x_0 x_1 + 0.1 x_1 x_4^2$ | Nonlinear interactions & out-of-sample holdout test |

---

### Detailed Scenario Formulations

#### 1. Scenario 1: One-Sample Null (`simple_1_1` & `simple_1_2`)
* **Purpose**: Verifies that discrete-time BART correctly recovers the parametric survival baseline without introducing artificial shrinkage or overfitting.
* **Evaluation Quantiles**: Evaluated at the 5 survival deciles: $[0.90, 0.75, 0.50, 0.25, 0.10]$.
* **Models Tested**: `p-BART`, `r-BART`, `Kaplan-Meier`.

#### 2. Scenario 2: Two-Sample Binary Exposure (`simple_2_1` & `simple_2_2`)
* **Purpose**: Tests estimation of group-specific survival curves when $x_0 \in \{0, 1\}$ modifies both the scale and shape of the Weibull distribution (creating non-proportional hazards between two arms).
* **Models Tested**: `p-BART`, `r-BART`, `Stratified Kaplan-Meier`.

#### 3. Scenario 3: Complex Multivariate Linear Proportional Hazards (`complex_1`)
* **Covariates**: 9 binary variables $x_0, \dots, x_8 \sim \text{Bernoulli}(0.5)$.
* **Scale**: $\lambda(X) = \exp(3 + 0.10(x_0 + x_1 + x_2 + x_3 + x_4 + x_5) + 1.0 x_6)$. Covariates $x_7, x_8$ are pure noise.
* **Shape**: Constant $\alpha = 2.0$. Because $\alpha$ is constant across all individuals, **proportional hazards strictly holds**:
  $$\frac{h(t \mid X)}{h(t \mid X')} = \exp\left( -2(\log\lambda(X) - \log\lambda(X')) \right)$$
* **Causal Contrasts**:
  * $x_0$ main effect: True Hazard Ratio $\text{HR} = \exp(-2 \times 0.10) = 0.819$
  * $x_6$ strong effect: True Hazard Ratio $\text{HR} = \exp(-2 \times 1.00) = 0.135$
* **Models Tested**: `p-BART`, `r-BART`, `CoxPH`.

#### 4. Scenario 4: Complex Multivariate Non-Proportional Hazards (`complex_2`)
* **Covariates**: 9 binary variables.
* **Scale**: $\lambda(X) = 20 + 5(x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + 10 x_6)$.
* **Shape**: $\alpha(X) = 0.7 + 1.3 x_6$.
* **Mechanism**: When $x_6 = 0$, $\alpha = 0.7$ (hazard decays over time). When $x_6 = 1$, $\alpha = 2.0$ (hazard increases linearly over time). The hazard functions cross over time, causing **Cox PH to fail dramatically** (a-RMSE $\sim 0.092$ overall, peaking at $0.141$ at $t_{25}$). Both `p-BART` and `r-BART` maintain low error ($\sim 0.061$).

#### 5. Scenario 5: Complex Multivariable Regression (`complex_3`)
* **Covariates**: 10 variables:
  * $x_0, x_1 \sim \text{Bernoulli}(0.5)$ (binary modifiers)
  * $x_2 \dots x_9 \sim \text{Uniform}(0, 1)$ (continuous predictors; $x_5 \dots x_9$ are pure noise)
* **Scale Interaction**:
  $$\lambda(X) = 50 + 100 x_0 - 100 x_0 x_1 + 10 x_2 + 20 x_3^2 + 20 x_3 x_4$$
  The $-100 x_0 x_1$ term turns off the effect of $x_0$, inducing a 3-fold scale drop ($150 \rightarrow 50$) when $x_1$ is active.
* **Shape Function**:
  $$\alpha(X) = 1 + 2 x_0 + 1 x_0 x_1 + 0.1 x_1 x_4^2$$
  Varies from $\alpha = 1.0$ (exponential decay) up to $\alpha = 4.0$ (steep sigmoid cliff).
* **Out-of-Sample Holdout Cohort**: For each iteration, an independent test cohort of size $N_{\text{test}} = 400$ is drawn to test generalizability across training sample sizes $N \in [400, 1000, 4000]$.

---

## 3. Discretization & Time-Grid Alignment

BART-Survival models discrete-time interval hazards:
$$\lambda_j(X) = P(T \in [t_{j-1}, t_j) \mid T \ge t_{j-1}, X)$$
The cumulative survival probability is recovered via:
$$S(t_k \mid X) = \prod_{j=1}^k (1 - \lambda_j(X))$$

### Critical Requirement: Train-Test Grid Alignment
When testing on out-of-sample holdout cohorts, the evaluation cutpoints must be pre-specified from the training distribution:
```python
# 1. Training cohort defines discrete evaluation days
train_data = generate_scenario_data(scenario_cfg, n=n, seed=seed)
eval_times = train_data["eval_times"]

# 2. Test cohort MUST use identical evaluation days
test_data = generate_scenario_data(
    scenario_cfg,
    n=test_n,
    seed=seed + 10000,
    eval_times=eval_times  # Pre-specified grid
)
```
> [!IMPORTANT]
> If the test cohort computes its cutpoints independently based on its own sample maximum event time, a 2–4 day horizontal offset between the predicted days and ground-truth days will artificially inflate the calculated RMSE around the steep median-survival slope by up to $10\%$. Always pass `eval_times=train_data['eval_times']`.

---

## 4. Evaluation Metrics & Mathematical Definitions

All metrics are computed point-wise at each discrete interval $j \in \{1, \dots, K\}$ and aggregated across all $N$ subjects and $M$ Monte Carlo iterations.

### 1. Average Root Mean Squared Error (a-RMSE)
Measures calibrated distance between estimated posterior mean survival $\hat{S}_i(t_j)$ and analytical ground truth $S_i(t_j)$:
$$\text{RMSE}_j = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \hat{S}_i(t_j) - S_i(t_j) \right)^2}$$
$$\text{a-RMSE} = \sqrt{\frac{1}{N \cdot K} \sum_{i=1}^N \sum_{j=1}^K \left( \hat{S}_i(t_j) - S_i(t_j) \right)^2}$$

### 2. Pointwise Signed Calibration Bias
Measures systematic over- or under-estimation:
$$\text{Bias}_j = \frac{1}{N} \sum_{i=1}^N \left( \hat{S}_i(t_j) - S_i(t_j) \right)$$

### 3. Empirical 95% Credible Interval Coverage
Proportion of subjects whose true analytical survival curve falls within the model's 95% equal-tailed posterior interval $[L_{95, i}(t_j), U_{95, i}(t_j)]$:
$$\text{Coverage}_j = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left( L_{95, i}(t_j) \le S_i(t_j) \le U_{95, i}(t_j) \right)$$
Nominal target is exactly **$95.0\%$**.

### 4. Uncertainty Interval Width (UIW / Precision)
Measures posterior sharpness (narrower width indicates higher precision):
$$\text{UIW}_j = \frac{1}{N} \sum_{i=1}^N \left( U_{95, i}(t_j) - L_{95, i}(t_j) \right)$$

### 5. Causal Risk Difference at Median Survival ($\text{RD}(t_{50})$)
For an exposure $A \in \{0, 1\}$, measures marginal counterfactual survival difference at the population median survival time $t_{50}$:
$$\text{RD}(t_{50}) = \frac{1}{N} \sum_{i=1}^N \left[ S(t_{50} \mid X_i, A=0) - S(t_{50} \mid X_i, A=1) \right]$$

### 6. Causal Hazard Ratio ($\text{HR}$)
Computed by contrasting marginal counterfactual interval hazards $\lambda_j(1)$ vs $\lambda_j(0)$, weighted by interval event incidence $w_j$:
$$\text{HR}_{\text{weighted}} = \exp\left( \sum_{j=1}^K w_j \ln\left[ \frac{-\ln(1 - \lambda_j(1))}{-\ln(1 - \lambda_j(0))} \right] \right)$$

### 7. MCMC Diagnostics
* **Gelman-Rubin $\hat{R}$**: Monitored across 4 parallel chains. Target: $\hat{R} \le 1.02$.
* **Effective Sample Size ($\text{ESS}$)**: Bulk and tail effective sample sizes across posterior draws. Target: $\text{ESS} \ge 400$.

---

## 5. Directory Structure & Key Files

```text
bart_survival_paper_1000/
├── figures/                              # Rendered publication PNG figures (Fig 1, 2, 3, etc.)
├── results/
│   ├── suite_multi_N_100iter_standard/   # Master 100-replicate validation results across all tiers
│   │   ├── simple_1_1_N200_standard/
│   │   ├── ...
│   │   ├── complex_3_N4000_standard/
│   │   └── multi_N_100iter_benchmark_results.xlsx # Consolidated multi-tab workbook
│   └── master_benchmark_results.xlsx     # Overall publication summary workbook
├── src/
│   ├── config/
│   │   ├── models.py                     # Standard, fast, and high-precision hyperparameter presets
│   │   └── scenarios.py                  # Declarative DGP definitions (equations, sample sizes, quantiles)
│   ├── validation/
│   │   ├── dgp.py                        # Simulation generator and interval assigner
│   │   ├── metrics.py                    # a-RMSE, signed bias, 95% coverage, interval width
│   │   ├── tracker.py                    # JSON/Parquet replicate logger and summary aggregator
│   │   └── runners/
│   │       ├── pbart_runner.py           # SurvBART execution, test holdout, causal effects
│   │       ├── rbart_runner.py           # R BART::mc.surv.bart subprocess execution
│   │       ├── cox_runner.py             # lifelines.CoxPHFitter execution
│   │       └── km_runner.py              # lifelines.KaplanMeierFitter execution
│   ├── tools/
│   │   ├── generate_master_excel.py      # Compiles raw replicate logs into publication-ready Excel
│   │   ├── recreate_paper_figures.py     # Generates Figure 1, 2, and 3 from raw benchmark outputs
│   │   └── run_draw_threshold_study.py   # MCMC draw sensitivity study
│   └── run_benchmark.py                  # Master CLI experiment runner
└── VALIDATION_EXPERIMENTS_GUIDE.md       # This comprehensive technical guide
```

---

## 6. How to Re-Implement & Run Experiments

### Prerequisites & Environment Setup

#### Python Environment
```bash
cd /Users/jacobtiegs/projects/bart_val_online
source .venv/bin/activate
pip install -e BART-Survival/
pip install lifelines openpyxl pandas numpy matplotlib scipy pymc
```

#### R Environment (for live `r-BART` execution)
Ensure R and the `BART` library are installed:
```bash
Rscript -e "install.packages('BART', repos='https://cloud.r-project.org')"
```

---

### Command-Line Usage

#### 1. Quick Smoke Test (Fast Verification, 2 Iterations)
```bash
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/run_benchmark.py \
    --scenarios simple_1_1 complex_1 \
    --iters 2 \
    --preset fast \
    --skip-r \
    --experiment-name test_smoke
```

#### 2. Running Complex 3 with All 3 Models (`p-BART`, `r-BART`, `CoxPH`)
```bash
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/run_benchmark.py \
    --scenarios complex_3 \
    --all-n \
    --iters 10 \
    --preset standard \
    --experiment-name suite_complex3_10iter
```

#### 3. Running the Full 100-Iteration Validation Suite (p-BART)
```bash
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/run_benchmark.py \
    --scenarios all \
    --all-n \
    --iters 100 \
    --preset standard \
    --skip-r \
    --experiment-name suite_multi_N_100iter_standard
```

---

### Generating Tables & Publication Figures

#### 1. Compile Master Multi-Tab Excel Workbook
```bash
# Multi-N benchmark workbook (includes Complex 3 3-model comparison across N=400, 1000, 4000):
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/tools/generate_multi_N_excel.py

# Full scenario-by-scenario multi-tab report:
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/tools/generate_master_excel.py \
    --experiment-name suite_multi_N_100iter_standard \
    --output-filename multi_N_100iter_benchmark_results.xlsx
```

#### 2. Re-Render All Publication Figures (Figure 1, 2, 3)
```bash
MPLCONFIGDIR=/tmp .venv/bin/python bart_survival_paper_1000/src/tools/recreate_paper_figures.py \
    --experiment-name suite_multi_N_100iter_standard
```
Figures are output directly to `bart_survival_paper_1000/figures/`:
* **`fig1_paper_simple1.png` / `fig1_paper_simple2.png`**: Calibration and bias for 1-sample and 2-sample null scenarios across KM, p-BART, and r-BART.
* **`fig2_paper_complex1_prop.png` / `fig2_paper_complex2_nonprop.png`**: Proportional vs. non-proportional hazards comparison against Cox PH.
* **`fig3_paper_complex3_holdout.png`**: Complex 3 out-of-sample test RMSE and bias scaling across $N \in [400, 1000, 4000]$.

---

## 7. Empirical Validation Results & Summary Table

### Complex 3: Hold-out Generalization & Proportional Hazards Violation

Under the non-proportional Weibull DGP ($\alpha(X) = 1 + 2 x_0 + x_0 x_1 + 0.1 x_1 x_4^2$), crossing hazard curves break the fundamental assumptions of Cox Proportional Hazards. While Cox PH error remains flat at $\sim 14.3\% - 15.1\%$ regardless of sample size, both Bayesian Additive Regression Tree implementations (`p-BART` and `r-BART`) adapt to the non-linear interaction surface and non-proportional shape, reducing out-of-sample error monotonically down to $\sim 0.037$:

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

