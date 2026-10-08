# Frugal Flows comparison on the paper's binary ATE experiment

Open `examples/compare_frugal_flows.ipynb` in VS Code. The notebook calls
`compare_frugal_flows_causl.py`, which adds Frugal Flows to the **existing**
binary experiment in `xwshen51/frengression/paper_exp/binary.ipynb`.
The included `causl_binary.R` retains its `data.causl` R function and
parameters: five instruments, five confounders, no other covariates,
binary treatment, n=5,000, ATE=2, and instrument strengths 0–2 by 0.5.
Frengression uses the paper's three layers, hidden width 100, noise dimension
1, learning rate 1e-4, and 1,000 updates. Both methods receive the same
observational draw for every seed and instrument strength.

Install R and the `causl` R package before running the notebook:

```r
install.packages("remotes")
remotes::install_github("rje42/causl")
```

Select a Python kernel for this checkout. The notebook setup cell checks R and
installs missing Python packages, including Frugal Flows with its documented
`flowjax==19.1.0` dependency. Restart the kernel after dependency changes.

The causal outcome margins specified in the R DGP are N(0,1) and N(2,1).
The code uses their exact ATE, means, and analytic Gaussian quantiles and
distribution distances. It never estimates truth by drawing a separate
intervention sample. The fitted models still require finite model draws to
estimate their own means, so report `model_draw_ate_se` and use a sufficiently
large `mc` (10,000 by default). A one-seed pilot runs by default; set
`RUN_FR_DIAGNOSTIC=True` for an optional 2,000-update Frengression fit on
the same data. Set `RUN_FULL=True` only after checking the pilot: it runs
the original five-level instrument-strength grid with 30 repetitions per
level for both methods and can take considerable time.

Command-line equivalent for one strength and one seed:

```bash
python examples/compare_frugal_flows_causl.py --n 5000 --repeats 1 \
  --strength-instr 0 --fr-iters 1000 --flow-epochs 2000 \
  --marginal-epochs 400 --mc 10000 \
  --output examples/benchmark_outputs/causl_pilot.csv
```

The `seconds` column is diagnostic: Frugal Flows fits covariate marginals
and its causal flow, whereas Frengression here fits only its f/h components.
For publication, report signed ATE bias, RMSE, MAE, marginal distribution
metrics, and variation over seeds at each instrument strength. Retain
competitor wins. Do not present the earlier pilot with 3 covariates,
`noise_dim=8`, and Monte Carlo truth as a paper comparison.

The older `compare_frugal_flows.py` remains available as an exploratory
custom-DGP example and provides the shared Frugal Flows fitting function.
Its optional two-outcome experiment compares joint Frengression against
products of separately fitted scalar margins; this is not a joint Frugal
Flows baseline. A separate joint-outcome experiment and a capable joint
baseline are needed for a multivariate superiority claim.

## Existing continuous-treatment figure

`adrf_replicate_bands.py` accepts a CSV containing `method,run,x,mean_estimate`
and optionally `true_mean`. If each original simulation's estimated ADRF was
saved, it can produce bands from the across-run distribution of estimated
*means* for both methods without retraining CausalEGM:

```bash
python adrf_replicate_bands.py saved_adrf_predictions.csv \
  --output adrf_bands.csv --figure adrf_bands.png
```

These descriptive percentiles are not confidence intervals. If the original
per-run CausalEGM predictions were not saved, there is no sound way to recover
them from the published aggregate figure; use mean curves without comparable
bands or rerun that baseline to regenerate the inputs.
