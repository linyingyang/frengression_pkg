# Frugal Flows comparison on the paper's binary ATE experiment

Open `examples/compare_frugal_flows.ipynb` in VS Code. It calls
`compare_frugal_flows_causl.py` to fit both methods to the **existing**
binary ATE experiment in `xwshen51/frengression/paper_exp/binary.ipynb`.
The included `causl_binary.R` retains the original `data.causl` function:
five instruments, five confounders, no other covariates, binary treatment,
n=5,000, true ATE=2, and instrument strengths 0–2 in steps of 0.5.
Frengression uses three layers, hidden width 100, noise dimension 1,
learning rate 1e-4, and 1,000 updates. Each paired fit sees the same
observational draw.

Install R and the `causl` package before running the notebook:

```r
install.packages("remotes")
remotes::install_github("rje42/causl")
```

Select a Python kernel for this checkout. The setup cell checks R and
installs missing Python packages, including `flowjax==19.1.0` and the
pinned Frugal Flows GitHub revision. Restart the kernel after dependency
changes.

The DGP specifies causal outcome margins N(0,1) and N(2,1). The code uses
their exact ATE, means, and analytic Gaussian distances and quantiles; it
does not estimate truth from a Monte Carlo intervention sample. Fitted
model means are still estimated from 10,000 model draws per arm, and the
CSV includes `model_draw_ate_se` to measure this sampling variability.

## Reading the five-seed results

The completed five-seed grid shows smaller Frengression ATE and
distribution errors in every reported instrument-strength row **against
the compact exploratory Frugal Flows fit**. That fit used copula width 64,
depth 2, 3 layers, learning rate 5e-4, and patience 25. The pinned
Frugal Flows authors' `benchmarking.py` instead defines a width-200,
depth-4, four-layer copula, learning rate 5e-3, and patience 100 for
10-dimensional settings. The current five-seed table therefore does not
establish a win over a sufficiently tuned Frugal Flows baseline.
The earlier custom three-covariate pilot is not part of this comparison.

The notebook now defaults to **one Frugal Flows only** official-configuration
check (`RUN_OFFICIAL_FF_CHECK=True`) at strength 2.0, seed 2026.
This saves a new CSV and does not rerun the Frengression fits or overwrite
the compact results. Compare that row with the **same seed and strength**
in the compact table, not the five-seed mean. Then set
`RUN_GRID_PILOT=True` for five paired seeds at each strength using the
official configuration, and eventually `RUN_FULL=True` for 30 seeds
at each strength. Those longer runs are off by default and can take
substantial time. Grid files are saved after each seed and reused on
rerun; the 30-seed loop currently writes each completed strength, so
plan its runtime before enabling it.

The official configuration is a documented starting point, not proof
that a single fit is optimal. The upstream comment also recommends
averaging at least five restarts for the causal margin. If the official
single fit is still poor, inspect fit diagnostics, transformation and
restart sensitivity before making a superiority claim. Prespecify the
final fitting protocol and retain all seeds and settings, including
competitor wins.

The equivalent one-seed command for the official configuration is:

```bash
python examples/compare_frugal_flows_causl.py --n 5000 --repeats 1 \
  --strength-instr 2 --seed 2026 --fr-iters 1000 --flow-epochs 1000 \
  --marginal-epochs 400 --mc 10000 --methods frugal_flows_official \
  --output examples/benchmark_outputs/causl_ffofficial_check.csv
```

The `seconds` column is diagnostic: Frugal Flows fits covariate
marginals and its causal flow, whereas this Frengression timing covers
the f/h fit. Report signed ATE bias, MAE, RMSE, distribution metrics,
seed variability, package commits and hardware. Keep the exploratory
and official results clearly labeled.

The older `compare_frugal_flows.py` remains an exploratory custom-DGP
example and provides the shared compact Frugal Flows fitting function.
Its optional two-outcome experiment compares joint Frengression with
products of separately fitted scalar margins, including two independent
Frugal Flows if enabled. This is not a joint Frugal Flows baseline;
a capable joint comparator is needed for a broad multivariate claim.

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
