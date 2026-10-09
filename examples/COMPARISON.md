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
model means are estimated from **400 model draws per arm**, matching the
original binary ATE notebook. The training sample remains n=5,000. The
CSV records the requested and actual draw counts for both arms and
`model_draw_ate_se`, which measures sampling variability conditional on
the fitted model; it is not uncertainty from estimating the model.

## Overnight Run All

Restart the kernel and Run All in `examples/compare_frugal_flows.ipynb`.
The default `RUN_FULL=True` attempts 30 seeds at each of five instrument
strengths, fitting Frengression and the pinned official 10-D Frugal Flows
configuration to the same observational dataset. Both use 400 outcome
draws per arm. Optional pilots and sensitivity fits are off.

Each successful method is saved immediately. Failed attempts are logged
and the run continues; rerunning fits only missing methods. Cache checks
reject results with unknown or mismatched draw counts or fitting settings.
Old 10,000-draw results are not reused.

Outputs are saved outside Git under `~/frengression_results/binary_ate/`:

- `binary_ate_table_n5000_fr1000_ff1000_m400_mc400.csv`: RMSE, Bias and
  MAE in the existing paper table's order.
- `causl_full30_summary_n5000_fr1000_ff1000_m400_mc400.csv`: completed
  repetition counts, ATE and distribution scores.
- `causl_full30_paired_n5000_fr1000_ff1000_m400_mc400.csv`: differences
  on complete shared-seed pairs, with descriptive bootstrap intervals.
- `completion_*.csv`, `failures_*.csv`, per-method results and metadata:
  progress, failed attempts, sampling settings and package versions.

An independent replication batch can supply a Frugal Flows column in the
existing table when the DGP and evaluation protocol match. State that it
is an independent batch; paired comparisons apply only to the new fits
on matched seeds. Check completed counts before reporting 30 repetitions.
The full run may take longer than one night; successful fits are retained.

## Earlier five-seed results

The completed five-seed grid shows smaller Frengression ATE and
distribution errors in every reported instrument-strength row **against
the compact exploratory Frugal Flows fit**. That fit used copula width 64,
depth 2, 3 layers, learning rate 5e-4, and patience 25. The pinned
Frugal Flows authors' `benchmarking.py` instead defines a width-200,
depth-4, four-layer copula, learning rate 5e-3, and patience 100 for
10-dimensional settings. The current five-seed table therefore does not
establish a win over a sufficiently tuned Frugal Flows baseline.
The earlier custom three-covariate pilot is not part of this comparison.

The optional `RUN_OFFICIAL_FF_CHECK` and `RUN_GRID_PILOT` switches retain
the one-seed and five-seed official-configuration checks. They are off by
default. These checks also use the configured 400 draws per arm, so do
not compare their finite-draw scores directly with old 10,000-draw rows.

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
  --marginal-epochs 400 --mc 400 --methods frugal_flows_official \
  --output ~/frengression_results/binary_ate/causl_ffofficial_check_mc400.csv
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

## Joint-outcome overnight experiment

Restart the kernel and Run All in `examples/joint_outcomes_causl.ipynb`.
Defaults run a 1,000-update pilot and 30 paired replications of joint
Frengression and conditional engression plus g-computation. The latter
fits the bivariate outcome conditional on treatment and covariates, then
averages over the covariate distribution under fixed treatment. Both
methods generate paired outcomes together. Shuffled margins are a
dependence ablation, not a fitted competitor or a Frugal Flows result.

This separate distribution experiment keeps 5,000 generated pairs per
arm and `noise_dim=2`. The optional 4,000-update sensitivity fit is off.
Per-method draw checkpoints allow missing fits to resume. Reports and
figures are saved under `~/frengression_results/joint_outcomes_causl/`,
including moment, joint-event, joint-CDF and paired-error summaries, noise
checks, completion counts and failure logs. PNG/PDF/SVG figures compare
the exact law with fitted draws using common axes and density scales,
and show paired errors across datasets. The representative distribution
figure uses the first complete paired seed in the specified order,
without selecting it by accuracy.

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

