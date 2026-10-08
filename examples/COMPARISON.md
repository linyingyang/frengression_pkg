# Focused comparison for the JMLR manuscript

For a VS Code/Jupyter workflow, open `examples/compare_frugal_flows.ipynb`,
select the Python environment, and choose **Run All**. The notebook installs
missing packages, runs a small wiring check and then the full comparison by
default; set `RUN_FULL = False` in its settings cell to stop after the quick
check. Results are saved to `examples/benchmark_outputs/`.

`compare_frugal_flows.py` fits both released implementations on exactly the
same observational draws. Treatment is binary and the scalar outcome has a
non-Gaussian intervention distribution. Frugal Flows is fitted with its
`flexible_continuous` margin, not its more restrictive Gaussian margin. The
reference intervention samples come from a separate draw of the known DGP.

Report the mean and standard deviation over independent seeds of: absolute ATE
error, average per-arm Wasserstein distance, average per-arm energy distance,
and average 10th/50th/90th quantile error. The fit time is diagnostic only:
Frugal Flows includes marginal-CDF fitting, while the Frengression timing covers
the `f,h` fit and excludes its observational-past generator `g`.

The optional `--multivariate` arm fits Frengression jointly to a two-dimensional
outcome and compares it to two independently fitted scalar Frengression models
on the same data. Add `--multivariate-ff` to fit two scalar Frugal Flows on
those data as another independent-margin reference (two extra fits per seed).
All rows score joint energy, mean error, covariance error, cross-covariance
error, and average marginal Wasserstein distance. Check marginal error before
interpreting a joint gap: it may reflect scalar fit differences as well as
missing dependence. A product of scalar margins cannot reproduce cross-outcome
dependence by construction. **This is not a joint Frugal Flows model** and is
not a numerical victory over one. A capable joint-outcome baseline would be
needed to argue broad multivariate superiority.

## Run

Create an environment with Python, PyTorch, `engression`, NumPy, SciPy, JAX,
FlowJAX, Equinox and Optax, then install this repository and the official
`llaurabatt/frugal-flows` repository in editable mode. Run from `examples/`:
Frugal Flows' `environment.yaml` pins `flowjax==19.1.0` and
`lineax==0.1.0`; older FlowJAX builds lack the `fit_to_data(data=...)` API
its current code uses. The notebook setup checks and installs those versions
in its selected kernel. Restart the kernel after changing package versions.

```bash
python compare_frugal_flows.py --n 200 --repeats 1 --fr-iters 20 \
  --flow-epochs 20 --marginal-epochs 20 --mc 100 --truth-mc 1000 \
  --output smoke.csv

python compare_frugal_flows.py --n 2000 --repeats 5 --multivariate \
  --output comparison_results.csv

# Optional: substantially more training, including two extra scalar flows per seed.
python compare_frugal_flows.py --n 2000 --repeats 5 --multivariate \
  --multivariate-ff --output comparison_with_ff_margins.csv
```

The first command only checks the end-to-end wiring. The second is a proposed
benchmark, not an already completed experiment. Inspect fit diagnostics and
convergence on a pilot before fixing final epochs and seeds, and report the
hardware and package commits. Do not add numerical conclusions to the paper
until the results exist.

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
