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
on the same data. It scores the *joint* intervention law (energy distance, mean
error, covariance error). The independent-margin reference cannot reproduce
cross-outcome dependence by construction; the comparison tests whether the joint
fit learns it. Label the Frugal Flows entry “not implemented in the released
scalar-outcome pipeline”; do not record it as a failed fit or as a performance
win. This is a scope demonstration, separate from the shared-setting comparison.

## Run

Create an environment with Python, PyTorch, `engression`, NumPy, SciPy, JAX,
FlowJAX, Equinox and Optax, then install this repository and the official
`llaurabatt/frugal-flows` repository in editable mode. Run from `examples/`:

```bash
python compare_frugal_flows.py --n 200 --repeats 1 --fr-iters 20 \
  --flow-epochs 20 --marginal-epochs 20 --mc 100 --truth-mc 1000 \
  --output smoke.csv

python compare_frugal_flows.py --n 2000 --repeats 5 --multivariate \
  --output comparison_results.csv
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
