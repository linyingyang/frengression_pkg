"""Reproducible shared-setting Frengression / Frugal Flows comparison.

Both methods fit the same simulated observational data with a binary treatment.
The scalar experiment scores the *whole* fitted interventional distribution,
not just the ATE. The optional two-outcome experiment compares one joint
Frengression fit with two independent one-outcome Frengression fits: the released
FrugalFlowModel pipeline has a scalar outcome margin.

Run with the frengression and frugal-flows repositories installed. No result
is included in this script; training must be run before reporting numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import expit
from scipy.spatial.distance import cdist, pdist
from scipy.stats import energy_distance, wasserstein_distance

# When run directly from a checkout, use its src-layout package even if the
# notebook kernel has not registered an editable install yet.
_source_root = Path(__file__).resolve().parents[1] / "src"
if (_source_root / "frengression" / "__init__.py").is_file():
    sys.path.insert(0, str(_source_root))


def simulate(n: int, seed: int, intervention: Optional[int] = None, dim: int = 1):
    """A confounded observational law with a known Monte Carlo intervention law.

    If ``intervention`` is set, X is fixed *after* drawing Z; the potential
    outcome noise remains independent of the treatment-assignment mechanism.
    The nonlinear Z term makes the scalar outcome margin non-Gaussian.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    propensity = expit(0.7 * z[:, 0] - 0.5 * z[:, 1] + 0.3 * z[:, 2])
    x = rng.binomial(1, propensity) if intervention is None else np.full(n, intervention)
    eps = rng.normal(size=(n, 2))
    y1 = 1.5 * x + 0.7 * z[:, 0] + 0.55 * (z[:, 1] ** 2 - 1) + 0.6 * eps[:, 0]
    y2 = -0.75 * x + 0.4 * z[:, 0] + 0.6 * np.sin(z[:, 2]) + 0.35 * eps[:, 0] + 0.5 * eps[:, 1]
    y = np.stack((y1, y2), axis=1)[:, :dim]
    return x.reshape(-1, 1).astype(np.float32), z.astype(np.float32), y.astype(np.float32)


def score_scalar(draw0, draw1, truth0, truth1):
    """Shared metrics on intervention samples; lower is better."""
    draw0, draw1, truth0, truth1 = [np.asarray(v).reshape(-1) for v in (draw0, draw1, truth0, truth1)]
    q = [0.1, 0.5, 0.9]
    true_ate = float(truth1.mean() - truth0.mean())
    est_ate = float(draw1.mean() - draw0.mean())
    return {
        "ate_error": abs(est_ate - true_ate),
        "wasserstein_mean": 0.5 * (wasserstein_distance(draw0, truth0) + wasserstein_distance(draw1, truth1)),
        "energy_mean": 0.5 * (energy_distance(draw0, truth0) + energy_distance(draw1, truth1)),
        "quantile_mae": 0.5 * sum(
            np.abs(np.quantile(d, q) - np.quantile(t, q)).mean()
            for d, t in ((draw0, truth0), (draw1, truth1))
        ),
    }


def multivariate_energy(draw, truth):
    """Empirical energy distance for the joint two-outcome distribution."""
    draw, truth = np.asarray(draw), np.asarray(truth)
    if draw.ndim != 2 or truth.ndim != 2 or draw.shape[1] != truth.shape[1]:
        raise ValueError("The joint outcome samples must be matrices of the same width")
    if min(len(draw), len(truth)) < 2:
        raise ValueError("At least two samples per distribution are required")
    return float(2 * cdist(draw, truth).mean()
                 - 2 * pdist(draw).sum() / (len(draw) ** 2)
                 - 2 * pdist(truth).sum() / (len(truth) ** 2))


def fit_frengression(x, z, y, seed: int, iterations: int, mc: int):
    import torch
    from frengression import Frengression

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    xt, zt, yt = (torch.as_tensor(v, dtype=torch.float32) for v in (x, z, y))
    model = Frengression(x_dim=1, y_dim=y.shape[1], z_dim=z.shape[1],
                         num_layer=2, hidden_dim=64, noise_dim=8,
                         x_binary=True, device=torch.device("cpu"))
    start = time.perf_counter()
    # Identical training sample for the two methods; only the causal-margin fit
    # is timed/scored. Training g for a joint-data benchmark is a separate task.
    model.train_y(xt, zt, yt, num_iters=iterations, lr=1e-4,
                  print_every_iter=iterations, tol=-1.0)
    elapsed = time.perf_counter() - start
    draws = []
    for arm in (0, 1):
        fixed_x = torch.tensor([[float(arm)]], dtype=torch.float32)
        sample = model.sample_causal_margin(fixed_x, sample_size=mc)
        draws.append(np.asarray(sample.detach().cpu()).reshape(mc, y.shape[1]))
    return draws, elapsed


def fit_frugal_flows(x, z, y, seed: int, epochs: int, marginal_epochs: int, mc: int):
    import jax
    import jax.numpy as jnp
    from frugal_flows.benchmarking import FrugalFlowModel

    jax.config.update("jax_enable_x64", True)
    model = FrugalFlowModel(Y=jnp.asarray(y, dtype=jnp.float64),
                            X=jnp.asarray(x, dtype=jnp.float64),
                            Z_cont=jnp.asarray(z, dtype=jnp.float64))
    flow_kwargs = dict(RQS_knots=8, flow_layers=3, nn_width=64, nn_depth=2,
                       learning_rate=5e-4, max_epochs=epochs,
                       max_patience=min(25, epochs), batch_size=min(256, len(x)),
                       show_progress=False)
    causal_kwargs = dict(RQS_knots=8, flow_layers=3, nn_width=64, nn_depth=2)
    start = time.perf_counter()
    model.train_marginal_cdfs(jax.random.key(seed),
                              dict(max_epochs=marginal_epochs,
                                   max_patience=min(25, marginal_epochs)))
    model.train_frugal_flow(jax.random.key(seed + 10000), flow_kwargs,
                            "flexible_continuous", causal_kwargs)
    # Force synchronization before timing a JAX fit.
    _ = float(np.asarray(model.min_val_loss))
    elapsed = time.perf_counter() - start
    draws = [np.asarray(model.sample_do(jax.random.key(seed + 20000 + arm), arm, mc)).reshape(mc, 1)
             for arm in (0, 1)]
    return draws, elapsed


def run(args):
    rows = []
    for rep in range(args.repeats):
        seed = args.seed + rep
        x, z, y = simulate(args.n, seed, dim=1)
        truth = [simulate(args.truth_mc, seed + 100000 + arm,
                          intervention=arm, dim=1)[2][:, 0] for arm in (0, 1)]
        for method in args.methods:
            if method == "frengression":
                draws, seconds = fit_frengression(x, z, y, seed, args.fr_iters, args.mc)
            else:
                draws, seconds = fit_frugal_flows(x, z, y, seed, args.flow_epochs,
                                                  args.marginal_epochs, args.mc)
            metrics = score_scalar(draws[0], draws[1], *truth)
            rows.append(dict(seed=seed, setting="binary_scalar", method=method,
                             seconds=seconds, **metrics))
            print(rows[-1], flush=True)

        if args.multivariate:
            x2, z2, y2 = simulate(args.n, seed, dim=2)
            true2 = [simulate(args.mc, seed + 300000 + arm,
                              intervention=arm, dim=2)[2] for arm in (0, 1)]
            draws2, seconds = fit_frengression(x2, z2, y2, seed + 500000,
                                                args.fr_iters, args.mc)
            # A product-of-margins reference is trained on the same data. It can
            # fit each outcome margin, but sampling independently loses their
            # interventional dependence; this tests the value of joint fitting.
            separate = []
            separate_seconds = 0.0
            for coordinate in range(2):
                arm_draws, fit_seconds = fit_frengression(
                    x2, z2, y2[:, coordinate:coordinate + 1],
                    seed + 600000 + coordinate, args.fr_iters, args.mc)
                separate.append(arm_draws)
                separate_seconds += fit_seconds
            independent_draws = [np.concatenate((separate[0][arm], separate[1][arm]), axis=1)
                                 for arm in (0, 1)]
            for method, method_draws, fit_seconds in (
                ("frengression_joint", draws2, seconds),
                ("frengression_independent_margins", independent_draws, separate_seconds),
            ):
                rows.append(dict(seed=seed, setting="binary_two_outcomes", method=method,
                                 seconds=fit_seconds,
                                 joint_energy_mean=np.mean([multivariate_energy(d, t)
                                                            for d, t in zip(method_draws, true2)]),
                                 mean_error=np.mean([np.abs(d.mean(0) - t.mean(0)).mean()
                                                     for d, t in zip(method_draws, true2)]),
                                 covariance_error=np.mean([np.linalg.norm(np.cov(d, rowvar=False)
                                                                         - np.cov(t, rowvar=False), ord="fro")
                                                           for d, t in zip(method_draws, true2)])))
                print(rows[-1], flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} fitted-model rows to {out}")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("frengression", "frugal_flows"),
                        default=["frengression", "frugal_flows"])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--truth-mc", type=int, default=50000)
    parser.add_argument("--fr-iters", type=int, default=2000)
    parser.add_argument("--flow-epochs", type=int, default=2000)
    parser.add_argument("--marginal-epochs", type=int, default=400)
    parser.add_argument("--multivariate", action="store_true")
    parser.add_argument("--output", default="comparison_results.csv")
    args = parser.parse_args()
    if min(args.n, args.mc, args.truth_mc, args.repeats,
           args.fr_iters, args.flow_epochs, args.marginal_epochs) < 2:
        parser.error("Sample sizes, repeats, and training iterations must all be at least 2")
    run(args)


if __name__ == "__main__":
    cli()
