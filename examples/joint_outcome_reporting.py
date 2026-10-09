"""Checkpointing and figures for the joint-outcome notebook; no fitted results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def atomic_csv(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _draws(values, expected=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 2
            or not np.isfinite(values).all()):
        raise ValueError('Expected finite joint draws with shape [draws, 2]')
    if expected is not None and len(values) != expected:
        raise ValueError('Checkpoint has an unexpected number of joint draws')
    return values


def save_draw_checkpoint(path, protocol, draws, seconds, extras=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = protocol['mc_draws']
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid fit time')
    payload = dict(arm0=_draws(draws[0], expected), arm1=_draws(draws[1], expected),
                   seconds=np.array(seconds), protocol=np.array(json.dumps(protocol, sort_keys=True)))
    for name, values in (extras or {}).items():
        if name in payload:
            raise ValueError('Duplicate checkpoint field')
        payload[name] = _draws(values)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def load_draw_checkpoint(path, protocol):
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        stored = json.loads(str(payload['protocol'].item()))
        if stored != protocol:
            raise ValueError(f'Checkpoint settings do not match: {path}')
        expected = protocol['mc_draws']
        draws = {arm: _draws(payload[f'arm{arm}'], expected) for arm in (0, 1)}
        seconds = float(payload['seconds'])
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('Checkpoint has an invalid fit time')
        extras = {name: _draws(payload[name]) for name in payload.files
                  if name not in {'arm0', 'arm1', 'seconds', 'protocol'}}
    return draws, seconds, extras


def _save_figure(fig, stem):
    if stem is None:
        return
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + suffix), dpi=240, bbox_inches='tight')


def plot_joint_distributions(methods, means, covariance, stem, title=None):
    """Shared axes/density scale; exact contours and unsmoothed model histograms."""
    labels = {
        'frengression_joint': 'Frengression',
        'conditional_engression_gcomp': 'Conditional engression\n+ g-computation',
        'shuffled_diagnostic': 'Shuffled margins',
    }
    order = [name for name in labels if name in methods]
    if not order:
        raise ValueError('No fitted draws to plot')
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2) or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError('Expected positive definite bivariate truth covariance')
    sd = np.sqrt(np.diag(covariance))
    low = np.min([means[a] - 3.6 * sd for a in (0, 1)], axis=0)
    high = np.max([means[a] + 3.6 * sd for a in (0, 1)], axis=0)
    edges = [np.linspace(low[j], high[j], 49) for j in range(2)]
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    xx, yy = np.meshgrid(*centers, indexing='ij')
    inverse = np.linalg.inv(covariance)
    scale = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))
    densities, samples = {}, {}
    for arm in (0, 1):
        delta = np.stack((xx - means[arm][0], yy - means[arm][1]), axis=-1)
        densities[arm, 'truth'] = scale * np.exp(-0.5 * np.einsum('...i,ij,...j->...', delta, inverse, delta))
        for method in order:
            values = _draws(methods[method][arm])
            samples[arm, method] = values
            counts, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=edges)
            # Divide by the full draw count, without renormalising omitted tails.
            densities[arm, method] = counts / (len(values) * np.diff(edges[0])[:, None]
                                                     * np.diff(edges[1])[None, :])
    norm = Normalize(0, max(float(d.max()) for d in densities.values()))
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, len(order) + 1, figsize=(3.4 * (len(order) + 1), 6.7),
                                 sharex=True, sharey=True, constrained_layout=True)
        theta = np.linspace(0, 2 * np.pi, 300)
        vals, vecs = np.linalg.eigh(covariance)
        unit = np.stack((np.cos(theta), np.sin(theta)))
        columns = ['truth', *order]
        rho = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        median_truth = 0.25 + np.arcsin(rho) / (2 * np.pi)
        for arm in (0, 1):
            for column, method in enumerate(columns):
                ax = axes[arm, column]
                mesh = ax.pcolormesh(edges[0], edges[1], densities[arm, method].T,
                                     cmap='viridis', norm=norm, shading='flat', rasterized=True)
                for mass, style in ((0.5, '--'), (0.95, '-')):
                    ellipse = (vecs @ np.diag(np.sqrt(vals)) @ unit
                               * np.sqrt(-2 * np.log(1 - mass)) + means[arm][:, None])
                    ax.plot(*ellipse, color='white', linewidth=1.2, linestyle=style)
                if method == 'truth':
                    cov12, probability = covariance[0, 1], median_truth
                else:
                    values = samples[arm, method]
                    cov12 = np.cov(values, rowvar=False)[0, 1]
                    probability = np.mean(np.all(values > means[arm], axis=1))
                ax.text(0.04, 0.04, f'Cov = {cov12:.3f}\nP(both above medians) = {probability:.3f}',
                        transform=ax.transAxes, fontsize=8.5, va='bottom',
                        bbox={'facecolor': 'white', 'alpha': 0.92, 'edgecolor': 'none', 'pad': 4})
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
                if arm == 0:
                    ax.set_title('Exact distribution' if method == 'truth' else labels[method], pad=9)
                if column == 0:
                    ax.set_ylabel(f'do(X={arm})\n$Y_2$')
                if arm == 1:
                    ax.set_xlabel('$Y_1$')
        fig.colorbar(mesh, ax=axes, shrink=0.82, label='Joint density', pad=0.015)
        fig.suptitle(title or 'Joint outcome distributions under intervention', fontsize=14)
        fig.supxlabel('White contours: exact 50% (dashed) and 95% (solid) Gaussian regions', fontsize=9)
        _save_figure(fig, stem)
    return fig


def plot_paired_errors(results, stem, title=None):
    """One point per paired dataset; all successful pairs are retained."""
    metrics = [('covariance_fro_error', 'Covariance error'),
               ('joint_cdf_grid_mae', 'Joint CDF grid error')]
    subset = results.loc[results.method.isin(['frengression_joint', 'conditional_engression_gcomp'])]
    if subset.duplicated(['seed', 'arm', 'method']).any():
        raise ValueError('Duplicate paired results')
    wide = subset.pivot(index=['seed', 'arm'], columns='method', values=[m for m, _ in metrics])
    if not {'frengression_joint', 'conditional_engression_gcomp'}.issubset(wide.columns.get_level_values(1)):
        return None
    wide = wide.dropna()
    if wide.empty:
        return None
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 4.4), constrained_layout=True)
        for ax, (metric, label) in zip(axes, metrics):
            x = wide[metric, 'frengression_joint']
            y = wide[metric, 'conditional_engression_gcomp']
            limit = max(float(x.max()), float(y.max()), 1e-8) * 1.08
            ax.plot([0, limit], [0, limit], '--', color='0.45', linewidth=1)
            for arm, color, marker in ((0, '#0072B2', 'o'), (1, '#D55E00', '^')):
                mask = wide.index.get_level_values('arm') == arm
                ax.scatter(x[mask], y[mask], s=34, alpha=0.8, color=color, marker=marker,
                           edgecolor='white', linewidth=0.35, label=f'do(X={arm})')
            ax.set(xlim=(0, limit), ylim=(0, limit), title=label,
                   xlabel='Frengression error', ylabel='Conditional engression error')
            ax.set_aspect('equal', adjustable='box')
        axes[0].legend(frameon=False)
        fig.suptitle(title or 'Paired comparison across independently generated datasets', fontsize=12)
        fig.supxlabel('Above the diagonal: smaller Frengression error. Each point is one arm of one dataset.', fontsize=9)
        _save_figure(fig, stem)
    return fig
