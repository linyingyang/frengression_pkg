"""Checkpointing and figures for the joint-outcome notebook; no fitted results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
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
        fig.savefig(stem.with_suffix('.' + suffix),
                    dpi=180 if suffix == 'pdf' else 240, bbox_inches='tight')


def _point_density(values, low, high, bandwidth, bins=140):
    """Density colours from a common Gaussian-smoothed histogram grid.

    Normalization uses the total draw count, including points outside the
    displayed limits. The grid and bandwidth are identical across methods.
    """
    padding = 4 * bandwidth
    edges = [np.linspace(low[j] - padding[j], high[j] + padding[j], bins + 1)
             for j in (0, 1)]
    widths = np.array([edge[1] - edge[0] for edge in edges])
    density, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=edges)
    density /= len(values) * np.prod(widths)
    for axis in (0, 1):
        sigma = bandwidth[axis] / widths[axis]
        radius = int(np.ceil(4 * sigma))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        density = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode='same'), axis, density)
    positions = (values - np.array([edge[0] for edge in edges])) / widths - 0.5
    positions = np.clip(positions, 0, bins - 1)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, bins - 1)
    weight = positions - lower
    x0, y0 = lower.T
    x1, y1 = upper.T
    wx, wy = weight.T
    return ((1 - wx) * (1 - wy) * density[x0, y0]
            + wx * (1 - wy) * density[x1, y0]
            + (1 - wx) * wy * density[x0, y1]
            + wx * wy * density[x1, y1])


def plot_joint_distributions(methods, means, covariance, stem, title=None):
    """Joint samples with exact Gaussian regions and sample-moment ellipses.

    All methods use the same axes, point size, opacity and density colour scale.
    Colours use a Gaussian-smoothed histogram with a common bandwidth of
    0.18 times the true marginal standard deviations; fitting is unchanged.
    The solid ellipses describe sample means/covariances, not empirical
    coverage regions for a potentially non-Gaussian fitted distribution.
    """
    labels = {
        'frengression_joint': 'Frengression',
        'conditional_engression_gcomp': 'Conditional engression\n+ g-computation',
        'shuffled_diagnostic': 'Shuffled pairs',
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
    point_colours = {(method, arm): _point_density(
        _draws(methods[method][arm]), low, high, 0.18 * sd)
        for method in order for arm in (0, 1)}
    norm = Normalize(vmin=0, vmax=max(float(values.max())
                                     for values in point_colours.values()))
    theta = np.linspace(0, 2 * np.pi, 300)
    unit = np.stack((np.cos(theta), np.sin(theta)))

    def ellipse(mean, cov, mass):
        vals, vecs = np.linalg.eigh(cov)
        if vals.min() <= 0:
            raise ValueError('Expected nondegenerate joint samples')
        return (vecs @ np.diag(np.sqrt(vals)) @ unit
                * np.sqrt(-2 * np.log(1 - mass)) + mean[:, None])

    truth_colour, fitted_colour = '#252525', '#E76F51'
    with plt.rc_context({'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.facecolor': 'white', 'figure.facecolor': 'white'}):
        fig, axes = plt.subplots(2, len(order), figsize=(3.35 * len(order) + 0.35, 6.5),
                                 sharex=True, sharey=True, squeeze=False)
        for arm in (0, 1):
            for column, method in enumerate(order):
                ax = axes[arm, column]
                values = _draws(methods[method][arm])
                fitted_mean, fitted_cov = values.mean(axis=0), np.cov(values, rowvar=False)
                density = point_colours[method, arm]
                drawing_order = np.argsort(density, kind='stable')
                scatter = ax.scatter(values[drawing_order, 0], values[drawing_order, 1],
                           c=density[drawing_order], cmap='viridis', norm=norm,
                           s=4, alpha=0.65, edgecolors='none', rasterized=True, zorder=1)
                for mass in (0.5, 0.95):
                    ax.plot(*ellipse(fitted_mean, fitted_cov, mass),
                            color=fitted_colour, linewidth=1.9, zorder=3)
                    ax.plot(*ellipse(np.asarray(means[arm]), covariance, mass),
                            color=truth_colour, linewidth=1.5, linestyle='--', zorder=4)
                cov_error = np.linalg.norm(fitted_cov - covariance, ord='fro')
                ax.text(0.04, 0.04, f'Covariance error = {cov_error:.3f}',
                        transform=ax.transAxes, fontsize=9.5, va='bottom',
                        bbox={'facecolor': 'white', 'alpha': 0.92, 'edgecolor': 'none', 'pad': 2})
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
                if arm == 0:
                    ax.set_title(labels[method], pad=10)
                if column == 0:
                    ax.set_ylabel(f'do(X={arm})\n$Y_2$')
                if arm == 1:
                    ax.set_xlabel('$Y_1$')
        handles = [Line2D([], [], color=truth_colour, linestyle='--', linewidth=1.5,
                          label='True Gaussian regions'),
                   Line2D([], [], color=fitted_colour, linewidth=1.8,
                          label='Sample mean/covariance ellipses')]
        fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 0.005))
        if title:
            fig.suptitle(title, fontsize=13)
        fig.subplots_adjust(left=0.075, right=0.915, bottom=0.105,
                            top=0.88 if title else 0.92, wspace=0.12, hspace=0.15)
        colour_axis = fig.add_axes([0.94, 0.19, 0.014, 0.63])
        # An opaque mappable keeps the colour bar faithful to the common scale.
        colour_bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
                                  cax=colour_axis)
        colour_bar.set_label('Joint density (smoothed)', labelpad=8)
        colour_bar.outline.set_visible(False)
        colour_bar.ax.tick_params(labelsize=9, length=3)
        _save_figure(fig, stem)
    return fig


def redraw_saved_joint_report(result_dir, illustration_seed=None):
    """Redraw completed experiments from checkpoints; no R, torch or fitting."""
    result_dir = Path(result_dir)
    metadata = json.loads((result_dir / 'metadata.json').read_text())
    if illustration_seed is not None and illustration_seed not in metadata['seeds']:
        raise ValueError('Illustration seed is not part of the recorded experiment')
    seeds = metadata['seeds'] if illustration_seed is None else [illustration_seed]
    methods = ('frengression_joint', 'conditional_engression_gcomp')
    representative = None
    for seed in seeds:
        if not all((result_dir / f'seed{seed}_{method}.npz').is_file() for method in methods):
            continue
        draws = {}
        for method in methods:
            protocol = dict(format=2, n=metadata['n'], seed=seed,
                strength=metadata['instrument_strength'], method=method,
                fr_iters=metadata['frengression_updates'], gcomp_iters=metadata['gcomp_updates'],
                mc_draws=metadata['mc_draws'], layers=metadata['layers'],
                width=metadata['width'], noise_dim=metadata['noise_dim'],
                learning_rate=metadata['learning_rate'],
                model_source_sha256=metadata['model_source_sha256'])
            cached = load_draw_checkpoint(result_dir / f'seed{seed}_{method}.npz', protocol)
            draws[method] = cached[0]
        rng = np.random.default_rng(seed + 99)
        draws['shuffled_diagnostic'] = {
            arm: np.column_stack((draws['frengression_joint'][arm][:, 0],
                                  rng.permutation(draws['frengression_joint'][arm][:, 1])))
            for arm in (0, 1)}
        representative = draws
        break
    if representative is None:
        raise ValueError('No complete paired checkpoints found in the prespecified seed order')
    means = {int(arm): np.asarray(mean, dtype=float)
             for arm, mean in metadata['true_mean'].items()}
    stem = result_dir / ('joint_distributions' if illustration_seed is None
                         else f'joint_distributions_seed{illustration_seed}')
    figures = [plot_joint_distributions(representative, means,
        np.asarray(metadata['true_covariance'], dtype=float), stem)]
    stem.with_suffix('.json').write_text(json.dumps({
        'seed': seed,
        'selection': ('first complete pair in prespecified seed order'
                      if illustration_seed is None else 'explicitly requested completed seed'),
        'n_draws_per_arm': metadata['mc_draws'],
        'n_planned_replications': len(metadata['seeds']),
        'aggregate_scores': 'scores.csv',
    }, indent=2))
    scores_path = result_dir / 'scores.csv'
    if scores_path.is_file():
        paired = plot_paired_errors(pd.read_csv(scores_path), result_dir / 'paired_error_comparison')
        if paired is not None:
            figures.append(paired)
    return figures


def plot_paired_errors(results, stem, title=None):
    """Paired error differences, averaging the two arms within each dataset.

    Every complete paired dataset is shown in seed order. Positive values
    indicate lower Frengression error. Differences use the original scores.
    """
    metrics = [('covariance_fro_error', 'Covariance error'),
               ('joint_cdf_grid_mae', 'Joint CDF grid error'),
               ('mean_mae', 'Marginal mean error'),
               ('joint_event_mae', 'Joint event probability error')]
    subset = results.loc[results.method.isin(['frengression_joint', 'conditional_engression_gcomp'])]
    if subset.duplicated(['seed', 'arm', 'method']).any():
        raise ValueError('Duplicate paired results')
    wide = subset.pivot(index=['seed', 'arm'], columns='method', values=[m for m, _ in metrics])
    if not {'frengression_joint', 'conditional_engression_gcomp'}.issubset(wide.columns.get_level_values(1)):
        return None
    wide = wide.dropna()
    if wide.empty:
        return None
    complete = [seed for seed, frame in wide.groupby(level='seed')
                if set(frame.index.get_level_values('arm')) == {0, 1}]
    if not complete:
        return None
    averaged = wide.loc[wide.index.get_level_values('seed').isin(complete)]
    averaged = averaged.groupby(level='seed').mean().sort_index()
    replication = np.arange(1, len(averaged) + 1)
    positive, negative, ink = '#168B86', '#E76F51', '#263445'
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#718096', 'axes.linewidth': 0.7,
                         'axes.facecolor': 'white', 'figure.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8), sharex=True)
        for ax, (metric, label) in zip(axes.flat, metrics):
            fr = averaged[metric, 'frengression_joint']
            comparator = averaged[metric, 'conditional_engression_gcomp']
            difference = (comparator - fr).to_numpy()
            limit = max(float(np.abs(difference).max()), 1e-8) * 1.45
            colours = np.where(difference >= 0, positive, negative)
            ax.axhspan(0, limit, color=positive, alpha=0.045, zorder=0)
            ax.axhspan(-limit, 0, color=negative, alpha=0.045, zorder=0)
            ax.axhline(0, color=ink, linewidth=1.1, zorder=1)
            ax.axhline(difference.mean(), color=ink, linestyle=':', linewidth=1.4,
                       zorder=2)
            ax.vlines(replication, 0, difference, colors=colours, linewidth=1.2,
                      alpha=0.65, zorder=2)
            ax.scatter(replication, difference, c=colours, s=31, edgecolors='white',
                       linewidths=0.5, zorder=3)
            ax.text(0.0, 1.015,
                    f'{int((difference > 0).sum())}/{len(difference)} lower Frengression error'
                    f'   |   mean Δ = {difference.mean():+.4f}',
                    transform=ax.transAxes, va='bottom', fontsize=8.5)
            ticks = sorted({1, len(replication), *range(10, len(replication), 10)})
            ax.set(xlim=(0, len(replication) + 1), ylim=(-limit, limit), xticks=ticks)
            ax.set_title(label, pad=28)
            ax.grid(axis='y', color='#CBD5E0', linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel('Replication')
        fig.supylabel('Error difference: conditional engression − Frengression',
                      x=0.015, fontsize=10.5)
        handles = [Line2D([], [], marker='o', linestyle='none', color=positive,
                          label='Lower Frengression error'),
                   Line2D([], [], marker='o', linestyle='none', color=negative,
                          label='Lower conditional engression error'),
                   Line2D([], [], color=ink, linestyle=':', linewidth=1.4,
                          label='Mean paired difference')]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 0.005), columnspacing=1.2)
        if title:
            fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11,
                            top=0.85 if title else 0.91, hspace=0.45, wspace=0.25)
        _save_figure(fig, stem)
    return fig
