"""Draw joint density blocks and density residuals from completed checkpoints.

Only the illustration changes. Training, saved outcome pairs and the
30-replication comparison are unchanged. An explicitly chosen favourable seed
must be identified as such when its figure is used in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from scipy.ndimage import gaussian_filter

from joint_outcome_reporting import load_draw_checkpoint


METHODS = ('frengression_joint', 'conditional_engression_gcomp')
LABELS = {'truth': 'True joint distribution',
          'frengression_joint': 'Frengression',
          'conditional_engression_gcomp': 'Conditional engression\n+ g-computation',
          'shuffled_diagnostic': 'Shuffled pairs'}


def _load_draws(result_dir, metadata, seed):
    if seed not in metadata['seeds']:
        raise ValueError('Illustration seed is not part of the recorded experiment')
    draws = {}
    for method in METHODS:
        protocol = dict(format=2, n=metadata['n'], seed=seed,
            strength=metadata['instrument_strength'], method=method,
            fr_iters=metadata['frengression_updates'], gcomp_iters=metadata['gcomp_updates'],
            mc_draws=metadata['mc_draws'], layers=metadata['layers'],
            width=metadata['width'], noise_dim=metadata['noise_dim'],
            learning_rate=metadata['learning_rate'],
            model_source_sha256=metadata['model_source_sha256'])
        cached = load_draw_checkpoint(result_dir / f'seed{seed}_{method}.npz', protocol)
        if cached is None:
            raise ValueError(f'No completed checkpoint for seed {seed}, {method}')
        draws[method] = cached[0]
    rng = np.random.default_rng(seed + 99)
    draws['shuffled_diagnostic'] = {
        arm: np.column_stack((draws['frengression_joint'][arm][:, 0],
                              rng.permutation(draws['frengression_joint'][arm][:, 1])))
        for arm in (0, 1)}
    return draws


def _density_fields(draws, means, covariance):
    """Apply one grid and one smoothing kernel to every generated distribution.

    Smoothing the Gaussian truth uses its exact convolution with the same
    Gaussian kernel, so density residuals do not penalise smoothing alone.
    Histograms are divided by all draws, including any omitted tail draws.
    """
    sd = np.sqrt(np.diag(covariance))
    bandwidth = 0.18 * sd
    low = np.min([means[a] - 3.6 * sd for a in (0, 1)], axis=0)
    high = np.max([means[a] + 3.6 * sd for a in (0, 1)], axis=0)
    edges = [np.linspace(low[j] - 4 * bandwidth[j],
                         high[j] + 4 * bandwidth[j], 141) for j in (0, 1)]
    widths = np.array([edge[1] - edge[0] for edge in edges])
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    xy = np.stack(np.meshgrid(*centers, indexing='ij'), axis=-1)
    smooth_covariance = covariance + np.diag(bandwidth ** 2)
    inverse = np.linalg.inv(smooth_covariance)
    constant = 1 / (2 * np.pi * np.sqrt(np.linalg.det(smooth_covariance)))
    fields = {}
    for arm in (0, 1):
        delta = xy - means[arm]
        fields['truth', arm] = constant * np.exp(-0.5 * np.einsum(
            '...i,ij,...j->...', delta, inverse, delta))
        for method, samples in draws.items():
            values = samples[arm]
            counts = np.histogram2d(values[:, 0], values[:, 1], bins=edges)[0]
            counts /= len(values) * np.prod(widths)
            fields[method, arm] = gaussian_filter(counts, bandwidth / widths,
                                                 mode='constant', truncate=4)
    return edges, low, high, fields


def _save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + extension), dpi=220, bbox_inches='tight')


def redraw_joint_density(result_dir, seed=2032, output_dir=None):
    result_dir = Path(result_dir)
    output_dir = result_dir if output_dir is None else Path(output_dir)
    metadata = json.loads((result_dir / 'metadata.json').read_text())
    means = {int(arm): np.array(mean, dtype=float)
             for arm, mean in metadata['true_mean'].items()}
    covariance = np.array(metadata['true_covariance'], dtype=float)
    draws = _load_draws(result_dir, metadata, seed)
    edges, low, high, fields = _density_fields(draws, means, covariance)
    density_norm = Normalize(0, max(float(field.max()) for field in fields.values()))
    theta = np.linspace(0, 2 * np.pi, 360)
    vals, vecs = np.linalg.eigh(covariance)
    ellipse = (vecs @ np.diag(np.sqrt(vals))
               @ np.stack((np.cos(theta), np.sin(theta)))
               * np.sqrt(-2 * np.log(0.05)))
    cov_errors = {f'{method}_arm{arm}': float(np.linalg.norm(
        np.cov(values[arm], rowvar=False) - covariance, ord='fro'))
        for method, values in draws.items() for arm in (0, 1)}
    style = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12,
             'axes.spines.top': False, 'axes.spines.right': False,
             'figure.facecolor': 'white', 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        order = ['truth', *draws]
        fig, axes = plt.subplots(2, len(order), figsize=(14.5, 6.8),
                                 sharex=True, sharey=True)
        for arm in (0, 1):
            for column, method in enumerate(order):
                ax = axes[arm, column]
                ax.pcolormesh(edges[0], edges[1], fields[method, arm].T,
                    cmap='viridis', norm=density_norm, shading='flat', rasterized=True)
                # One reference boundary; no dots or fitted ellipse overlays.
                ax.plot(*(ellipse + means[arm][:, None]), color='white',
                        linewidth=1.1, alpha=0.88)
                if method != 'truth':
                    ax.text(0.035, 0.035,
                        f'Covariance error = {cov_errors[f"{method}_arm{arm}"]:.3f}',
                        transform=ax.transAxes, fontsize=9.7, color='white',
                        va='bottom')
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                       aspect='equal')
                if arm == 0:
                    ax.set_title(LABELS[method], pad=12)
                if column == 0:
                    ax.set_ylabel(f'do(X={arm})\n$Y_2$')
                if arm == 1:
                    ax.set_xlabel('$Y_1$')
        fig.subplots_adjust(left=0.055, right=0.92, top=0.91, bottom=0.12,
                            wspace=0.09, hspace=0.18)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=density_norm, cmap='viridis'),
                         cax=fig.add_axes([0.94, 0.23, 0.011, 0.53]))
        cb.set_label('Smoothed joint density', labelpad=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9, length=3)
        fig.text(0.47, 0.012, 'White outline: true 95% Gaussian region',
                 ha='center', fontsize=10, color='#444444')
        main_stem = output_dir / f'joint_density_blocks_seed{seed}'
        _save(fig, main_stem)
        plt.close(fig)

        differences = {(method, arm): fields[method, arm] - fields['truth', arm]
                       for method in METHODS for arm in (0, 1)}
        max_difference = max(float(np.abs(d).max()) for d in differences.values())
        difference_norm = TwoSlopeNorm(vmin=-max_difference, vcenter=0,
                                       vmax=max_difference)
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.1), sharex=True, sharey=True)
        for arm in (0, 1):
            for column, method in enumerate(METHODS):
                ax = axes[arm, column]
                ax.pcolormesh(edges[0], edges[1], differences[method, arm].T,
                    cmap='RdBu_r', norm=difference_norm, shading='flat', rasterized=True)
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                       aspect='equal')
                if arm == 0:
                    ax.set_title(LABELS[method], pad=12)
                if column == 0:
                    ax.set_ylabel(f'do(X={arm})\n$Y_2$')
                if arm == 1:
                    ax.set_xlabel('$Y_1$')
        fig.subplots_adjust(left=0.085, right=0.86, top=0.91, bottom=0.12,
                            wspace=0.10, hspace=0.18)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=difference_norm, cmap='RdBu_r'),
                         cax=fig.add_axes([0.90, 0.24, 0.022, 0.52]))
        cb.set_label('Generated density $-$ true density', labelpad=10)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9, length=3)
        fig.text(0.48, 0.012,
                 'Blue: less mass     White: close to truth     Red: more mass',
                 ha='center', fontsize=10, color='#444444')
        residual_stem = output_dir / f'joint_density_residuals_seed{seed}'
        _save(fig, residual_stem)
        plt.close(fig)
    info = {'seed': seed,
            'selection': 'explicitly requested completed seed; disclose favourable selection',
            'draws_per_arm': metadata['mc_draws'], 'grid_bins': 140,
            'bandwidth_in_true_marginal_sd': 0.18,
            'truth_smoothing': 'exact Gaussian convolution with the same bandwidth',
            'density_scale': 'shared across all methods and arms',
            'residual_scale': 'shared symmetric scale across fitted methods and arms',
            'aggregate_comparison': 'unchanged; all 30 replications retained',
            'covariance_errors_from_original_draws': cov_errors}
    output_dir.mkdir(parents=True, exist_ok=True)
    main_stem.with_suffix('.json').write_text(json.dumps(info, indent=2))
    return main_stem, residual_stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
        default=Path.home() / 'frengression_results' / 'joint_outcomes_causl'
        / 'repeats_v2_n5000_s1_fr1000_gcomp1000_mc5000_k30')
    parser.add_argument('--illustration-seed', type=int, default=2032)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    for stem in redraw_joint_density(args.results_dir, args.illustration_seed,
                                    args.output_dir):
        print(stem)


if __name__ == '__main__':
    main()
