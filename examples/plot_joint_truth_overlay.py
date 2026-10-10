"""Compare fitted probability regions directly with truth using saved draws.

The selected seed changes only this illustration. The aggregate evaluation
retains every completed replication. The seed-selection rule must be reported.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_joint_outcome_density import METHODS, LABELS, _load_draws, _density_fields


def _region_threshold(density, area, probability):
    ordered = np.sort(density.ravel())[::-1]
    mass = np.cumsum(ordered) * area
    if mass[-1] < probability:
        raise ValueError('The density grid omits too much probability mass')
    return float(ordered[np.searchsorted(mass, probability)])


def redraw_truth_overlay(result_dir, seed=2032, output_dir=None):
    result_dir = Path(result_dir)
    output_dir = result_dir if output_dir is None else Path(output_dir)
    metadata = json.loads((result_dir / 'metadata.json').read_text())
    means = {int(arm): np.array(mean, dtype=float)
             for arm, mean in metadata['true_mean'].items()}
    covariance = np.array(metadata['true_covariance'], dtype=float)
    sd = np.sqrt(np.diag(covariance))
    draws = _load_draws(result_dir, metadata, seed)
    edges, _, _, densities = _density_fields(draws, means, covariance)
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    area = (edges[0][1] - edges[0][0]) * (edges[1][1] - edges[1][0])
    # The truth is convolved with the same kernel used to display fitted draws.
    smooth_covariance = covariance + np.diag((0.18 * sd) ** 2)
    vals, vecs = np.linalg.eigh(smooth_covariance)
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.stack((np.cos(theta), np.sin(theta)))
    unit_ellipse = vecs @ np.diag(np.sqrt(vals)) @ circle
    colours = ['#6975B9', '#25B7B2', '#F2D45B']
    thresholds = {}
    cov_errors = {}
    with plt.rc_context({'font.size': 10.5, 'axes.titlesize': 11.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelsize': 12, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 4, figsize=(13.8, 8.0))
        for arm in (0, 1):
            mean = means[arm]
            for method_index, method in enumerate(METHODS):
                density = densities[method, arm]
                levels = [_region_threshold(density, area, mass)
                          for mass in (0.90, 0.50, 0.20)]
                thresholds[f'{method}_arm{arm}'] = levels
                levels.append(float(density.max()) * (1 + 1e-8))
                if np.any(np.diff(levels) <= 0):
                    raise ValueError('Density levels must be strictly ordered')
                cov_errors[f'{method}_arm{arm}'] = float(np.linalg.norm(
                    np.cov(draws[method][arm], rowvar=False) - covariance, ord='fro'))
                for zoom in (False, True):
                    column = method_index + (2 if zoom else 0)
                    ax = axes[arm, column]
                    ax.contourf(centers[0], centers[1], density.T,
                                levels=levels, colors=colours, antialiased=True)
                    for mass in ((0.20, 0.50) if zoom else (0.50, 0.90)):
                        reference = (unit_ellipse * np.sqrt(-2 * np.log(1 - mass))
                                     + mean[:, None])
                        ax.plot(*reference, color='#20252C', linewidth=1.7,
                                linestyle=(0, (4, 2.6)), zorder=4)
                    radius = (1.5 if zoom else 2.9) * sd
                    ax.set(xlim=(mean[0] - radius[0], mean[0] + radius[0]),
                           ylim=(mean[1] - radius[1], mean[1] + radius[1]),
                           aspect='equal')
                    ax.set_xticks(mean[0] + (np.array([-1, 0, 1]) if zoom
                                             else np.array([-2, 0, 2])))
                    ax.set_yticks(mean[1] + (np.array([-1, 0, 1]) if zoom
                                             else np.array([-2, 0, 2])))
                    if arm == 0:
                        ax.set_title(LABELS[method], pad=10)
                    if column in (0, 2):
                        ax.set_ylabel(f'do(X={arm})\n$Y_2$')
                    if arm == 1:
                        ax.set_xlabel('$Y_1$')
                    if not zoom:
                        error = cov_errors[f'{method}_arm{arm}']
                        ax.text(0.04, 0.035, f'Covariance error = {error:.3f}',
                                transform=ax.transAxes, fontsize=9.5,
                                color='#333B44', va='bottom')
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.115,
                            top=0.85, wspace=0.24, hspace=0.19)
        fig.text(0.275, 0.965, 'Full distribution', ha='center',
                 fontsize=14, weight='semibold', color='#333B44')
        fig.text(0.752, 0.965, 'Central region enlarged', ha='center',
                 fontsize=14, weight='semibold', color='#333B44')
        fig.text(0.275, 0.925, 'True outlines: 50% and 90%', ha='center',
                 fontsize=10.5, color='#555E66')
        fig.text(0.752, 0.925, 'True outlines: 20% and 50%', ha='center',
                 fontsize=10.5, color='#555E66')
        handles = [Patch(facecolor=colour, edgecolor='none', label=f'{mass}% region')
                   for colour, mass in zip(colours, (90, 50, 20))]
        handles.append(Line2D([], [], color='#20252C', linewidth=1.7,
                             linestyle=(0, (4, 2.6)), label='True regions'))
        fig.legend(handles=handles, loc='lower center', frameon=False, ncol=4,
                   bbox_to_anchor=(0.52, 0.012), fontsize=11,
                   handlelength=2.2, columnspacing=2.2)
        stem = output_dir / f'joint_truth_overlay_seed{seed}'
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
        plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps({
        'seed': seed, 'draws_per_arm': metadata['mc_draws'],
        'selection': ('seed 2032 is a favourable illustration selected by the largest arm-averaged covariance error advantage in the 30 completed replications'
                      if seed == 2032 else 'explicitly requested completed seed'),
        'display': 'nested fitted high-density regions, with analytic true regions overlaid',
        'region_probabilities': [0.9, 0.5, 0.2],
        'full_view_true_outlines': [0.9, 0.5],
        'central_view_true_outlines': [0.5, 0.2],
        'grid_bins': 140, 'bandwidth_in_true_marginal_sd': 0.18,
        'truth': 'analytic Gaussian convolution with the same smoothing kernel',
        'thresholds': thresholds,
        'covariance_errors_from_original_draws': cov_errors,
        'aggregate_evaluation': 'unchanged; uses all completed replications'
    }, indent=2))
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
        default=Path.home() / 'frengression_results' / 'joint_outcomes_causl'
        / 'repeats_v2_n5000_s1_fr1000_gcomp1000_mc5000_k30')
    parser.add_argument('--illustration-seed', type=int, default=2032)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(redraw_truth_overlay(args.results_dir, args.illustration_seed, args.output_dir))


if __name__ == '__main__':
    main()
