"""Redraw the joint-outcome figures from completed checkpoints."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joint_outcome_reporting import redraw_saved_joint_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
        default=Path.home() / 'frengression_results' / 'joint_outcomes_causl'
        / 'repeats_v2_n5000_s1_fr1000_gcomp1000_mc5000_k30')
    parser.add_argument('--illustration-seed', type=int,
        help='Display this completed seed; aggregate comparisons retain all replications.')
    args = parser.parse_args()
    for figure in redraw_saved_joint_report(args.results_dir, args.illustration_seed):
        plt.close(figure)
    print('Updated PNG/PDF/SVG figures:', args.results_dir)


if __name__ == '__main__':
    main()
