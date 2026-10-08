"""Summarize already saved per-run ADRF predictions without refitting CausalEGM.

Input CSV columns: method, run, x, mean_estimate (one point per run/dose).
The output bands are across *replicate estimated means* for every method.
They describe between-run variation, not confidence intervals or coverage.
An optional true_mean column is carried through and checked for consistency.

Example:
  python adrf_replicate_bands.py predictions.csv --output adrf_bands.csv \
      --figure adrf_bands.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(records):
    groups = defaultdict(list)
    truth = {}
    seen = set()
    for row in records:
        method = str(row["method"])
        run = str(row["run"])
        x = float(row["x"])
        value = float(row["mean_estimate"])
        if not np.isfinite(value):
            raise ValueError("Non-finite mean_estimate")
        key = (method, x)
        if (key, run) in seen:
            raise ValueError(f"Duplicate run {run} for {key}")
        seen.add((key, run))
        groups[key].append(value)
        if row.get("true_mean") not in (None, ""):
            true_value = float(row["true_mean"])
            if x in truth and not np.isclose(truth[x], true_value):
                raise ValueError(f"Inconsistent truth for dose {x}")
            truth[x] = true_value
    result = []
    for (method, x), values in sorted(groups.items()):
        if len(values) < 2:
            raise ValueError(f"At least two replicate estimates are needed for {(method, x)}")
        result.append(dict(method=method, x=x, replicates=len(values),
                           mean_estimate=float(np.mean(values)),
                           lower=float(np.quantile(values, 0.025)),
                           upper=float(np.quantile(values, 0.975)),
                           true_mean=truth.get(x, "")))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with method,run,x,mean_estimate")
    parser.add_argument("--output", default="adrf_bands.csv")
    parser.add_argument("--figure", help="Optional PNG plot")
    args = parser.parse_args()
    with open(args.input, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"method", "run", "x", "mean_estimate"}
        if not required.issubset(reader.fieldnames or []):
            parser.error(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = summarize(reader)
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.figure:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for method in sorted({row["method"] for row in rows}):
            block = sorted((row for row in rows if row["method"] == method), key=lambda row: row["x"])
            x = np.array([row["x"] for row in block])
            ax.plot(x, [row["mean_estimate"] for row in block], label=method)
            ax.fill_between(x, [row["lower"] for row in block],
                            [row["upper"] for row in block], alpha=0.17)
        truth_rows = sorted({row["x"]: row["true_mean"] for row in rows
                             if row["true_mean"] != ""}.items())
        if truth_rows:
            ax.plot([r[0] for r in truth_rows], [r[1] for r in truth_rows],
                    color="black", linestyle="--", label="Truth")
        ax.set(xlabel="Treatment level", ylabel="Mean outcome")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.figure, dpi=200)
        plt.close(fig)
    print(f"Wrote {len(rows)} method/dose rows to {Path(args.output)}")


if __name__ == "__main__":
    main()
