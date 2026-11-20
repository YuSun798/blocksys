"""Derive the plotted values from the archived CPU results."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parent / "figures"
WIDTHS = (1, 4, 16, 64)


def plot_tables():
    causal = {}
    with (ROOT / "results/aggregate.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            x = round(100 * float(row["private_fraction"]))
            causal.setdefault(x, {})[row["pattern"]] = (
                100 * float(row["mean_safe_public_fraction"])
            )
    causal_rows = ["private suffix interleaved prefix expected"]
    for x, row in sorted(causal.items()):
        k = x * 16 // 100
        expected = 100 * (16 - k) / (16 * (k + 1))
        causal_rows.append(
            f"{x} {row['suffix']:.6f} {row['interleaved']:.6f} "
            f"{row['prefix']:.6f} {expected:.6f}"
        )

    trials = json.loads((ROOT / "results/rewrite_island.json").read_text())["results"]
    ablation_rows = ["width full rewrite combined"]
    for width in WIDTHS:
        metrics = [trials[f"width={width}/rewrite={r}/islands={i}"]
                   for r, i in ((0, 0), (0, 1), (1, 0), (1, 1))]
        full, islands, rewrite, combined = [
            row["multiplications"] // row["cases"] for row in metrics
        ]
        if full != islands:
            raise ValueError(f"Full MPC and islands-only diverge at width {width}")
        ablation_rows.append(f"{width} {full} {rewrite} {combined}")

    return {
        "causal-sweep.dat": "\n".join(causal_rows) + "\n",
        "ablation-sweep.dat": "\n".join(ablation_rows) + "\n",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in plot_tables().items():
        path = FIGURES / name
        if args.check:
            if path.read_text() != content:
                raise SystemExit(f"Stale plot data: {path}")
        else:
            path.write_text(content)
