"""Derive the plotted values from the archived CPU results."""
import argparse
import csv
from pathlib import Path

from loopback_2pc import WIDTHS

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parent / "figures"


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

    with (ROOT / "results/loopback_summary.csv").open(newline="") as source:
        timing = {(row["template"], int(row["width"]), row["arm"]): row
                  for row in csv.DictReader(source)}
    ablation_rows = ["width full islands rewrite joint algebra"]
    for width in WIDTHS:
        medians = [timing["public", width, arm]["online_median_us"]
                   for arm in ("full", "islands", "rewrite", "joint", "algebra")]
        ablation_rows.append(" ".join((str(width), *medians)))

    matched_rows = []
    for name, arm in (("All squares in 2PC", "full"), ("Islands only", "islands"),
                      ("Rewrite only", "rewrite"), ("Joint", "joint"),
                      ("Hand-written algebra", "algebra")):
        trial = timing["public", 64, arm]
        values = [trial[key] for key in ("secure_products", "peer_bytes", "client_bytes",
                                        "offline_bytes", "online_median_us", "cold_median_us")]
        matched_rows.append(" & ".join((name, *values)) + r" \\")
    return {
        "causal-sweep.dat": "\n".join(causal_rows) + "\n",
        "ablation-sweep.dat": "\n".join(ablation_rows) + "\n",
        "matched-rows.tex": "\n".join(matched_rows + [r"\bottomrule"]) + "\n",
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
