"""Derive the plotted values from the archived CPU results."""
import argparse
import csv
import json
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
    head = json.loads((ROOT / "results/pythia_head_summary.json").read_text())["summary"]
    head_rows = []
    for title, arm in (("Full output", "full"), ("Prefix reuse", "reuse"),
                       ("Rows only", "rows"), ("Joint", "joint"),
                       ("Joint, private pool", "private_pool")):
        row = head[arm]
        values = (row["projection_rows"], row["upload_bytes"] + row["download_bytes"],
                  row["median_projection_us"], row["median_prefill_pair_ms"])
        head_rows.append(" & ".join((title, *(str(value) for value in values))) + r" \\")
    scores = json.loads((ROOT / "results/pythia_qk_summary.json").read_text())["summary"]
    qk_rows = []
    for placement in ("early", "late"):
        for title, arm in (("All shared", "all_shared"), ("Selective", "selective")):
            row = scores[placement][arm]
            values = [f"{row['median_' + key]:.0f}" for key in
                      ("beaver_multiplications", "peer_payload_bytes", "dealer_bytes")]
            values.append(f"{row['median_online_us']:.1f}")
            qk_rows.append(" & ".join((placement.title(), title, *values)) + r" \\")
    pipeline = json.loads((ROOT / "results/pythia_composed_summary.json").read_text())["summary"]
    composed_rows = []
    for label, arm in (("All-shared QK / full head", "full"), ("Selective QK only", "qk_only"),
                       ("Head rows only", "rows_only"), ("Joint", "joint"),
                       ("Joint, private pool", "private_pool")):
        row = pipeline[arm]
        values = [f"{row['median_' + key]:.0f}" for key in
                  ("qk_peer_bytes", "qk_dealer_bytes", "head_bytes")]
        values.append(f"{row['median_elapsed_ms']:.2f}")
        composed_rows.append(" & ".join((label, *values)) + r" \\")
    compiled = json.loads((ROOT / "results/cross_layer_rewrite_summary.json").read_text())["summary"]
    compiled_rows = []
    for label, arm in (("Original, all shared", "full"), ("Original, selective", "operator_only"),
                       ("Rewrite, all shared", "rewrite_only"), ("Automatic + selective", "joint"),
                       ("Manual + selective", "manual")):
        row = compiled[arm]
        total = sum(row["mean_" + key] for key in
                    ("qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes"))
        values = (f"{row['mean_selected_prefix_tokens']:.2f}",
                  f"{row['mean_qk_peer_bytes'] / 1000:,.1f}",
                  f"{total / 1000:,.1f}", f"{row['median_elapsed_ms']:.2f}",
                  f"{row['protected_correct']}/{row['prompts']}")
        compiled_rows.append(" & ".join((label, *values)) + r" \\")
    scale_rows = []
    for checkpoint, suffix in (("70M", ""), ("160M", "_160m")):
        if scale_rows:
            scale_rows.append(r"\addlinespace")
        path = ROOT / f"results/cross_layer_scale{suffix}_summary.json"
        scaling = json.loads(path.read_text())["summary"]
        for index, tier in enumerate(sorted(scaling, key=int)):
            panel = scaling[tier]
            operator = panel["operator_only"]
            joint = panel["joint"]
            local = panel["local_rewrite"]
            first, last = joint["range_tokens"]
            scale_rows.append(" & ".join((
                checkpoint if index == 0 else "",
                f"{first}--{last}",
                f"{operator['mean_total_application_bytes'] / 1e6:.3f}",
                f"{joint['mean_total_application_bytes'] / 1e6:.3f}",
                f"{operator['median_elapsed_ms']:.2f}",
                f"{joint['median_elapsed_ms']:.2f}",
                f"{local['median_elapsed_ms']:.2f}",
                str(operator["correct"]),
                str(joint["correct"]),
            )) + r" \\")
    return {
        "causal-sweep.dat": "\n".join(causal_rows) + "\n",
        "ablation-sweep.dat": "\n".join(ablation_rows) + "\n",
        "qk-rows.tex": "\n".join(qk_rows + [r"\bottomrule"]) + "\n",
        "composed-rows.tex": "\n".join(composed_rows + [r"\bottomrule"]) + "\n",
        "cross-layer-rows.tex": "\n".join(compiled_rows + [r"\bottomrule"]) + "\n",
        "scale-rows.tex": "\n".join(scale_rows + [r"\bottomrule"]) + "\n",
        "head-rows.tex": "\n".join(head_rows + [r"\bottomrule"]) + "\n",
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
