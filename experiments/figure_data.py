"""Derive the plotted values from the archived CPU results."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from check_sst2_layout import check as check_sst2
from loopback_2pc import WIDTHS
from run_sst2_qwen3 import verify_archive as check_qwen3

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
    scale_summaries = {}
    for checkpoint, suffix in (("70M", ""), ("160M", "_160m")):
        if scale_rows:
            scale_rows.append(r"\addlinespace")
        path = ROOT / f"results/cross_layer_scale{suffix}_summary.json"
        scaling = json.loads(path.read_text())["summary"]
        scale_summaries[checkpoint.lower()] = scaling
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
    path = ROOT / "results/layout_gate_benchmark_summary.json"
    gate = json.loads(path.read_text())["summary"]
    gate_rows = []
    for model in ("70m", "160m"):
        if gate_rows:
            gate_rows.append(r"\addlinespace")
        for index, tier in enumerate(sorted(gate[model], key=int)):
            row = gate[model][tier]
            projections = [
                stats["projected_break_even_requests"] for stats in row["schemas"].values()
                if stats["projected_break_even_requests"] is not None
            ]
            first, last = scale_summaries[model][tier]["joint"]["range_tokens"]
            gate_rows.append(" & ".join((
                model.upper() if index == 0 else "",
                f"{first}--{last}",
                f"{row['admitted_schemas']}/{len(row['schemas'])}",
                str(len(projections)),
                f"{row['median_certificate_ms']:.2f}",
                f"{row['median_paired_baseline_over_gated']:.3f}",
                f"{statistics.median(projections):g}" if projections else r"\textemdash",
            )) + r" \\")
    sst2 = check_sst2(ROOT / "results" / "sst2-full-dev", require_full_tcp=True)
    qwen3 = check_qwen3(ROOT / "results" / "sst2-qwen3")["results"]["full_dev"]
    sample_size = sst2["records"]
    if qwen3["examples"] != sample_size:
        raise ValueError("SST-2 table checkpoints must share the full development split")
    model_rows = [
        ("Pythia-" + key.upper(), sst2["models"][key])
        for key in ("70m", "160m")
    ]
    model_rows.append(("Qwen3-0.6B", {
        **qwen3["paired"],
        "original_correct": qwen3["original"]["native_correct"],
        "rewritten_correct": qwen3["rewritten"]["native_correct"],
    }))
    sst2_rows = []
    for model, row in model_rows:
        delta = 100 * row["paired_accuracy_difference"]
        lower, upper = (100 * bound for bound in row["paired_bootstrap_95_interval"])
        sst2_rows.append(" & ".join((
            model,
            f"{row['original_correct']}/{sample_size}",
            f"{row['rewritten_correct']}/{sample_size}",
            str(row["incorrect_to_correct"]),
            str(row["correct_to_incorrect"]),
            rf"${delta:+.1f}\,[{lower:.1f},\,{upper:.1f}]$",
        )) + r" \\")
    return {
        "causal-sweep.dat": "\n".join(causal_rows) + "\n",
        "ablation-sweep.dat": "\n".join(ablation_rows) + "\n",
        "qk-rows.tex": "\n".join(qk_rows + [r"\bottomrule"]) + "\n",
        "composed-rows.tex": "\n".join(composed_rows + [r"\bottomrule"]) + "\n",
        "cross-layer-rows.tex": "\n".join(compiled_rows + [r"\bottomrule"]) + "\n",
        "scale-rows.tex": "\n".join(scale_rows + [r"\bottomrule"]) + "\n",
        "gate-rows.tex": "\n".join(gate_rows + [r"\bottomrule"]) + "\n",
        "sst2-rows.tex": "\n".join(sst2_rows + [r"\bottomrule"]) + "\n",
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
