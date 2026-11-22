"""Independently check measured gate certificates, paired requests, and costs."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from check_cross_layer_evidence import DIGEST, payloads
from check_cross_layer_scale import LEVELS, PINNED, TRAFFIC, checkpoint_width

MODELS = {
    "70m": ("EleutherAI/pythia-70m-deduped", "cross_layer_scale_cases.csv"),
    "160m": ("EleutherAI/pythia-160m-deduped", "cross_layer_scale_160m_cases.csv"),
}
ARMS = ("operator_only", "gated")
CERTIFICATES = "layout_gate_benchmark_certificates.csv"
REQUESTS = "layout_gate_benchmark_requests.csv"
SUMMARY = "layout_gate_benchmark_summary.json"
CERT_INTS = (
    "extra_sentences", "repeat", "bindings", "native_forwards", "compile_ns", "native_ns",
    "certificate_ns", "original_tokens", "rewritten_tokens", "original_prefix",
    "rewritten_prefix", "admitted",
)
CERT_LISTS = ("targets", "original_choices", "rewritten_choices")
REQUEST_INTS = (
    "extra_sentences", "variant", "admitted", "tokens", "selected_prefix_tokens", "target",
    "native_choice", "choice", "elapsed_ns", "exact_ring", "qk_products", "rows_sent", *TRAFFIC,
)


def parse_rows(path, fields):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            row.update({key: int(row[key]) for key in fields})
            yield row


def key(row):
    return row["model"], row["extra_sentences"], row["record"]


def references(scale_dir):
    result = defaultdict(dict)
    for label, (_, filename) in MODELS.items():
        path = scale_dir / filename
        ints = ("extra_sentences", "variant", "target", "native_choice", "tokens",
                "selected_prefix_tokens", "public_prefix_tokens")
        for row in parse_rows(path, ints):
            arm = row["arm"]
            if arm in ("operator_only", "joint"):
                idx = label, row["extra_sentences"], row["record"], row["variant"]
                if arm in result[idx]:
                    raise ValueError(f"duplicate reference route at {idx}/{arm}")
                result[idx][arm] = row
    records = {idx[2] for idx in result}
    expected = {(model, tier, record, variant) for model in MODELS for tier in LEVELS
                for record in records for variant in range(4)}
    if len(records) != 12 or set(result) != expected or any(
        set(group) != {"operator_only", "joint"} for group in result.values()
    ):
        raise ValueError("pinned scale references are incomplete")
    return result


def certs_for(path, ref):
    by_schema = defaultdict(dict)
    for row in parse_rows(path, CERT_INTS):
        idx = key(row)
        repeat = row["repeat"]
        if repeat in by_schema[idx]:
            raise ValueError(f"duplicate certificate timing at {idx}/{repeat}")
        for name in CERT_LISTS:
            row[name] = json.loads(row[name])
        before = [ref[idx + (v,)]["operator_only"] for v in range(4)]
        after = [ref[idx + (v,)]["joint"] for v in range(4)]
        targets = [x["target"] for x in before]
        expected = (
            targets, [x["native_choice"] for x in before], [x["native_choice"] for x in after],
            before[0]["candidate_ids_sha256"], before[0]["tokens"], after[0]["tokens"],
            before[0]["public_prefix_tokens"], after[0]["public_prefix_tokens"],
        )
        fields = (*CERT_LISTS, "candidate_ids_sha256", "original_tokens", "rewritten_tokens",
                  "original_prefix", "rewritten_prefix")
        if tuple(row[name] for name in fields) != expected:
            raise ValueError(f"certificate does not check all four pinned public bindings at {idx}")
        if (row["original_tokens"] != row["rewritten_tokens"]
                or row["original_prefix"] >= row["rewritten_prefix"]):
            raise ValueError(f"rewrite does not reduce QK work for matched token counts at {idx}")
        admitted = int(row["original_choices"] == row["rewritten_choices"])
        if (row["admitted"], row["selected_layout"]) != (
                admitted, "rewritten" if admitted else "original"):
            raise ValueError(f"certificate admission contradicts its FP32 choices at {idx}")
        if (row["bindings"], row["native_forwards"]) != (4, 8):
            raise ValueError(f"wrong number of timed native forwards at {idx}")
        if (min(row["compile_ns"], row["native_ns"]) <= 0
                or row["certificate_ns"] < row["compile_ns"] + row["native_ns"]):
            raise ValueError(f"invalid certificate duration at {idx}/{repeat}")
        by_schema[idx][repeat] = row
    if set(by_schema) != {idx[:3] for idx in ref} or any(
        set(repeats) != {0, 1, 2} for repeats in by_schema.values()
    ):
        raise ValueError("expected 216 certificate runs with three repeats in 72 schema settings")
    return by_schema


def requests_for(path, ref, certs):
    by_binding = defaultdict(dict)
    for row in parse_rows(path, REQUEST_INTS):
        idx = key(row) + (row["variant"],)
        arm = row["arm"]
        if arm not in ARMS or arm in by_binding[idx]:
            raise ValueError(f"unknown or duplicate request arm at {idx}/{arm}")
        cert = certs[idx[:3]][0]
        chosen = "joint" if arm == "gated" and cert["admitted"] else "operator_only"
        expected = ref[idx][chosen]
        fields = ("layout", "tokens", "selected_prefix_tokens", "target", "native_choice",
                  "prompt_sha256", "candidate_ids_sha256")
        if (row["admitted"] != cert["admitted"]
                or any(row[field] != expected[field] for field in fields)
                or row["choice"] != expected["native_choice"]):
            raise ValueError(f"protected request changed the certified route or answer at {idx}/{arm}")
        counters = payloads(row["tokens"], row["selected_prefix_tokens"], rows=4)
        model = MODELS[row["model"]][0]
        width = PINNED[model][3]
        counters["head_bytes"] = 16 * width + 104
        if (any(row[field] != count for field, count in counters.items())
                or row["rows_sent"] != 4 or row["exact_ring"] != 1
                or row["elapsed_ns"] <= 0 or not DIGEST(row["score_sha256"])):
            raise ValueError(f"invalid payload, exactness evidence, or timing at {idx}/{arm}")
        by_binding[idx][arm] = row
    if set(by_binding) != set(ref) or any(set(arms) != set(ARMS) for arms in by_binding.values()):
        raise ValueError("expected 288 matched bindings and their 576 protected requests")
    for idx, arms in by_binding.items():
        before, after = (arms[arm] for arm in ARMS)
        if before["choice"] != after["choice"]:
            raise ValueError(f"gated execution changed the certified output at {idx}")
        if not after["admitted"] and before["score_sha256"] != after["score_sha256"]:
            raise ValueError(f"same-layout fallback changed exact scores at {idx}")
    return by_binding


def break_even(certificate_ns, paired_savings_sum_ns, bindings, admitted):
    if not admitted or paired_savings_sum_ns <= 0:
        return None
    return -(-certificate_ns * bindings // paired_savings_sum_ns)


def recompute_summary(certs, paired):
    summary = {}
    for idx in sorted(certs):
        repeats = certs[idx]
        duration = statistics.median(row["certificate_ns"] for row in repeats.values())
        groups = [paired[idx + (binding,)] for binding in range(4)]
        savings = sum(group["operator_only"]["elapsed_ns"] - group["gated"]["elapsed_ns"]
                      for group in groups)
        model, tier, record = idx
        schemas = summary.setdefault(model, {}).setdefault(str(tier), {"schemas": {}})["schemas"]
        admitted = repeats[0]["admitted"]
        schemas[record] = {
            "admitted": admitted,
            "median_certificate_ns": duration,
            "paired_requests": len(groups),
            "total_paired_saving_ns": savings,
            "projected_break_even_requests": break_even(duration, savings, len(groups), admitted),
        }
    for model, tiers in summary.items():
        for tier, panel in tiers.items():
            groups = [arms for idx, arms in paired.items() if idx[:2] == (model, int(tier))]
            schema_rows = panel["schemas"].values()
            panel.update({
                "admitted_schemas": sum(row["admitted"] for row in schema_rows),
                "prompts": len(groups),
                "same_choices": sum(arms["operator_only"]["choice"] == arms["gated"]["choice"]
                                    for arms in groups),
                "correct": {arm: sum(group[arm]["choice"] == group[arm]["target"]
                                     for group in groups) for arm in ARMS},
                "median_certificate_ms": statistics.median(row["median_certificate_ns"]
                                                           for row in schema_rows) / 1e6,
                "median_warm_ms": {arm: statistics.median(group[arm]["elapsed_ns"]
                                                         for group in groups) / 1e6 for arm in ARMS},
                "median_paired_baseline_over_gated": statistics.median(
                    group["operator_only"]["elapsed_ns"] / group["gated"]["elapsed_ns"]
                    for group in groups),
                "mean_total_application_bytes": {
                    arm: statistics.mean(sum(group[arm][field] for field in TRAFFIC)
                                         for group in groups) for arm in ARMS},
            })
    return summary


def check_metadata(metadata, directory, transport):
    names = {"tcp": "TCP loopback", "unix": "AF_UNIX socketpair"}
    if metadata["transport"] != names[transport] or metadata.get("replay_only"):
        raise ValueError("manifest is not a measured gate run on the expected transport")
    for filename, field in ((CERTIFICATES, "certificates_sha256"), (REQUESTS, "requests_sha256")):
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != metadata[field]:
            raise ValueError(f"raw digest disagrees with metadata for {filename}")
    if set(metadata["models"]) != set(MODELS):
        raise ValueError("expected both pinned checkpoints")
    for label, (name, _) in MODELS.items():
        manifest = metadata["models"][label]
        if manifest["model"] != name:
            raise ValueError(f"unexpected checkpoint for {label}")
        checkpoint_width({**manifest, "weight_sha256": manifest["artifact_sha256"]["model.safetensors"]})
    fixed_design = ("certificate_repeats", "discarded_certificate_warmups_per_model_tier",
                    "discarded_request_warmups_per_model_tier_per_arm", "ring_bits", "scale",
                    "layer_index", "head_index")
    if tuple(metadata[field] for field in fixed_design) != (3, 1, 1, 64, 1 << 16, 1, 0):
        raise ValueError("metadata does not match the registered gate run design")


def check(directory, scale_dir, transport):
    metadata = json.loads((directory / SUMMARY).read_text())
    check_metadata(metadata, directory, transport)
    ref = references(scale_dir)
    certs = certs_for(directory / CERTIFICATES, ref)
    paired = requests_for(directory / REQUESTS, ref, certs)
    summary = recompute_summary(certs, paired)
    if metadata["summary"] != summary:
        raise ValueError("reported gate outcomes or timing do not match measured row recomputation")
    for tiers in summary.values():
        for panel in tiers.values():
            schemas = list(panel.pop("schemas").values())
            finite = [row["projected_break_even_requests"] for row in schemas
                      if row["projected_break_even_requests"] is not None]
            panel["positive_admitted_savings"] = len(finite)
            panel["max_projected_break_even_requests"] = max(finite, default=None)
            panel["min_projected_break_even_requests"] = min(finite, default=None)
            panel["median_projected_break_even_requests"] = statistics.median(finite) if finite else None
    return {"certificate_runs": 3 * len(certs), "protected_requests": 2 * len(paired),
            "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--scale-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--transport", choices=("tcp", "unix"), default="tcp")
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.scale_dir, args.transport), indent=2))
