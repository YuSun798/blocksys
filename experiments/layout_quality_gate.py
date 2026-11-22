"""Replay a quality gate over the four public bindings of each typed record.

This analysis selects archived routes. It measures neither compilation cost
nor latency for a deployed gate. Usage: python layout_quality_gate.py CASES.csv.
"""
import argparse
import csv
import hashlib
import json
import statistics
import struct
from collections import defaultdict
from pathlib import Path

LEVELS = (0, 3, 9)
ROUTES = ("operator_only", "joint")
TRAFFIC = ("qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes")
INTEGER_COLUMNS = (
    "extra_sentences", "variant", "tokens", "public_prefix_tokens",
    "selected_prefix_tokens", "target", "native_choice", "choice", "qk_products",
    "rows_sent", "exact_ring", *TRAFFIC,
)


def read_pairs(source):
    pairs = defaultdict(dict)
    head_widths = set()
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"arm", "record", "layout", "candidate_ids_sha256",
                    "token_multiset_sha256", *INTEGER_COLUMNS}
        if missing := required.difference(reader.fieldnames or ()):
            raise ValueError(f"missing archive columns: {', '.join(sorted(missing))}")
        for raw in reader:
            route = raw["arm"]
            if route not in ROUTES:
                continue
            row = {**raw, **{key: int(raw[key]) for key in INTEGER_COLUMNS}}
            key = row["extra_sentences"], row["record"], row["variant"]
            if route in pairs[key]:
                raise ValueError(f"duplicate route {route} for {key}")
            length, prefix = row["tokens"], row["selected_prefix_tokens"]
            if not 0 <= prefix == row["public_prefix_tokens"] < length:
                raise ValueError(f"invalid disclosed prefix for {key}/{route}")
            suffix = length - prefix
            products = 64 * suffix * (suffix + 1) // 2
            if (row["qk_products"], row["qk_peer_bytes"], row["qk_dealer_bytes"],
                    row["qk_client_bytes"], row["rows_sent"], row["exact_ring"]) != (
                        products, 32 * products, 48 * products + 26,
                        2048 * length + 8 * length * (length + 1), 4, 1):
                raise ValueError(f"unexpected protected traffic or exactness for {key}/{route}")
            # Upload from the client to two workers includes a count and four
            # row IDs per worker. Each worker returns four 64-bit candidates.
            if row["head_bytes"] not in (8296, 12392):
                raise ValueError(f"unsupported head payload for {key}/{route}")
            head_widths.add((row["head_bytes"] - 104) // 16)
            if row["choice"] != row["native_choice"]:
                raise ValueError(f"archive requires protected/native choice agreement at {key}/{route}")
            pairs[key][route] = row

    records = {record for _, record, _ in pairs}
    if len(records) != 12 or len(head_widths) != 1:
        raise ValueError("expected twelve record schemas and one checkpoint head width")
    expected = {(tier, record, variant) for tier in LEVELS
                for record in records for variant in range(4)}
    if set(pairs) != expected or any(set(options) != set(ROUTES) for options in pairs.values()):
        raise ValueError("incomplete paired route coverage")
    static = ("tokens", "target", "candidate_ids_sha256", "token_multiset_sha256")
    for key, options in pairs.items():
        before, after = (options[route] for route in ROUTES)
        if any(before[field] != after[field] for field in static):
            raise ValueError(f"matched routes change the task or token inventory at {key}")
        if (before["layout"], after["layout"]) != ("original", "rewritten"):
            raise ValueError(f"unexpected paired layouts at {key}")
    return pairs, sorted(records), head_widths.pop()


def payload(row):
    return sum(row[field] for field in TRAFFIC)


def replay(source):
    pairs, records, hidden_width = read_pairs(source)
    result = {}
    candidate_targets = {
        record: [pairs[0, record, variant]["operator_only"]["target"] for variant in range(4)]
        for record in records
    }
    for tier in LEVELS:
        admissions = []
        original, rewritten, gated = [], [], []
        for record in records:
            bindings = [pairs[tier, record, variant] for variant in range(4)]
            before = [row["operator_only"] for row in bindings]
            after = [row["joint"] for row in bindings]
            targets = [row["target"] for row in before]
            schema_hash = hashlib.sha256(struct.pack("<4q", *targets)).hexdigest()
            if (targets != candidate_targets[record] or len(set(targets)) != 4
                    or any(row["candidate_ids_sha256"] != schema_hash
                           or row["native_choice"] not in targets for row in before + after)):
                raise ValueError(f"invalid public candidate schema at {tier}/{record}")
            for group in (before, after):
                if any(len({row[field] for row in group}) != 1
                       for field in ("tokens", "selected_prefix_tokens")):
                    raise ValueError(f"disclosed shape depends on binding at {tier}/{record}")
            # The decision inspects the entire public domain and is therefore
            # identical for all actual private bindings covered by the schema.
            allow = all(a["native_choice"] == b["native_choice"] for a, b in zip(before, after))
            admissions.append(allow)
            original.extend(before)
            rewritten.extend(after)
            gated.extend(after if allow else before)

        if any(a["choice"] != b["choice"] for a, b in zip(original, gated)):
            raise ValueError("gated choices differ from the original-layout baseline")
        correct = {key: sum(r["choice"] == r["target"] for r in rows)
                   for key, rows in (("operator_only", original), ("ungated_joint", rewritten),
                                     ("gated_joint", gated))}
        same_native_choices = sum(a["native_choice"] == b["native_choice"]
                                  for a, b in zip(original, rewritten))
        result[str(tier)] = {
            "schemas": len(admissions),
            "admitted_schemas": sum(admissions),
            "requests": len(original),
            "admitted_requests": 4 * sum(admissions),
            "native_choices_equal_before_gate": same_native_choices,
            "native_choice_comparison_fraction": same_native_choices / len(original),
            "correct": correct,
            "scored_candidate_rows_per_request": 4,
            "mean_total_application_bytes": {
                key: statistics.mean(map(payload, rows)) for key, rows in (
                    ("operator_only", original), ("ungated_joint", rewritten), ("gated_joint", gated))
            },
            "baseline_over_gated_ratio_of_summed_bytes": (
                sum(map(payload, original)) / sum(map(payload, gated))
            ),
            "baseline_over_gated_paired_median_byte_ratio": statistics.median(
                payload(a) / payload(b) for a, b in zip(original, gated)
            ),
        }
    return {"hidden_width": hidden_width, "replay_only": True, "summary": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.cases), indent=2))
