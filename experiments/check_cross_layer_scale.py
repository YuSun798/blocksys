"""Check scale-study pairing, protocol accounting, and local-baseline evidence."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from collections import defaultdict
from pathlib import Path

from check_cross_layer_evidence import ARMS as PROTECTED_ARMS, DIGEST, payloads

LEVELS = (0, 3, 9)
ARMS = (*PROTECTED_ARMS, "local_original", "local_rewrite", "parser_original", "parser_rewrite")
TRAFFIC = ("qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes")
COUNT_FIELDS = (*TRAFFIC, "qk_products", "selected_prefix_tokens", "rows_sent")


def total(row):
    return sum(row[name] for name in TRAFFIC)


def analyze(case_path, manifest):
    trial_rows = defaultdict(dict)
    with case_path.open(newline="") as source:
        for raw in csv.DictReader(source):
            row = {
                **raw, **{key: int(raw[key]) for key in (
                    *COUNT_FIELDS, "extra_sentences", "variant", "tokens", "public_prefix_tokens",
                    "target", "choice", "native_choice", "elapsed_ns", "exact_ring",
                )},
            }
            key = row["extra_sentences"], row["record"], row["variant"]
            if row["arm"] in trial_rows[key]:
                raise ValueError(f"duplicate arm at {key}")
            trial_rows[key][row["arm"]] = row
    records = {record for _, record, _ in trial_rows}
    if len(records) != 12 or set(trial_rows) != {
        (level, record, variant) for level in LEVELS for record in records for variant in range(4)
    }:
        raise ValueError("expected 12 templates, four bindings and three public-note lengths")

    targets = {record: [trial_rows[0, record, variant]["full"]["target"] for variant in range(4)]
               for record in records}
    if any(len(set(ids)) != 4 for ids in targets.values()):
        raise ValueError("each schema must fix four distinct authorized candidates")
    for setting in LEVELS:
        for record in records:
            bindings = [trial_rows[setting, record, variant] for variant in range(4)]
            for name in ARMS:
                for field in ("tokens", "public_prefix_tokens", "selected_prefix_tokens"):
                    if len({group[name][field] for group in bindings}) != 1:
                        raise ValueError(f"binding changed {field} at {setting}/{record}/{name}")
    for key, group in trial_rows.items():
        if set(group) != set(ARMS):
            raise ValueError(f"missing matched baseline at {key}")
        expected = targets[key[1]][key[2]]
        for name, row in group.items():
            if row["target"] != expected or row["choice"] not in targets[key[1]] or row["elapsed_ns"] <= 0:
                raise ValueError(f"target, choice or timing is invalid at {key}/{name}")
            if not 0 <= row["public_prefix_tokens"] < row["tokens"]:
                raise ValueError(f"disclosure prefix outside prompt at {key}/{name}")
        for field in ("tokens", "candidate_ids_sha256", "token_multiset_sha256"):
            if len({row[field] for row in group.values()}) != 1:
                raise ValueError(f"layout changed paired {field} at {key}")
        expected_candidates = hashlib.sha256(struct.pack("<4q", *targets[key[1]])).hexdigest()
        if group["full"]["candidate_ids_sha256"] != expected_candidates or any(
            not DIGEST(row["prompt_sha256"]) or not DIGEST(row["token_multiset_sha256"])
            for row in group.values()
        ):
            raise ValueError(f"invalid schema/prompt fingerprint at {key}")

        full, operator, rewrite, joint, manual = (group[name] for name in PROTECTED_ARMS)
        if (tuple(row["layout"] for row in (full, operator, rewrite, joint, manual)) != (
                "original", "original", "rewritten", "rewritten", "manual")
                or operator["public_prefix_tokens"] >= joint["public_prefix_tokens"]
                or operator["prompt_sha256"] == joint["prompt_sha256"]):
            raise ValueError(f"protected arm has the wrong rewrite or prefix at {key}")
        for row in (full, operator, rewrite, joint, manual):
            selective = row["arm"] not in ("full", "rewrite_only")
            prefix = row["public_prefix_tokens"] if selective else 0
            expected_counts = payloads(row["tokens"], prefix, rows=4)
            if (row["rows_sent"] != 4 or row["selected_prefix_tokens"] != prefix
                    or any(row[field] != count for field, count in expected_counts.items())
                    or not DIGEST(row["protected_logit_sha256"]) or row["exact_ring"] != 1):
                raise ValueError(f"unequal output scope or payload mismatch at {key}/{row['arm']}")
            error = float(row["max_abs_logit_error"])
            if not math.isfinite(error) or error < 0:
                raise ValueError(f"invalid deviation from FP32 at {key}/{row['arm']}")
        if total(joint) != min(total(operator), total(manual)):
            raise ValueError(f"joint did not choose the minimum-traffic eligible small plan at {key}")
        for layout, secure, native, parser in (
            ("original", (full, operator), group["local_original"], group["parser_original"]),
            ("rewrite", (rewrite, joint, manual), group["local_rewrite"], group["parser_rewrite"]),
        ):
            if any(row["prompt_sha256"] != native["prompt_sha256"] for row in secure) or (
                    len({row["protected_logit_sha256"] for row in secure}) != 1
                    or len({row["choice"] for row in secure}) != 1):
                raise ValueError(f"same-layout protected/native reference disagrees at {key}/{layout}")
            if parser["prompt_sha256"] != native["prompt_sha256"] or parser["choice"] != expected:
                raise ValueError(f"parser did not resolve the exact same typed input at {key}/{layout}")
            if any(row["native_choice"] != native["choice"] or
                   row["public_prefix_tokens"] != native["public_prefix_tokens"]
                   for row in (*secure, native, parser)):
                raise ValueError(f"inconsistent native reference or policy prefix at {key}/{layout}")
            if any(row[field] for row in (native, parser) for field in COUNT_FIELDS):
                raise ValueError(f"local baseline has nonzero network work at {key}/{layout}")
            if any(row["protected_logit_sha256"] or row["exact_ring"] for row in (native, parser)):
                raise ValueError(f"local baseline incorrectly marked as a protected run at {key}/{layout}")

    levels = {}
    for level in LEVELS:
        trials = [group for (setting, _, _), group in trial_rows.items() if setting == level]
        arms = {}
        for name in ARMS:
            rows = [group[name] for group in trials]
            elapsed = [row["elapsed_ns"] / 1e6 for row in rows]
            expected = {
                "prompts": len(rows),
                "range_tokens": [min(row["tokens"] for row in rows), max(row["tokens"] for row in rows)],
                "mean_tokens": statistics.mean(row["tokens"] for row in rows),
                "mean_public_prefix_tokens": statistics.mean(row["public_prefix_tokens"] for row in rows),
                "correct": sum(row["choice"] == row["target"] for row in rows),
                "native_agreement": sum(row["choice"] == row["native_choice"] for row in rows),
                "exact_ring": sum(row["exact_ring"] for row in rows),
                "median_elapsed_ms": statistics.median(elapsed),
                "p95_elapsed_ms": statistics.quantiles(elapsed, n=20, method="inclusive")[18],
                "mean_total_application_bytes": statistics.mean(total(row) for row in rows),
                **{f"mean_{name}": statistics.mean(row[name] for row in rows)
                   for name in (*TRAFFIC, "qk_products", "rows_sent")},
            }
            if name in PROTECTED_ARMS:
                expected["max_abs_logit_error"] = max(float(row["max_abs_logit_error"]) for row in rows)
            if expected != manifest["summary"][str(level)][name]:
                raise ValueError(f"archive summary differs for length setting {level}/{name}")
            arms[name] = {
                "count": len(rows),
                "correct": expected["correct"], "median_ms": expected["median_elapsed_ms"],
                "mean_peer": expected["mean_qk_peer_bytes"],
                "mean_dealer": expected["mean_qk_dealer_bytes"],
                "mean_qk_client": expected["mean_qk_client_bytes"],
                "mean_total": expected["mean_total_application_bytes"],
            }
        paired = [(group["operator_only"], group["joint"]) for group in trials]
        transitions = {
            "same_choices": sum(a["choice"] == b["choice"] for a, b in paired),
            "correct_to_incorrect": sum(a["choice"] == a["target"] != b["choice"] for a, b in paired),
            "incorrect_to_correct": sum(a["choice"] != a["target"] == b["choice"] for a, b in paired),
            "median_elapsed_operator_over_joint": statistics.median(
                a["elapsed_ns"] / b["elapsed_ns"] for a, b in paired),
        }
        if transitions != manifest["summary"][str(level)]["paired_operator_vs_joint"]:
            raise ValueError(f"archive paired comparison differs for length setting {level}")
        levels[level] = {
            "min_tokens": min(group["full"]["tokens"] for group in trials),
            "max_tokens": max(group["full"]["tokens"] for group in trials),
            "median_operator_over_joint_payload": statistics.median(
                total(group["operator_only"]) / total(group["joint"]) for group in trials),
            "median_joint_over_local_rewrite_time": statistics.median(
                group["joint"]["elapsed_ns"] / group["local_rewrite"]["elapsed_ns"] for group in trials),
            "arms": arms,
        }
    return levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--transport", choices=("tcp", "unix"), default="tcp")
    args = parser.parse_args()
    metadata = json.loads(args.summary.read_text())
    if metadata["raw_sha256"] != hashlib.sha256(args.cases.read_bytes()).hexdigest():
        raise ValueError("scale archive SHA-256 does not match metadata")
    names = {"tcp": "TCP loopback", "unix": "AF_UNIX socketpair"}
    if metadata["transport"] != names[args.transport]:
        raise ValueError("scale archive transport does not match requested transport")
    if (metadata["templates"], metadata["bindings_per_template"], metadata["ring_bits"],
            metadata["scale"], metadata["warmup_requests_per_arm_per_note_length"],
            metadata["layer_index"], metadata["head_index"]) != (12, 4, 64, 1 << 16, 1, 1, 0):
        raise ValueError("scale archive metadata does not match study design")
    print(json.dumps(analyze(args.cases, metadata), indent=2))
