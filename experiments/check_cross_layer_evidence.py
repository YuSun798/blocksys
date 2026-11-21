"""Verify matched Pythia layout arms, fixed-point outcomes, and application payloads."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("full", "operator_only", "rewrite_only", "joint", "manual")
DIGEST_FIELDS = ("token_multiset_sha256", "candidate_ids_sha256")
DIGEST = re.compile(r"[a-f0-9]{64}").fullmatch
FIELDS_AS_INTEGERS = (
    "variant", "tokens", "public_prefix_tokens", "selected_prefix_tokens", "target",
    "fp32_choice", "protected_choice", "qk_products", "qk_peer_bytes",
    "qk_dealer_bytes", "qk_client_bytes", "head_bytes", "rows_sent", "elapsed_ns",
    "note_moved", "legal_layouts", "schema_planning_ns",
)


def payloads(tokens, prefix, rows):
    """Calculate protocol bytes directly from the measured request dimensions."""
    residual = tokens - prefix
    products = 64 * residual * (residual + 1) // 2
    scores = tokens * (tokens + 1) // 2
    return {
        "qk_products": products,
        "qk_peer_bytes": 32 * products,
        "qk_dealer_bytes": 48 * products + 26,
        "qk_client_bytes": 32 * 64 * tokens + 16 * scores,
        "head_bytes": 2 * (4 + 4 * rows + 8 * 512) + 16 * (rows or 50304),
    }


def total_payload(row):
    return sum(row[key] for key in ("qk_peer_bytes", "qk_dealer_bytes",
                                   "qk_client_bytes", "head_bytes"))


def validate_row(row):
    key = row["record"], row["variant"], row["arm"]
    n, allowed, selected = (row[key] for key in (
        "tokens", "public_prefix_tokens", "selected_prefix_tokens"))
    rows = row["rows_sent"]
    if not (0 <= selected <= allowed < n and selected in (0, allowed)):
        raise ValueError(f"invalid public prefix or unauthorized selection: {key}")
    if (rows != 4 or row["elapsed_ns"] <= 0 or row["schema_planning_ns"] <= 0
            or not all(DIGEST(row[field]) for field in (
                *DIGEST_FIELDS, "prompt_sha256", "score_sha256"))):
        raise ValueError(f"invalid score fingerprint or transport fields: {key}")
    if (row["layout"] not in ("original", "rewritten", "manual") or row["legal_layouts"] != 2
            or row["note_moved"] != (row["layout"] != "original")):
        raise ValueError(f"invalid recorded layout or public-note metadata: {key}")
    error = row["max_abs_logit_error"]
    if not (math.isfinite(error) and error >= 0):
        raise ValueError(f"invalid fixed-point score error: {key}")
    if any(row[name] != size for name, size in payloads(n, selected, rows).items()):
        raise ValueError(f"application payload does not match protocol accounting: {key}")


def validate_pair(key, arms):
    if set(arms) != set(ARMS):
        raise ValueError(f"missing policy arms for {key}")
    full, operator, rewrite, joint, manual = (arms[name] for name in ARMS)
    for invariant in ("tokens", "target"):
        if len({arm[invariant] for arm in arms.values()}) != 1:
            raise ValueError(f"matched arms changed {invariant}: {key}")
    for invariant in DIGEST_FIELDS:
        digests = {arm[invariant] for arm in arms.values()}
        if len(digests) != 1:
            raise ValueError(f"matched arms changed {invariant}: {key}")

    for row in (full, rewrite):
        if row["selected_prefix_tokens"] != 0:
            raise ValueError(f"all-shared attention arm changed execution scope: {key}")
    for row in (operator, joint, manual):
        if row["selected_prefix_tokens"] != row["public_prefix_tokens"]:
            raise ValueError(f"selective arm changed authorized operator scope: {key}")

    # These two arms differ only in execution. Semantic equivalence of a
    # cross-layout answer does not require equality of its floating logits.
    if (full["layout"], operator["layout"], rewrite["layout"], manual["layout"]) != (
        "original", "original", "rewritten", "manual"
    ):
        raise ValueError(f"control changed prompt layout: {key}")
    grouped = defaultdict(list)
    for row in arms.values():
        layout = "rewritten" if row["layout"] == "manual" else row["layout"]
        grouped[layout].append(row)
    for group in grouped.values():
        for invariant in ("prompt_sha256", "public_prefix_tokens", "score_sha256",
                          "fp32_choice", "protected_choice"):
            if len({row[invariant] for row in group}) != 1:
                raise ValueError(f"same-layout policies changed {invariant}: {key}")
    if full["prompt_sha256"] == rewrite["prompt_sha256"]:
        raise ValueError(f"measured rewrite retained the original token order: {key}")
    if full["schema_planning_ns"] != operator["schema_planning_ns"]:
        raise ValueError(f"matched original artifact has inconsistent planning time: {key}")

    if joint["layout"] not in (full["layout"], rewrite["layout"]):
        raise ValueError(f"joint policy selected a layout outside the measured candidates: {key}")
    restricted_rewrite = sum(value for name, value in payloads(
        rewrite["tokens"], rewrite["public_prefix_tokens"], 4).items() if name != "qk_products")
    if total_payload(joint) != min(total_payload(operator), restricted_rewrite):
        raise ValueError(f"joint policy did not minimize application payload: {key}")


def verify_summary(case_path, summary_path, by_case, transport=None):
    report = json.loads(summary_path.read_text())
    if report["raw_sha256"] != hashlib.sha256(case_path.read_bytes()).hexdigest():
        raise ValueError("raw archive SHA-256 does not match summary")
    names = {"tcp": "TCP loopback", "unix": "AF_UNIX socketpair"}
    if transport is not None and report["transport"] != names[transport]:
        raise ValueError("archive records a different socket transport")
    records = {record for record, _ in by_case}
    refusals = dict.fromkeys(("private_note_templates", "order_sensitive_templates"), len(records))
    if (report["templates_disjoint_from_probe"], report["candidates_per_template"],
            report["typed_contract_refusals"], report["ring_bits"], report["scale"],
            report["layer_index"], report["head_index"]) != (
                len(records), 4, refusals, 64, 1 << 16, 1, 0):
        raise ValueError("experiment metadata does not match the typed task and protocol")
    if not report["all_qk_and_head_outputs_exact_ring"]:
        raise ValueError("runner did not certify exact integer comparisons")
    if report["head_row_policy"] != "same four authorized rows in every arm":
        raise ValueError("head-row scope must be fixed across the layout ablation")
    for name in ARMS:
        group = [arms[name] for arms in by_case.values()]
        expected = {
            "prompts": len(group),
            "layouts": dict(Counter(row["layout"] for row in group)),
            "fp32_correct": sum(row["target"] == row["fp32_choice"] for row in group),
            "protected_correct": sum(row["target"] == row["protected_choice"] for row in group),
            "fp32_agreement": sum(row["fp32_choice"] == row["protected_choice"] for row in group),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in group),
            "median_elapsed_ms": statistics.median(row["elapsed_ns"] for row in group) / 1e6,
            "median_schema_planning_ms": statistics.median(row["schema_planning_ns"]
                                                          for row in group) / 1e6,
            **{f"mean_{key}": statistics.mean(row[key] for row in group)
               for key in ("tokens", "selected_prefix_tokens", "qk_products", "qk_peer_bytes",
                           "qk_dealer_bytes", "qk_client_bytes", "head_bytes")},
        }
        if report["summary"][name] != expected:
            raise ValueError(f"summary differs from per-case records: {name}")
    pairs = [(arms["operator_only"], arms["joint"]) for arms in by_case.values()]
    paired = {
        "same_fp32_choices": sum(a["fp32_choice"] == b["fp32_choice"] for a, b in pairs),
        "same_protected_choices": sum(a["protected_choice"] == b["protected_choice"] for a, b in pairs),
        "same_fixed_point_scores": sum(a["score_sha256"] == b["score_sha256"] for a, b in pairs),
        "correct_to_incorrect": sum(a["protected_choice"] == a["target"] != b["protected_choice"]
                                    for a, b in pairs),
        "incorrect_to_correct": sum(a["protected_choice"] != a["target"] == b["protected_choice"]
                                    for a, b in pairs),
    }
    if report["operator_only_vs_joint"] != paired:
        raise ValueError("paired score and accuracy summary differs from per-case records")


def verify(case_path, summary_path=None, transport=None):
    by_case = defaultdict(dict)
    with case_path.open(newline="") as source:
        for row in csv.DictReader(source):
            row.update({field: int(row[field]) for field in FIELDS_AS_INTEGERS})
            row["max_abs_logit_error"] = float(row["max_abs_logit_error"])
            validate_row(row)
            key = row["record"], row["variant"]
            if row["arm"] in by_case[key]:
                raise ValueError(f"duplicate arm: {key}, {row['arm']}")
            by_case[key][row["arm"]] = row
    records = {record for record, _ in by_case}
    if len(records) != 12 or set(by_case) != {
        (record, variant) for record in records for variant in range(4)
    }:
        raise ValueError("expected four requests for each of twelve templates")
    for key, arms in by_case.items():
        validate_pair(key, arms)
    for record in records:
        variants = [by_case[record, i] for i in range(4)]
        target_ids = [arms["full"]["target"] for arms in variants]
        targets = set(target_ids)
        if len(targets) != 4 or any(
            row[choice] not in targets for arms in variants for row in arms.values()
            for choice in ("fp32_choice", "protected_choice")
        ):
            raise ValueError(f"changed candidate contract for {record}")
        candidate_hash = hashlib.sha256(struct.pack("<4q", *target_ids)).hexdigest()
        for arm in ARMS:
            group = [arms[arm] for arms in variants]
            if any(row["candidate_ids_sha256"] != candidate_hash for row in group):
                raise ValueError(f"candidate digest differs from shared schema: {record}, {arm}")
            for invariant in ("layout", "tokens", "public_prefix_tokens", "schema_planning_ns"):
                if len({row[invariant] for row in group}) != 1:
                    raise ValueError(f"private binding changed static {invariant}: {record}, {arm}")

    if summary_path:
        verify_summary(case_path, summary_path, by_case, transport)
    elif transport:
        raise ValueError("transport verification requires a summary")

    transitions = Counter()
    layouts_changed = 0
    ratios = []
    opening_ratios = []
    for arms in by_case.values():
        old, new = arms["operator_only"], arms["joint"]
        transition = old["protected_choice"] == old["target"], new["protected_choice"] == new["target"]
        transitions[transition] += 1
        layouts_changed += old["layout"] != new["layout"]
        ratios.append(total_payload(old) / total_payload(new))
        opening_ratios.append(old["qk_peer_bytes"] / new["qk_peer_bytes"])
    report = {
        "matched_records": len(by_case),
        "archive_runs": len(by_case) * len(ARMS),
        "joint_changed_layout": layouts_changed,
        "operator_only_to_joint_accuracy": {
            f"{'correct' if before else 'wrong'}_to_{'correct' if after else 'wrong'}": transitions[before, after]
            for before in (False, True) for after in (False, True)
        },
        "operator_only_over_joint_paired_median_payload_ratio": statistics.median(ratios),
        "operator_only_over_joint_paired_median_opening_ratio": statistics.median(opening_ratios),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--transport", choices=("tcp", "unix"))
    args = parser.parse_args()
    print(json.dumps(verify(args.cases, args.summary, args.transport), indent=2))
