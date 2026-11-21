"""Run a typed public-note lift through a pretrained forward and protected operators."""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from local_lm_quality import MODEL, REVISION, WEIGHT_SHA256, verify_checkpoint
from pythia_composed import replace_scores
from pythia_qk_2pc import HEAD, LAYER, SCALE, workers
from pythia_secure_head import project, projection_workers, quantize

HEAD_WIDTH = 64
BASE = ("private_field", "public_note")
ARMS = ("full", "operator_only", "rewrite_only", "joint", "manual")


@dataclass(frozen=True)
class Task:
    name: str
    field: str
    note: str
    values: tuple
    order_insensitive: bool = True
    note_public: bool = True


# These fixtures are disjoint from the original synthetic Pythia probe.
TASKS = (
    Task("Passport", "City", "The document is valid for international travel.", ("Tokyo", "Paris", "Berlin", "London")),
    Task("Orchestra", "Instrument", "The musicians meet for practice tonight.", ("piano", "violin", "guitar", "drum")),
    Task("Cinema", "Genre", "The theater announces the next screening tomorrow.", ("comedy", "drama", "action", "horror")),
    Task("Cafe", "Drink", "Orders close before the lunch break.", ("coffee", "tea", "water", "juice")),
    Task("Shelter", "Animal", "The shelter accepts new visitors this afternoon.", ("dog", "cat", "rabbit", "bird")),
    Task("Museum", "Material", "Staff record the item before the exhibit opens.", ("glass", "wood", "stone", "metal")),
    Task("Tournament", "Sport", "The referee confirms the match schedule.", ("tennis", "football", "basketball", "baseball")),
    Task("Bakery", "Flavor", "The chef plans a new batch for the market.", ("vanilla", "chocolate", "lemon", "cinnamon")),
    Task("Classroom", "Language", "The teacher distributed the worksheets.", ("English", "French", "Spanish", "German")),
    Task("Exhibit", "Shade", "The room opens to guests this evening.", ("white", "green", "pink", "yellow")),
    Task("Airfield", "Country", "The flight arrives before midday today.", ("Canada", "France", "Japan", "Brazil")),
    Task("Campsite", "Weather", "Guests pick up their welcome packets at reception.", ("rain", "snow", "sun", "wind")),
)


@dataclass(frozen=True)
class Plan:
    ids: tuple
    prefix: int
    width: int

    @property
    def private_products(self):
        tail = self.width - self.prefix
        return HEAD_WIDTH * tail * (tail + 1) // 2


def lift_note(task, order):
    # The contract applies to a field-binding task over an unordered record.
    # Moving a private annotation or an order-sensitive row is ineligible.
    if order != BASE or not task.order_insensitive or not task.note_public:
        return order
    return ("public_note", "private_field")


def render(task, order, value):
    lines = {"private_field": f"{task.field}: {value}", "public_note": task.note}
    header = f"{task.name} record.\n"
    note_offset = len(task.note) + 1 if order[0] == "public_note" else 0
    start = len(header) + note_offset + len(task.field) + 1
    return header + "\n".join(lines[row] for row in order) + f"\n{task.field}:", start


def manual_render(task, order, value):
    if order != ("public_note", "private_field"):
        raise ValueError("manual baseline requires the public-first layout")
    before_value = f"{task.name} record.\n{task.note}\n{task.field}:"
    return f"{before_value} {value}\n{task.field}:", len(before_value)


def verify_record(text, task, value):
    header, first, second, query = text.splitlines()
    if header != f"{task.name} record." or query != f"{task.field}:":
        raise ValueError("template changed the task or its query")
    if {first, second} != {task.note, f"{task.field}: {value}"}:
        raise ValueError("template changed the typed record contents")


def compile_plan(tokenizer, task, order, targets, renderer=render):
    encodings = []
    prefixes = []
    for value, target in zip(task.values, targets):
        text, start = renderer(task, order, value)
        verify_record(text, task, value)
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        finish = start + 1 + len(value)
        slots = [i for i, (l, r) in enumerate(encoding.offset_mapping) if r > start and l < finish]
        if len(slots) != 1 or encoding.input_ids[slots[0]] != target:
            raise ValueError(f"private slot is not one aligned candidate token in {task.name}: {value}")
        encodings.append(tuple(encoding.input_ids))
        prefixes.append(slots[0])
    prefix = prefixes[0]
    if len(set(prefixes)) != 1 or any(len(seq) != len(encodings[0]) for seq in encodings):
        raise ValueError(f"candidate binding changed the public shape in {task.name}")
    if any(seq[:prefix] != encodings[0][:prefix] or seq[prefix + 1:] != encodings[0][prefix + 1:]
           for seq in encodings):
        raise ValueError(f"candidate binding contaminated another token in {task.name}")
    return Plan(tuple(encodings), prefix, len(encodings[0]))


def task_plans(tokenizer, task):
    if lift_note(replace(task, note_public=False), BASE) != BASE:
        raise ValueError("rewrite accepted a private note")
    if lift_note(replace(task, order_insensitive=False), BASE) != BASE:
        raise ValueError("rewrite accepted an order-sensitive task")

    start = time.perf_counter_ns()
    target_seq = [tokenizer.encode(f" {value}", add_special_tokens=False) for value in task.values]
    if any(len(ids) != 1 for ids in target_seq):
        raise ValueError(f"candidate must be one token in {task.name}")
    answer_ids = np.asarray([ids[0] for ids in target_seq])
    if len(set(answer_ids)) != 4:
        raise ValueError(f"duplicate candidate token in {task.name}")
    target_ns = time.perf_counter_ns() - start

    started = time.perf_counter_ns()
    original = compile_plan(tokenizer, task, BASE, answer_ids)
    original_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    order = lift_note(task, BASE)
    if order == BASE:
        raise ValueError("public fixtures require a type-admissible note lift")
    rewritten = compile_plan(tokenizer, task, order, answer_ids)
    rewrite_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    plans = {"original": original, "rewritten": rewritten}
    if any(sorted(a) != sorted(b) for a, b in zip(original.ids, rewritten.ids)):
        raise ValueError(f"the rewrite changed token inventory in {task.name}")
    # These costs are based on public specifications over all possible
    # bindings. The actual field value cannot affect the chosen layout.
    best = min(plans, key=lambda key: plans[key].private_products)
    joint_ns = original_ns + rewrite_ns + time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    plans["manual"] = compile_plan(tokenizer, task, order, answer_ids, renderer=manual_render)
    manual_ns = time.perf_counter_ns() - started
    costs = (original_ns, original_ns, rewrite_ns, joint_ns, manual_ns)
    planning_ns = {arm: target_ns + cost for arm, cost in zip(ARMS, costs)}
    return answer_ids, plans, best, planning_ns


def digest_words(words):
    return hashlib.sha256(np.asarray(words, dtype="<i8").tobytes()).hexdigest()


@torch.inference_mode()
def run(checkpoint, transport):
    verify_checkpoint(checkpoint)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    body = model.base_model
    layer = body.layers[LAYER].attention
    weight = model.get_output_embeddings().weight.numpy()
    weight_int = quantize(weight)
    max_weight = int(np.abs(weight_int).max())
    compiled = [(task, task_plans(tokenizer, task)) for task in TASKS]
    rows = []
    rng = random.Random(20271121)
    with workers(transport) as qk_channels, projection_workers(weight_int, transport=transport) as head_channels:
        for warmup, (task, (ids, plans, best, schema_ns)) in enumerate((compiled[0], *compiled), -1):
            candidate_hash = digest_words(ids)
            variants = {
                "full": ("original", False),
                "operator_only": ("original", True),
                "rewrite_only": ("rewritten", False),
                "joint": (best, True),
                "manual": ("manual", True),
            }
            # Manual and automatic layout construction are distinct artifact paths.
            # Their traffic must tie whenever both choose the same legal serialization.
            candidate_weight = weight[ids]
            integer_weight = weight_int[ids]
            for variant, answer_id in enumerate(ids):
                originals = {}
                for plan in plans.values():
                    bound = plan.ids[variant]
                    if bound not in originals:
                        h = body(torch.tensor([bound]), use_cache=False).last_hidden_state[0, -1].numpy()
                        originals[bound] = candidate_weight @ h
                fixed_outputs = {}
                order = list(ARMS)
                rng.shuffle(order)
                for arm in order:
                    layout, selective = variants[arm]
                    plan = plans[layout]
                    token_ids = plan.ids[variant]
                    tokens = torch.tensor([token_ids])
                    reference = originals[token_ids]
                    prefix = plan.prefix if selective else 0
                    trace = []
                    start = time.perf_counter_ns()
                    with replace_scores(layer, qk_channels, prefix, trace):
                        h = body(tokens, use_cache=False).last_hidden_state[0, -1].numpy()
                    fixed_h = quantize(h)
                    if int(np.abs(fixed_h).max()) * max_weight * fixed_h.size >= 1 << 63:
                        raise ValueError("selected projection can overflow signed int64")
                    projected, upload, download, rows_sent, _ = project(
                        head_channels, fixed_h, ids, weight_int.shape[0], True, True,
                    )
                    elapsed = time.perf_counter_ns() - start
                    if len(trace) != 1:
                        raise ValueError("expected exactly one protected attention tile")
                    scores, signed_qk, ri, ci, counts = trace[0]
                    if not np.array_equal(scores, (signed_qk[0] @ signed_qk[1].T)[ri, ci]):
                        raise ValueError("protected attention score differs from the ring reference")
                    if not np.array_equal(projected, integer_weight @ fixed_h):
                        raise ValueError("protected output projection differs from the ring reference")
                    fingerprint = digest_words(projected)
                    if plan.ids in fixed_outputs and fixed_outputs[plan.ids] != fingerprint:
                        raise ValueError("a disclosure policy changed the result for a fixed layout")
                    fixed_outputs[plan.ids] = fingerprint

                    if warmup >= 0:
                        rows.append({
                            "record": task.name, "variant": variant, "arm": arm, "layout": layout,
                            "tokens": plan.width, "public_prefix_tokens": plan.prefix,
                            "note_moved": int(layout != "original"), "legal_layouts": 2,
                            "candidate_ids_sha256": candidate_hash, "prompt_sha256": digest_words(token_ids),
                            "token_multiset_sha256": digest_words(sorted(token_ids)),
                            "selected_prefix_tokens": prefix, "target": int(answer_id),
                            "fp32_choice": int(ids[reference.argmax()]),
                            "protected_choice": int(ids[projected.argmax()]), "score_sha256": fingerprint,
                            "qk_products": counts["beaver_multiplications"],
                            "qk_peer_bytes": counts["peer_payload_bytes"],
                            "qk_dealer_bytes": counts["dealer_bytes"],
                            "qk_client_bytes": counts["client_upload_bytes"] + counts["client_output_bytes"],
                            "head_bytes": upload + download, "rows_sent": rows_sent,
                            "elapsed_ns": elapsed, "schema_planning_ns": schema_ns[arm],
                            "max_abs_logit_error": float(np.abs(projected.astype(np.float64) / SCALE**2 - reference).max()),
                        })
    return rows


def summarize(rows, transport):
    summary = {}
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "prompts": len(group),
            "layouts": {layout: sum(r["layout"] == layout for r in group)
                        for layout in sorted({r["layout"] for r in group})},
            "fp32_correct": sum(row["fp32_choice"] == row["target"] for row in group),
            "protected_correct": sum(row["protected_choice"] == row["target"] for row in group),
            "fp32_agreement": sum(row["protected_choice"] == row["fp32_choice"] for row in group),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in group),
            "median_elapsed_ms": statistics.median(row["elapsed_ns"] for row in group) / 1e6,
            "median_schema_planning_ms": statistics.median(row["schema_planning_ns"] for row in group) / 1e6,
            **{f"mean_{key}": statistics.mean(row[key] for row in group)
               for key in ("tokens", "selected_prefix_tokens", "qk_products", "qk_peer_bytes",
                           "qk_dealer_bytes", "qk_client_bytes", "head_bytes")},
        }
    index = {(row["record"], row["variant"], row["arm"]): row for row in rows}
    pairs = [(r, index[r["record"], r["variant"], "joint"])
             for r in rows if r["arm"] == "operator_only"]
    return {
        "model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
        "platform": platform.platform(), "torch": str(torch.__version__),
        "transformers": transformers_version, "numpy": np.__version__, "threads": 2,
        "layer_index": LAYER, "head_index": HEAD, "ring_bits": 64, "scale": SCALE,
        "transport": "AF_UNIX socketpair" if transport == "unix" else "TCP loopback",
        "templates_disjoint_from_probe": len(TASKS), "candidates_per_template": 4,
        "warmup_requests_per_arm": 4, "all_qk_and_head_outputs_exact_ring": True,
        "head_row_policy": "same four authorized rows in every arm",
        "scheme": "typed public-note lift, QK placement ablation, and manual public-first baseline",
        "typed_contract_refusals": {"private_note_templates": len(TASKS), "order_sensitive_templates": len(TASKS)},
        "operator_only_vs_joint": {
            "same_fp32_choices": sum(a["fp32_choice"] == b["fp32_choice"] for a, b in pairs),
            "same_protected_choices": sum(a["protected_choice"] == b["protected_choice"] for a, b in pairs),
            "same_fixed_point_scores": sum(a["score_sha256"] == b["score_sha256"] for a, b in pairs),
            "correct_to_incorrect": sum(a["protected_choice"] == a["target"] and b["protected_choice"] != b["target"]
                                        for a, b in pairs),
            "incorrect_to_correct": sum(a["protected_choice"] != a["target"] and b["protected_choice"] == b["target"]
                                        for a, b in pairs),
        },
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("tcp", "unix"), default="unix")
    args = parser.parse_args()
    rows = run(args.checkpoint, args.transport)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "cross_layer_rewrite_cases.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows, args.transport)
    report["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    summary = json.dumps(report, indent=2) + "\n"
    (args.output / "cross_layer_rewrite_summary.json").write_text(summary)
    print(summary, end="")
