"""Measure finite-domain layout certification and paired warm protected requests."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from cross_layer_rewrite import BASE, TASKS, compile_plan, digest_words, lift_note
from cross_layer_scale import (
    CHECKPOINTS, EXTRA_SENTENCES, HEAD_WIDTH, PUBLIC_TEXT, VOCAB_SIZE,
    checkpoint_artifacts, evaluate_arm, validate_checkpoint,
)
from pythia_qk_2pc import HEAD, LAYER, SCALE, workers
from pythia_secure_head import projection_workers, quantize

ARMS = ("operator_only", "gated")
TRAFFIC = ("qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes")


@dataclass(frozen=True)
class Certificate:
    candidates: np.ndarray
    original: object
    rewritten: object
    original_choices: tuple
    rewritten_choices: tuple
    admitted: bool

    @property
    def selected_layout(self):
        return "rewritten" if self.admitted else "original"

    def choose(self, arm, variant):
        if not 0 <= variant < len(self.candidates):
            raise ValueError("binding lies outside the certified public domain")
        if arm == "gated" and self.admitted:
            return self.selected_layout, self.rewritten, self.rewritten_choices[variant]
        return "original", self.original, self.original_choices[variant]


def native_choices(body, plan, projection, ids):
    selected = []
    for tokens in plan.ids:
        hidden = body(torch.tensor([tokens]), use_cache=False).last_hidden_state[0, -1].numpy()
        logits = projection @ hidden
        selected.append(int(ids[logits.argmax()]))
    return tuple(selected)


def certify(tokenizer, body, weight, task):
    started = time.perf_counter_ns()
    target_rows = [tokenizer.encode(f" {value}", add_special_tokens=False) for value in task.values]
    if any(len(ids) != 1 for ids in target_rows):
        raise ValueError("candidate does not match the single-token schema")
    ids = np.asarray([row[0] for row in target_rows], dtype=np.int64)
    if len(set(ids)) != len(task.values):
        raise ValueError("candidate set contains a duplicate token")

    rewritten_order = lift_note(task, BASE)
    if rewritten_order == BASE:
        raise ValueError("fixture has no contract-eligible rewrite")
    original = compile_plan(tokenizer, task, BASE, ids)
    rewritten = compile_plan(tokenizer, task, rewritten_order, ids)
    if any(sorted(a) != sorted(b) for a, b in zip(original.ids, rewritten.ids)):
        raise ValueError("rewrite changed the token inventory")
    projection = weight[ids]
    compile_finished = time.perf_counter_ns()

    a = native_choices(body, original, projection, ids)
    b = native_choices(body, rewritten, projection, ids)
    native_finished = time.perf_counter_ns()
    admitted = a == b and rewritten.private_products < original.private_products
    certificate = Certificate(ids, original, rewritten, a, b, admitted)
    elapsed = time.perf_counter_ns() - started
    return certificate, (compile_finished - started, native_finished - compile_finished, elapsed)


def record_certificate(model, tier, task, repeat, certificate, timing):
    compile_ns, native_ns, certificate_ns = timing
    return {
        "model": model, "extra_sentences": tier, "record": task.name, "repeat": repeat,
        "bindings": len(certificate.candidates),
        "native_forwards": len(certificate.original_choices) + len(certificate.rewritten_choices),
        "compile_ns": compile_ns, "native_ns": native_ns, "certificate_ns": certificate_ns,
        "targets": json.dumps(certificate.candidates.tolist()),
        "original_choices": json.dumps(certificate.original_choices),
        "rewritten_choices": json.dumps(certificate.rewritten_choices),
        "candidate_ids_sha256": digest_words(certificate.candidates),
        "original_tokens": certificate.original.width,
        "rewritten_tokens": certificate.rewritten.width,
        "original_prefix": certificate.original.prefix,
        "rewritten_prefix": certificate.rewritten.prefix,
        "admitted": int(certificate.admitted), "selected_layout": certificate.selected_layout,
    }


def serve(model, tier, task, variant, certificate, order, body, layer, weight, fixed_weight,
          max_weight, qk_channels, head_channels):
    ids = certificate.candidates
    projection = weight[ids]
    fixed_projection = fixed_weight[ids]
    answers = {}
    for arm in order:
        started = time.perf_counter_ns()
        layout, plan, native = certificate.choose(arm, variant)
        tokens = torch.tensor([plan.ids[variant]])
        route_ns = time.perf_counter_ns() - started
        prepared = (layout, plan.prefix, tokens, "", None, projection, fixed_projection, ids)
        selected = "joint" if layout == "rewritten" else "operator_only"
        choice, _, logits, elapsed, metrics = evaluate_arm(
            selected, prepared, body, layer, max_weight, weight.shape[0],
            qk_channels, head_channels,
        )
        if native != choice:
            raise ValueError(f"protected choice differs from native certificate: {task.name}/{variant}/{arm}")
        answers[arm] = {
            "model": model, "extra_sentences": tier, "record": task.name, "variant": variant,
            "arm": arm, "layout": layout, "admitted": int(certificate.admitted),
            "tokens": plan.width, "selected_prefix_tokens": plan.prefix,
            "target": int(ids[variant]), "native_choice": native, "choice": choice,
            "elapsed_ns": elapsed + route_ns, "exact_ring": 1,
            "prompt_sha256": digest_words(plan.ids[variant]),
            "candidate_ids_sha256": digest_words(ids), "score_sha256": digest_words(logits),
            **metrics,
        }
    if answers["operator_only"]["choice"] != answers["gated"]["choice"]:
        raise ValueError("the gate changed a measured in-domain request choice")
    return [answers[arm] for arm in ARMS]


@torch.inference_mode()
def run_model(key, checkpoint, transport, repeats):
    spec = CHECKPOINTS[key]
    validate_checkpoint(checkpoint, spec)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    body = model.base_model
    layer = body.layers[LAYER].attention
    weight = model.get_output_embeddings().weight.numpy()
    if weight.shape != (VOCAB_SIZE, spec.hidden_width) or layer.head_size != HEAD_WIDTH:
        raise ValueError("selected checkpoint does not match the protected operator shapes")
    fixed_weight = quantize(weight)
    max_weight = int(np.abs(fixed_weight).max())
    groups = [
        (extra, replace(task, note=" ".join((task.note, *PUBLIC_TEXT[:extra]))))
        for extra in EXTRA_SENTENCES for task in TASKS
    ]
    warmups = [next(item for item in groups if item[0] == tier) for tier in EXTRA_SENTENCES]
    for _, task in warmups:
        certify(tokenizer, body, weight, task)

    # Every allowed binding is synthesized from public schema information here.
    # No actual measured request is dispatched until the certificates are sealed.
    rng = random.Random(171203)
    rng.shuffle(groups)
    cold_rows = []
    sealed = {}
    for tier, task in groups:
        prev = None
        for repeat in range(repeats):
            certificate, timing = certify(tokenizer, body, weight, task)
            verdict = (certificate.original_choices, certificate.rewritten_choices, certificate.admitted)
            if prev is not None and prev != verdict:
                raise ValueError("repeated native certificate decisions disagree")
            prev = verdict
            cold_rows.append(record_certificate(key, tier, task, repeat, certificate, timing))
        sealed[tier, task.name] = certificate

    warm_rows = []
    baseline_groups = rng.sample(groups, len(groups))
    with workers(transport) as qk_channels, projection_workers(fixed_weight, transport=transport) as head_channels:
        for tier, task in warmups:
            serve(key, tier, task, 0, sealed[tier, task.name], ARMS, body, layer, weight,
                  fixed_weight, max_weight, qk_channels, head_channels)
        for tier, task in baseline_groups:
            certificate = sealed[tier, task.name]
            for variant in rng.sample(range(len(task.values)), len(task.values)):
                order = rng.sample(ARMS, len(ARMS))
                warm_rows.extend(serve(
                    key, tier, task, variant, certificate, order, body, layer,
                    weight, fixed_weight, max_weight, qk_channels, head_channels,
                ))
    return cold_rows, warm_rows


def summarize(cold_rows, warm_rows):
    cold = defaultdict(list)
    warm = defaultdict(dict)
    for row in cold_rows:
        cold[row["model"], row["extra_sentences"], row["record"]].append(row)
    for row in warm_rows:
        warm[row["model"], row["extra_sentences"], row["record"], row["variant"]][row["arm"]] = row

    result = {}
    for key, rows in sorted(cold.items()):
        model, tier, task = key
        group = [arms for fields, arms in warm.items() if fields[:3] == key]
        pairs = [(arms["operator_only"], arms["gated"]) for arms in group]
        stats = result.setdefault(model, {}).setdefault(str(tier), {"schemas": {}})
        setup = statistics.median(row["certificate_ns"] for row in rows)
        saving = sum(a["elapsed_ns"] - b["elapsed_ns"] for a, b in pairs)
        admitted = rows[0]["admitted"]
        stats["schemas"][task] = {
            "admitted": admitted, "median_certificate_ns": setup, "paired_requests": len(pairs),
            "total_paired_saving_ns": saving,
            "projected_break_even_requests": (
                math.ceil(setup * len(pairs) / saving) if admitted and saving > 0 else None
            ),
        }
    for model, tiers in result.items():
        for tier, stats in tiers.items():
            keys = [(key, arms) for key, arms in warm.items() if key[:2] == (model, int(tier))]
            pairs = [(arms["operator_only"], arms["gated"]) for _, arms in keys]
            stats.update(
                admitted_schemas=sum(row["admitted"] for row in stats["schemas"].values()),
                prompts=len(pairs),
                same_choices=sum(a["choice"] == b["choice"] for a, b in pairs),
                correct={arm: sum(r["choice"] == r["target"] for r in (arms[arm] for _, arms in keys))
                         for arm in ARMS},
                median_certificate_ms=statistics.median(r["median_certificate_ns"]
                                                         for r in stats["schemas"].values()) / 1e6,
                median_warm_ms={
                    arm: statistics.median(arms[arm]["elapsed_ns"] for _, arms in keys) / 1e6
                    for arm in ARMS
                },
                median_paired_baseline_over_gated=statistics.median(
                    a["elapsed_ns"] / b["elapsed_ns"] for a, b in pairs
                ),
                mean_total_application_bytes={
                    arm: statistics.mean(sum(arms[arm][field] for field in TRAFFIC) for _, arms in keys)
                    for arm in ARMS
                },
            )
    return result


def archive(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-70m", type=Path)
    parser.add_argument("--checkpoint-160m", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("tcp", "unix"), default="tcp")
    parser.add_argument("--certificate-repeats", type=int, default=3)
    args = parser.parse_args()
    checkpoints = {key: path for key, path in (
        ("70m", args.checkpoint_70m), ("160m", args.checkpoint_160m),
    ) if path is not None}
    if not checkpoints or args.certificate_repeats < 1:
        parser.error("provide at least one checkpoint and one certificate repeat")
    cold = []
    warm = []
    for key, checkpoint in checkpoints.items():
        cold_rows, warm_rows = run_model(key, checkpoint, args.transport, args.certificate_repeats)
        cold.extend(cold_rows)
        warm.extend(warm_rows)
        print(f"{key}: {len(cold_rows)} executed certificates, {len(warm_rows)} paired-policy requests",
              flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    cert_path = args.output / "layout_gate_benchmark_certificates.csv"
    requests_path = args.output / "layout_gate_benchmark_requests.csv"
    summary = {
        "models": {
            key: {
                "model": spec.model, "revision": spec.revision, "hidden_width": spec.hidden_width,
                "head_width": HEAD_WIDTH, "artifact_sha256": checkpoint_artifacts(spec),
            }
            for key in checkpoints for spec in (CHECKPOINTS[key],)
        },
        "transport": "TCP loopback" if args.transport == "tcp" else "AF_UNIX socketpair",
        "platform": platform.platform(), "python": platform.python_version(), "threads": 2,
        "torch": str(torch.__version__), "numpy": np.__version__, "transformers": transformers_version,
        "ring_bits": 64, "scale": SCALE, "layer_index": LAYER, "head_index": HEAD,
        "certificate_repeats": args.certificate_repeats,
        "discarded_certificate_warmups_per_model_tier": 1,
        "discarded_request_warmups_per_model_tier_per_arm": 1,
        "timing_scope": {
            "certificate": "schema preparation, all eight native forwards and gate comparison; checkpoint and tokenizer loading excluded",
            "requests": "finite-domain lookup, token tensor preparation, protected prefill and output projection including dealer; connections and independent ring checks excluded",
            "amortization": "projection from per-schema median certificate and mean paired warm saving; reported only for admitted routes with positive savings",
        },
        "certificates_sha256": archive(cert_path, cold), "requests_sha256": archive(requests_path, warm),
        "summary": summarize(cold, warm),
    }
    (args.output / "layout_gate_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
