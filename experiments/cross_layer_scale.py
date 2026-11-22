"""Scale public annotations under the admitted rewrite and compare trusted-local baselines."""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from cross_layer_rewrite import ARMS as MPC_ARMS
from cross_layer_rewrite import BASE, HEAD_WIDTH, TASKS, digest_words, lift_note, render, task_plans
from local_lm_quality import MODEL, REVISION, WEIGHT_SHA256, verify_checkpoint
from pythia_composed import replace_scores
from pythia_qk_2pc import HEAD, LAYER, SCALE, workers
from pythia_secure_head import project, projection_workers, quantize

EXTRA_SENTENCES = (0, 3, 9)
PUBLIC_TEXT = (
    "The local office uses a standard filing form.",
    "The document remains available during normal office hours.",
    "Staff follow the usual procedure for updating records.",
    "The building has a separate desk for general inquiries.",
    "Scheduled maintenance follows the regular operational calendar.",
    "A copy of each blank form is available upon request.",
    "The information board lists all changes to office hours.",
    "The lobby is accessible through the main entrance.",
    "Printed instructions are provided at the reception desk.",
)
LOCAL_ARMS = ("local_original", "local_rewrite", "parser_original", "parser_rewrite")
ARMS = (*MPC_ARMS, *LOCAL_ARMS)
COUNTERS = ("qk_products", "qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes", "rows_sent")
VOCAB_SIZE = 50304


@dataclass(frozen=True)
class Checkpoint:
    model: str
    revision: str
    weight_sha256: str
    config_sha256: str
    hidden_width: int
    heads: int
    layers: int
    archive_stem: str


CHECKPOINTS = {
    "70m": Checkpoint(
        MODEL, REVISION, WEIGHT_SHA256,
        "002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43",
        512, 8, 6, "cross_layer_scale",
    ),
    "160m": Checkpoint(
        "EleutherAI/pythia-160m-deduped",
        "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e",
        "bbaae5c00917b163baa499fc8eb64859ee0c850c5fdecfc32f4d70dc07213575",
        "76eb275107220e450d31258f792a2efcbee109d8b62ae0088260057dec06362f",
        768, 12, 12, "cross_layer_scale_160m",
    ),
}


def checkpoint_artifacts(spec):
    return {
        "config.json": spec.config_sha256,
        "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
        "tokenizer_config.json": "70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f",
        "special_tokens_map.json": "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
        "model.safetensors": spec.weight_sha256,
    }


def validate_checkpoint(checkpoint, spec):
    for file_name, expected in checkpoint_artifacts(spec).items():
        digest = hashlib.sha256()
        with (checkpoint / file_name).open("rb") as stream:
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{file_name} does not match {spec.model} revision {spec.revision}")

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    shape = (config.model_type, config.vocab_size, config.hidden_size,
             config.num_attention_heads, config.num_hidden_layers)
    if shape != ("gpt_neox", VOCAB_SIZE, spec.hidden_width, spec.heads, spec.layers):
        raise ValueError(f"unexpected architecture for {spec.model}: {shape}")


def take_field(text, task, candidate_ids):
    """Read the field from a rendered record without consulting the benchmark target."""
    header, first, second, query = text.splitlines()
    if header != f"{task.name} record." or query != f"{task.field}:":
        raise ValueError("unexpected local parser record type")
    marker = f"{task.field}: "
    values = [line[len(marker):] for line in (first, second) if line.startswith(marker)]
    if len(values) != 1 or values[0] not in candidate_ids:
        raise ValueError("ambiguous or out-of-contract local parser field")
    return candidate_ids[values[0]]


def evaluate_arm(arm, prepared, body, layer, max_weight, vocab, qk_channels, head_channels):
    layout, prefix, tokens, text, authorized, candidate_weight, fixed_weight, ids = prepared
    start = time.perf_counter_ns()
    trace = []
    projected = None
    upload = download = rows_sent = 0
    fixed_hidden = None
    if arm.startswith("parser_"):
        choice = take_field(text, authorized[0], authorized[1])
        logits = None
    elif arm.startswith("local_"):
        hidden = body(tokens, use_cache=False).last_hidden_state[0, -1].numpy()
        logits = candidate_weight @ hidden
        choice = int(ids[logits.argmax()])
    else:
        with replace_scores(layer, qk_channels, prefix, trace):
            hidden = body(tokens, use_cache=False).last_hidden_state[0, -1].numpy()
        fixed_hidden = quantize(hidden)
        if int(np.abs(fixed_hidden).max()) * max_weight * fixed_hidden.size >= 1 << 63:
            raise ValueError("head projection can overflow signed int64")
        projected, upload, download, rows_sent, _ = project(
            head_channels, fixed_hidden, ids, vocab, True, True,
        )
        choice = int(ids[projected.argmax()])
        logits = projected.astype(np.float64) / SCALE**2
    elapsed = time.perf_counter_ns() - start

    # Exact validation is deliberately outside the measured path.
    metrics = dict.fromkeys(COUNTERS, 0)
    if arm in MPC_ARMS:
        if len(trace) != 1:
            raise ValueError("expected exactly one protected attention tile")
        qk, signed, ri, ci, counts = trace[0]
        if not np.array_equal(qk, (signed[0] @ signed[1].T)[ri, ci]):
            raise ValueError("protected attention score differs from exact ring reference")
        if not np.array_equal(projected, fixed_weight @ fixed_hidden):
            raise ValueError("protected head differs from exact ring reference")
        metrics = dict(
            qk_products=counts["beaver_multiplications"],
            qk_peer_bytes=counts["peer_payload_bytes"],
            qk_dealer_bytes=counts["dealer_bytes"],
            qk_client_bytes=counts["client_upload_bytes"] + counts["client_output_bytes"],
            head_bytes=upload + download,
            rows_sent=rows_sent,
        )
    return choice, logits, projected, elapsed, metrics


@torch.inference_mode()
def run(checkpoint, transport, spec=None):
    if spec is None:
        verify_checkpoint(checkpoint)
    else:
        validate_checkpoint(checkpoint, spec)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    body = model.base_model
    if model.config.model_type != "gpt_neox" or LAYER >= len(body.layers):
        raise ValueError("benchmark requires the selected GPT-NeoX layer")
    layer = body.layers[LAYER].attention
    if layer.head_size != HEAD_WIDTH or HEAD >= model.config.num_attention_heads:
        raise ValueError("attention shape does not match the fixed-width planner")
    weight = model.get_output_embeddings().weight.numpy()
    if weight.shape != (VOCAB_SIZE, model.config.hidden_size):
        raise ValueError("unexpected output-head matrix dimensions")
    fixed_weight = quantize(weight)
    max_weight = int(np.abs(fixed_weight).max())
    vocab = weight.shape[0]
    groups = []
    for n in EXTRA_SENTENCES:
        for base_task in TASKS:
            task = replace(base_task, note=" ".join((base_task.note, *PUBLIC_TEXT[:n])))
            ids, plans, best, _ = task_plans(tokenizer, task)
            groups.append((n, task, ids, plans, best))

    rng = random.Random(604281)
    rng.shuffle(groups)
    warmups = [next(item for item in groups if item[0] == n) for n in EXTRA_SENTENCES]
    rows = []
    with workers(transport) as qk_channels, projection_workers(fixed_weight, transport=transport) as head_channels:
        for group_number, (n, task, ids, plans, best) in enumerate((*warmups, *groups)):
            warmup = group_number < len(warmups)
            routing = {
                "full": ("original", False), "operator_only": ("original", True),
                "rewrite_only": ("rewritten", False), "joint": (best, True),
                "manual": ("manual", True), "local_original": ("original", False),
                "local_rewrite": ("rewritten", False), "parser_original": ("original", False),
                "parser_rewrite": ("rewritten", False),
            }
            if best != "rewritten":
                raise ValueError("fixture did not produce expected legal public-first minimum")
            allowed = dict(zip(task.values, map(int, ids)))
            candidate_weight = weight[ids]
            integer_weight = fixed_weight[ids]
            order_variants = [0] if warmup else rng.sample(range(len(ids)), len(ids))
            for variant in order_variants:
                order = list(ARMS)
                rng.shuffle(order)
                outputs = {}
                for arm in order:
                    layout, selective = routing[arm]
                    plan = plans[layout]
                    prefix = plan.prefix if selective else 0
                    text, _ = render(task, BASE if layout == "original" else lift_note(task, BASE), task.values[variant])
                    prepared = (layout, prefix, torch.tensor([plan.ids[variant]]), text, (task, allowed),
                                candidate_weight, integer_weight, ids)
                    outputs[arm] = evaluate_arm(
                        arm, prepared, body, layer, max_weight, vocab, qk_channels, head_channels,
                    )
                if warmup:
                    continue
                canonical_output = outputs["joint"][2]
                if not all(np.array_equal(canonical_output, outputs[a][2]) for a in ("manual", "rewrite_only")):
                    raise ValueError("rewritten layouts disagree across protected policies")
                if not np.array_equal(outputs["full"][2], outputs["operator_only"][2]):
                    raise ValueError("original layout disagrees across protected policies")
                for arm in ARMS:
                    layout, selective = routing[arm]
                    plan = plans[layout]
                    choice, logits, projected, elapsed, metrics = outputs[arm]
                    native = outputs["local_original" if layout == "original" else "local_rewrite"]
                    error = float(np.abs(logits - native[1]).max()) if logits is not None else ""
                    rows.append(dict(
                        extra_sentences=n, record=task.name, variant=variant, arm=arm, layout=layout,
                        tokens=plan.width, public_prefix_tokens=plan.prefix,
                        selected_prefix_tokens=plan.prefix if selective else 0,
                        target=int(ids[variant]), native_choice=native[0], choice=choice,
                        candidate_ids_sha256=digest_words(ids), prompt_sha256=digest_words(plan.ids[variant]),
                        token_multiset_sha256=digest_words(sorted(plan.ids[variant])),
                        protected_logit_sha256=digest_words(projected) if projected is not None else "",
                        exact_ring=int(arm in MPC_ARMS), elapsed_ns=elapsed, max_abs_logit_error=error,
                        **metrics,
                    ))
    return rows


def summarize(rows, transport, spec=None):
    cells = defaultdict(list)
    for row in rows:
        cells[row["extra_sentences"], row["arm"]].append(row)
    result = {}
    for n in EXTRA_SENTENCES:
        panels = result[str(n)] = {}
        for arm in ARMS:
            group = cells[n, arm]
            elapsed = [r["elapsed_ns"] / 1e6 for r in group]
            panels[arm] = {
                "prompts": len(group),
                "range_tokens": [min(r["tokens"] for r in group), max(r["tokens"] for r in group)],
                "mean_tokens": statistics.mean(r["tokens"] for r in group),
                "mean_public_prefix_tokens": statistics.mean(r["public_prefix_tokens"] for r in group),
                "correct": sum(r["choice"] == r["target"] for r in group),
                "native_agreement": sum(r["choice"] == r["native_choice"] for r in group),
                "exact_ring": sum(r["exact_ring"] for r in group),
                "median_elapsed_ms": statistics.median(elapsed),
                "p95_elapsed_ms": statistics.quantiles(elapsed, n=20, method="inclusive")[18],
                "mean_total_application_bytes": statistics.mean(
                    sum(r[k] for k in ("qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes"))
                    for r in group
                ),
                **{f"mean_{key}": statistics.mean(r[key] for r in group) for key in COUNTERS},
            }
            if arm in MPC_ARMS:
                panels[arm]["max_abs_logit_error"] = max(r["max_abs_logit_error"] for r in group)
        index = {(r["record"], r["variant"], r["arm"]): r for r in rows if r["extra_sentences"] == n}
        pairs = [(r, index[r["record"], r["variant"], "joint"]) for r in cells[n, "operator_only"]]
        panels["paired_operator_vs_joint"] = {
            "same_choices": sum(a["choice"] == b["choice"] for a, b in pairs),
            "correct_to_incorrect": sum(a["choice"] == a["target"] and b["choice"] != b["target"] for a, b in pairs),
            "incorrect_to_correct": sum(a["choice"] != a["target"] and b["choice"] == b["target"] for a, b in pairs),
            "median_elapsed_operator_over_joint": statistics.median(a["elapsed_ns"] / b["elapsed_ns"] for a, b in pairs),
        }
    metadata = {
        "model": spec.model if spec else MODEL,
        "revision": spec.revision if spec else REVISION,
        "weight_sha256": spec.weight_sha256 if spec else WEIGHT_SHA256,
        "platform": platform.platform(), "torch": str(torch.__version__),
        "transformers": transformers_version, "numpy": np.__version__, "threads": 2,
        "layer_index": LAYER, "head_index": HEAD, "ring_bits": 64, "scale": SCALE,
        "transport": "AF_UNIX socketpair" if transport == "unix" else "TCP loopback",
        "templates": len(TASKS), "bindings_per_template": len(TASKS[0].values),
        "warmup_requests_per_arm_per_note_length": 1,
        "timing_scope": {
            "mpc": "protected prefill and output projection, including dealer; no compilation, tokenization, model loading, connection setup or reference checks",
            "fp32": "pretokenized local eager backbone and same four-row native FP32 head projection and choice",
            "parser": "same rendered raw text with precompiled candidate map, parse and choice",
        },
        "summary": result,
    }
    if spec:
        metadata.update(
            hidden_width=spec.hidden_width, head_width=HEAD_WIDTH, vocab_size=VOCAB_SIZE,
            num_attention_heads=spec.heads, num_hidden_layers=spec.layers,
            artifact_sha256=checkpoint_artifacts(spec),
        )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("tcp", "unix"), default="tcp")
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), default="70m")
    args = parser.parse_args()
    spec = CHECKPOINTS[args.model]
    cases = run(args.checkpoint, args.transport, spec)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / f"{spec.archive_stem}_cases.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, cases[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    report = summarize(cases, args.transport, spec)
    report["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    (args.output / f"{spec.archive_stem}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(cases)} paired-policy runs on {report['transport']}")
