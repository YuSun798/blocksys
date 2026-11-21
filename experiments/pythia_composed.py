"""Feed two-party attention scores through an actual Pythia forward and shared head."""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
from transformers.models.gpt_neox import modeling_gpt_neox as neox

from local_lm_quality import MODEL, REVISION, WEIGHT_SHA256, matched_prompts, verify_checkpoint
from pythia_head_inputs import answer_tokens
from pythia_qk_2pc import HEAD, LAYER, SCALE, evaluate_scores, workers
from pythia_secure_head import project, projection_workers, quantize

ARMS = {
    "full": (False, False),
    "qk_only": (True, False),
    "rows_only": (False, True),
    "joint": (True, True),
    "private_pool": (True, False),
}


@contextmanager
def replace_scores(target, channels, prefix, traces, one_way_s=0):
    eager = neox.eager_attention_forward

    def mixed(module, query, key, value, attention_mask, scaling, dropout=0., head_mask=None, **kwargs):
        if module is not target:
            return eager(module, query, key, value, attention_mask, scaling, dropout, head_mask, **kwargs)
        if query.shape[0] != 1 or query.shape != key.shape or module.training or dropout or head_mask is not None:
            raise ValueError("selected layer requires one inference prompt with complete prefill")
        qk = torch.stack((query[0, HEAD], key[0, HEAD])).numpy().astype(np.float64)
        signed = np.rint(qk * SCALE).astype(np.int64)
        length, width = signed.shape[1:]
        if int(np.abs(signed[0]).max()) * int(np.abs(signed[1]).max()) * width >= 1 << 63:
            raise ValueError("signed fixed-point bound exceeded")
        restored, counts = evaluate_scores(channels, signed.view(np.uint64), prefix, one_way_s)
        ri, ci = np.tril_indices(length)
        rest = [index for index in range(query.shape[1]) if index != HEAD]
        scores = query.new_zeros((1, query.shape[1], length, length))
        scores[:, rest] = (query[:, rest] @ key[:, rest].transpose(2, 3)) * scaling
        scores[0, HEAD].fill_(float("-inf"))
        scores[0, HEAD, ri, ci] = torch.from_numpy(restored.astype(np.float64) * (scaling / SCALE**2)).to(query.dtype)
        if attention_mask is None:
            future = torch.ones(length, length, dtype=torch.bool).triu(1)
            scores.masked_fill_(future, float("-inf"))
        else:
            scores = scores + attention_mask[:, :, :, :length]
        probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        traces.append((restored, signed, ri, ci, counts))
        return (probabilities @ value).transpose(1, 2).contiguous(), probabilities

    neox.eager_attention_forward = mixed
    try:
        yield
    finally:
        neox.eager_attention_forward = eager


@torch.inference_mode()
def run(checkpoint, one_way_s=0, arms=None):
    arms = tuple(ARMS if arms is None else arms)
    verify_checkpoint(checkpoint)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    choices = answer_tokens(tokenizer)
    weight = model.get_output_embeddings().weight.numpy()
    weight_int = quantize(weight)
    max_weight = int(np.abs(weight_int).max())
    body = model.base_model
    layer = body.layers[LAYER].attention
    prompts = list(matched_prompts(tokenizer))
    rng = random.Random(711)
    cases = []
    with workers("tcp", one_way_s) as score_channels, projection_workers(weight_int, one_way_s) as head_channels:
        for pair_id, (record, placement, _, prefix, pair) in enumerate([prompts[0], *prompts], -1):
            ids = choices[record].numpy()
            selected_weight = weight[ids]
            quant_rows = weight_int[ids]
            for variant in range(len(pair)):
                inputs = pair[variant:variant + 1]
                expected_hidden = body(inputs, use_cache=False).last_hidden_state[0, -1].numpy()
                expected_logits = selected_weight @ expected_hidden
                target = int(inputs[0, prefix])
                order = list(arms)
                rng.shuffle(order)
                fingerprints = set()
                for arm in order:
                    use_prefix, send_rows = ARMS[arm]
                    trace = []
                    start = time.perf_counter_ns()
                    with replace_scores(layer, score_channels, prefix if use_prefix else 0, trace, one_way_s):
                        last = body(inputs, use_cache=False).last_hidden_state[0, -1].numpy()
                    hidden = quantize(last)
                    if int(np.abs(hidden).max()) * max_weight * hidden.size >= 1 << 63:
                        raise ValueError("head projection exceeds signed range")
                    logits, upload, download, rows_sent, _ = project(
                        head_channels, hidden, ids, weight_int.shape[0],
                        send_rows or arm == "private_pool", arm != "private_pool", one_way_s)
                    elapsed = time.perf_counter_ns() - start
                    if len(trace) != 1:
                        raise ValueError("must execute exactly one protected QK tile")
                    result, signed, ri, ci, counts = trace[0]
                    if not np.array_equal(result, (signed[0] @ signed[1].T)[ri, ci]):
                        raise ValueError("protected QK result differs from fixed-point reference")
                    if not np.array_equal(logits, quant_rows @ hidden):
                        raise ValueError("protected head result differs from fixed-point reference")
                    fingerprint = hashlib.sha256(logits.astype("<i8").tobytes()).hexdigest()
                    fingerprints.add(fingerprint)
                    if pair_id < 0:
                        continue
                    cases.append({
                        "pair": pair_id, "variant": variant, "placement": placement,
                        "tokens": inputs.shape[1], "approved_prefix": prefix, "arm": arm,
                        "qk_products": counts["beaver_multiplications"],
                        "qk_peer_bytes": counts["peer_payload_bytes"],
                        "qk_dealer_bytes": counts["dealer_bytes"],
                        "qk_client_bytes": counts["client_upload_bytes"] + counts["client_output_bytes"],
                        "head_bytes": upload + download, "rows_sent": rows_sent,
                        "elapsed_ns": elapsed, "score_sha256": fingerprint, "reference": int(ids[expected_logits.argmax()]),
                        "protected": int(ids[logits.argmax()]), "target": target,
                        "max_abs_logit_error": float(
                            np.abs(logits.astype(np.float64) / SCALE**2 - expected_logits).max()
                        ),
                    })
                if len(fingerprints) != 1:
                    raise ValueError("disclosure mode changed the exact fixed-point output")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "pythia_composed_cases.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "prompts": len(group),
            "fp32_choices": sum(row["reference"] == row["protected"] for row in group),
            "correct_choices": sum(row["target"] == row["protected"] for row in group),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in group),
            "median_elapsed_ms": statistics.median(row["elapsed_ns"] for row in group) / 1e6,
            **{f"median_{key}": statistics.median(row[key] for row in group)
               for key in ("qk_client_bytes", "qk_dealer_bytes", "qk_peer_bytes", "head_bytes")},
        }
    output = {"model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
              "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "summary": summary,
              "platform": platform.platform(), "torch": str(torch.__version__),
              "transformers": transformers_version, "numpy": np.__version__, "threads": 2,
              "warmups_per_arm": 2, "transport": "TCP loopback"}
    (args.output / "pythia_composed_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
