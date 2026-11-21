"""Evaluate one real Pythia attention-score head with two-party Beaver multiplication.

The client computes Q/K and assigns disclosure labels. Workers evaluate the
causal score tile and return shares only. The client also acts as the trusted
dealer. This is a fixed-point operator benchmark on synthetic prompts.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import random
import secrets
import socket
import statistics
import struct
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from loopback_2pc import TIMEOUT, receive, tcp_pair

LAYER = 1
HEAD = 0
SCALE = 1 << 16
WIRE = np.dtype(">u8")
HEADER = struct.Struct("!III")
COUNTER = struct.Struct("!Q")
ARMS = ("all_shared", "selective")


def serialize(values):
    return np.asarray(values, dtype=np.uint64).astype(WIRE).tobytes()


def receive_words(channel, count):
    return np.frombuffer(receive(channel, count * WIRE.itemsize), dtype=WIRE).astype(np.uint64)


def uniform(shape):
    count = int(np.prod(shape))
    return np.frombuffer(secrets.token_bytes(count * WIRE.itemsize), dtype=np.uint64).reshape(shape)


def additive_shares(values):
    first = uniform(values.shape)
    return first, values - first


def causal_cells(length, public_prefix):
    rows, cols = np.tril_indices(length)
    return rows, cols, cols >= public_prefix


def score_worker(index, control, peer, one_way_s=0):
    with control, peer:
        control.settimeout(TIMEOUT)
        peer.settimeout(TIMEOUT)
        while True:
            length, width, public_prefix = HEADER.unpack(receive(control, HEADER.size))
            if length == 0:
                return
            rows, cols, private_products = causal_cells(length, public_prefix)
            count = int(private_products.sum()) * width
            a, b, c = receive_words(control, 3 * count).reshape(3, count)
            if one_way_s:
                time.sleep(one_way_s)
            control.sendall(b"R")
            q, k = receive_words(control, 2 * length * width).reshape(2, length, width)

            left, right = q[rows[private_products]], k[cols[private_products]]
            masked = np.stack((left.reshape(-1) - a, right.reshape(-1) - b))
            packet = serialize(masked)
            if index == 0:
                if one_way_s and packet:
                    time.sleep(one_way_s)
                peer.sendall(packet)
                opened = receive_words(peer, 2 * count).reshape(2, count)
            else:
                opened = receive_words(peer, 2 * count).reshape(2, count)
                if one_way_s and packet:
                    time.sleep(one_way_s)
                peer.sendall(packet)
            d, e = masked + opened
            product_shares = c + d * b + e * a
            if index == 0:
                product_shares += d * e

            out = np.zeros(len(rows), dtype=np.uint64)
            out[private_products] = product_shares.reshape(-1, width).sum(axis=1)
            # Public-key dot products are linear in the query shares. Party 0
            # alone contributes the fully public score when the query is public.
            local = (cols < public_prefix) & ((rows >= public_prefix) | (index == 0))
            out[local] = (q[rows[local]] * k[cols[local]]).sum(axis=1)
            if one_way_s:
                time.sleep(one_way_s)
            control.sendall(serialize(out) + COUNTER.pack(len(packet)))


@contextmanager
def workers(transport, one_way_s=0):
    pair = tcp_pair if transport == "tcp" else socket.socketpair
    p0_link, p1_link = pair()
    client0, control0 = pair()
    client1, control1 = pair()
    channels = client0, client1
    ctx = mp.get_context("spawn")
    parties = (
        ctx.Process(target=score_worker, args=(0, control0, p0_link, one_way_s)),
        ctx.Process(target=score_worker, args=(1, control1, p1_link, one_way_s)),
    )
    try:
        for worker in parties:
            worker.start()
        for endpoint in (control0, control1, p0_link, p1_link):
            endpoint.close()
        for channel in channels:
            channel.settimeout(TIMEOUT)
        yield channels
        for channel in channels:
            channel.sendall(HEADER.pack(0, 0, 0))
    finally:
        for endpoint in (*channels, control0, control1, p0_link, p1_link):
            endpoint.close()
        for worker in parties:
            if worker.pid is not None:
                worker.join(TIMEOUT)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                if worker.exitcode:
                    raise RuntimeError(f"score worker failed with exit code {worker.exitcode}")


def evaluate_scores(channels, qk, public_prefix, one_way_s=0):
    _, length, width = qk.shape
    rows, _, private_products = causal_cells(length, public_prefix)
    n_scores = len(rows)
    count = int(private_products.sum()) * width
    header = HEADER.pack(length, width, public_prefix)

    prep_start = time.perf_counter_ns()
    a, b = uniform((2, count))
    prep = additive_shares(np.stack((a, b, a * b)))
    if one_way_s:
        time.sleep(one_way_s)
    for channel, share in zip(channels, prep):
        channel.sendall(header + serialize(share))
    for channel in channels:
        if receive(channel, 1) != b"R":
            raise ValueError("unexpected dealer acknowledgment")
    prep_ns = time.perf_counter_ns() - prep_start

    online_start = time.perf_counter_ns()
    first, right = additive_shares(qk)
    left = first.copy()
    left[:, :public_prefix] = qk[:, :public_prefix]
    right[:, :public_prefix] = qk[:, :public_prefix]
    if one_way_s:
        time.sleep(one_way_s)
    for channel, values in zip(channels, (left, right)):
        channel.sendall(serialize(values))
    result0, bytes0 = read_result(channels[0], n_scores)
    result1, bytes1 = read_result(channels[1], n_scores)
    result = (result0 + result1).view(np.int64)
    online_ns = time.perf_counter_ns() - online_start
    peer_bytes = bytes0 + bytes1
    if peer_bytes != 4 * WIRE.itemsize * count:
        raise ValueError("unexpected masked-opening byte count")
    return result, {
        "beaver_multiplications": count,
        "peer_payload_bytes": peer_bytes,
        "client_upload_bytes": 4 * length * width * WIRE.itemsize,
        "client_output_bytes": 2 * n_scores * WIRE.itemsize,
        "dealer_bytes": 2 * (HEADER.size + 3 * count * WIRE.itemsize + 1),
        "preprocessing_us": prep_ns / 1000,
        "online_us": online_ns / 1000,
    }


def read_result(channel, count):
    return receive_words(channel, count), COUNTER.unpack(receive(channel, COUNTER.size))[0]


def collect_inputs(checkpoint, pairs):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    from local_lm_quality import MODEL, REVISION, WEIGHT_SHA256, matched_prompts, verify_checkpoint

    verify_checkpoint(checkpoint)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    captured = None

    def capture(module, args, kwargs):
        nonlocal captured
        hidden = args[0]
        qkv = module.query_key_value(hidden).view(*hidden.shape[:-1], -1, 3 * module.head_size)
        q, k, _ = qkv.transpose(1, 2).chunk(3, dim=-1)
        q, k = apply_rotary_pos_emb(q, k, *kwargs["position_embeddings"])
        captured = torch.stack((q[:, HEAD], k[:, HEAD]), dim=1)

    handle = model.gpt_neox.layers[LAYER].attention.register_forward_pre_hook(
        capture, with_kwargs=True,
    )
    inputs = []
    try:
        with torch.inference_mode():
            for idx, (record, placement, _, public_len, ids) in enumerate(matched_prompts(tokenizer)):
                if idx >= 2 * pairs:
                    break
                outputs = model(ids, use_cache=False, output_attentions=True)
                q, k = captured.unbind(dim=1)
                length, width = q.shape[1:]
                if not torch.equal(captured[0, :, :public_len], captured[1, :, :public_len]):
                    raise ValueError("Public-prefix post-RoPE Q/K differ within a matched pair")
                future = torch.ones(length, length, dtype=torch.bool).triu(1)
                logits = (q @ k.transpose(1, 2)) / width ** 0.5
                expected_attention = logits.masked_fill(future, float("-inf")).softmax(dim=-1)
                attention_error = (
                    expected_attention - outputs.attentions[LAYER][:, HEAD]
                ).abs().max().item()
                if attention_error > 1e-6:
                    raise ValueError("Captured Q/K do not reproduce the eager attention weights")

                ri, ci = np.tril_indices(length)
                for variant, qk in enumerate(captured.numpy()):
                    qk_fp64 = qk.astype(np.float64)
                    if not np.isfinite(qk_fp64).all():
                        raise ValueError("Q/K contains a nonfinite coordinate")
                    signed = np.rint(qk_fp64 * SCALE).astype(np.int64)
                    bound = int(np.abs(signed[0]).max()) * int(np.abs(signed[1]).max()) * width
                    if bound >= 1 << 63:
                        raise ValueError("Unscaled attention score can overflow signed 64-bit")
                    ring = (signed[0] @ signed[1].T)[ri, ci]
                    float_scores = (qk_fp64[0, ri] * qk_fp64[1, ci]).sum(axis=1) / width ** 0.5
                    # Bound the error from nearest quantization by the maximum
                    # coordinate magnitudes, with margin for the float64 reference.
                    quantum = 0.5 / SCALE
                    tolerance = (width ** 0.5 * quantum *
                                 (np.abs(qk_fp64[0]).max() + np.abs(qk_fp64[1]).max() + quantum))
                    inputs.append(({
                        "pair_id": idx // 2, "record": record, "placement": placement,
                        "variant": variant, "tokens": length, "public_prefix_tokens": public_len,
                        "causal_scores": len(ri), "reference_attention_error": attention_error,
                    }, signed.view(np.uint64), ring, float_scores, tolerance))
    finally:
        handle.remove()
    if len(inputs) != 4 * pairs:
        raise ValueError("Not enough matched prompts")
    return inputs, {
        "model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "scale": SCALE, "layer_index": LAYER, "head_index": HEAD,
    }


def run(inputs, transport, warmups):
    order = random.Random(1997)
    rows = []
    with workers(transport) as channels:
        for idx in range(warmups):
            case, qk, _, _, _ = inputs[idx % len(inputs)]
            for arm in ARMS:
                evaluate_scores(channels, qk, case["public_prefix_tokens"] if arm == "selective" else 0)
        for case, qk, ring, float_scores, tolerance in inputs:
            arms = list(ARMS)
            order.shuffle(arms)
            for arm in arms:
                prefix = case["public_prefix_tokens"] if arm == "selective" else 0
                recovered, metrics = evaluate_scores(channels, qk, prefix)
                if not np.array_equal(recovered, ring):
                    raise ValueError(f"incorrect quantized attention scores: {case}, {arm}")
                recovered_float = recovered.astype(np.float64) / (SCALE ** 2 * qk.shape[-1] ** 0.5)
                error = np.abs(recovered_float - float_scores).max().item()
                if error > tolerance + 1e-10:
                    raise ValueError("Quantized attention score exceeds rounding bound")
                rows.append({**case, "arm": arm, **metrics, "exact_ring": 1,
                             "max_abs_score_error": error})
    return rows


def summarize(rows):
    output = {}
    for placement in ("early", "late"):
        output[placement] = {}
        selected = [row for row in rows if row["placement"] == placement]
        for arm in ARMS:
            subset = [row for row in selected if row["arm"] == arm]
            output[placement][arm] = {
                "prompts": len(subset),
                "exact_ring": sum(row["exact_ring"] for row in subset),
                "causal_scores": sum(row["causal_scores"] for row in subset),
                "max_abs_score_error": max(row["max_abs_score_error"] for row in subset),
                "max_reference_attention_error": max(row["reference_attention_error"] for row in subset),
                **{f"median_{key}": statistics.median(row[key] for row in subset)
                   for key in ("beaver_multiplications", "peer_payload_bytes", "client_upload_bytes",
                               "client_output_bytes", "dealer_bytes", "preprocessing_us", "online_us")},
            }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--transport", choices=("unix", "tcp"), default="unix")
    args = parser.parse_args()
    if not (1 <= args.pairs <= 24) or args.warmups < 0:
        parser.error("provide 1–24 prompt pairs and nonnegative warmups")
    inputs, meta = collect_inputs(args.checkpoint, args.pairs)
    rows = run(inputs, args.transport, args.warmups)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "pythia_qk_cases.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        **meta, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "transport": "AF_UNIX socketpair" if args.transport == "unix" else "TCP loopback",
        "trusted_client_dealer": True, "ring_bits": 64, "processes": 2,
        "prompt_pairs_per_placement": args.pairs, "warmups_per_arm": args.warmups,
        "score_protocol": "additive sharing with dealer-assisted Beaver multiplication",
        "summary": summarize(rows),
    }
    (args.output / "pythia_qk_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
