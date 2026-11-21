"""Project private Pythia hidden states with two independent TCP workers.

Workers hold a public fixed-point projection matrix. Only fresh additive
shares of each hidden state are sent to them. Row-selection IDs are sent
only when the task policy authorizes the candidate pool.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import random
import secrets
import statistics
import struct
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from loopback_2pc import TIMEOUT, receive, tcp_pair

FIXED_SCALE = 1 << 16
WORD_BYTES = 8
WORDS = np.dtype("<i8")
ROW_IDS = np.dtype(">u4")
COUNT = struct.Struct("!I")
STOP = (1 << 32) - 1
ARMS = (
    ("full", False, False, True),
    ("reuse", True, False, True),
    ("rows", False, True, True),
    ("joint", True, True, True),
    ("private_pool", True, True, False),
)


def projection_party(path, channel):
    with channel:
        channel.settimeout(TIMEOUT)
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        while True:
            count, = COUNT.unpack(receive(channel, COUNT.size))
            if count == STOP:
                return
            indices = np.frombuffer(receive(channel, count * ROW_IDS.itemsize), dtype=ROW_IDS)
            hidden = np.frombuffer(receive(channel, matrix.shape[1] * WORD_BYTES), dtype=WORDS)
            scores = (matrix[indices] if count else matrix) @ hidden
            channel.sendall(scores.astype(WORDS, copy=False).tobytes())


@contextmanager
def projection_workers(matrix):
    with tempfile.TemporaryDirectory(prefix="pythia-public-head-") as scratch:
        path = Path(scratch) / "head.npy"
        np.save(path, matrix, allow_pickle=False)
        ctx = mp.get_context("spawn")
        pairs = [tcp_pair() for _ in range(2)]
        clients = [left for left, _ in pairs]
        workers = [ctx.Process(target=projection_party, args=(path, right))
                   for _, right in pairs]
        try:
            for worker in workers:
                worker.start()
            for _, endpoint in pairs:
                endpoint.close()
            yield clients
            for channel in clients:
                channel.sendall(COUNT.pack(STOP))
        finally:
            for left, right in pairs:
                left.close()
                right.close()
            for worker in workers:
                if worker.pid is not None:
                    worker.join(TIMEOUT)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
            failures = [worker.exitcode for worker in workers if worker.exitcode not in (None, 0)]
            if failures:
                raise RuntimeError(f"projection workers failed: {failures}")


def project(channels, hidden, candidates, vocab, restrict, pool_public):
    requested = candidates if restrict and pool_public else ()
    count = len(requested)
    started = time.perf_counter_ns()
    # NumPy fixed-width integer arithmetic implements operations mod 2^64.
    mask = np.frombuffer(secrets.token_bytes(hidden.nbytes), dtype=WORDS)
    selector = COUNT.pack(count) + np.asarray(requested, dtype=ROW_IDS).tobytes()
    upload = sum(len(selector) + hidden.nbytes for _ in channels)
    for channel, share in zip(channels, (mask, hidden - mask)):
        channel.sendall(selector + share.astype(WORDS, copy=False).tobytes())
    reply_size = (count or vocab) * WORD_BYTES
    replies = [receive(channel, reply_size) for channel in channels]
    first, second = [np.frombuffer(reply, dtype=WORDS) for reply in replies]
    result = first + second
    if not count:
        result = result[candidates]
    return result, upload, sum(map(len, replies)), count, time.perf_counter_ns() - started


def quantize(values):
    return np.rint(values * FIXED_SCALE).astype(np.int64)


def run(weight, cases):
    public_weight = quantize(weight)
    max_weight = int(np.abs(public_weight).max())
    vocab, width = public_weight.shape
    rng = random.Random(20271120)
    rows = []
    with projection_workers(public_weight) as channels:
        warm = cases[0]
        for arm, cached, restrict, public in ARMS:
            project(channels, quantize(warm["hidden_full"][0]), warm["candidate_ids"],
                    vocab, restrict, public)
        for case in cases:
            ids = case["candidate_ids"]
            if len(ids) != 4 or len(set(map(int, ids))) != 4:
                raise ValueError("expected four distinct candidate ids")
            selected_weight = public_weight[ids]
            sources = {
                False: quantize(case["hidden_full"]),
                True: quantize(case["hidden_cached"]),
            }
            max_hidden = max(int(np.abs(source).max()) for source in sources.values())
            if max_hidden * max_weight * width >= 1 << 63:
                raise ValueError("unwrapped fixed-point score may exceed signed ring range")
            for prompt in range(2):
                order = list(ARMS)
                rng.shuffle(order)
                reference = case["logits_fp32"][prompt]
                argmax_ref = int(ids[reference.argmax()])
                for arm, cached, restrict, public in order:
                    hidden = sources[cached][prompt]
                    local_started = time.perf_counter_ns()
                    local = selected_weight @ hidden
                    local_ns = time.perf_counter_ns() - local_started
                    scores, upload, download, rows_sent, elapsed = project(
                        channels, hidden, ids, vocab, restrict, public)
                    if not np.array_equal(scores, local):
                        raise ValueError(f"integer projection mismatch in {arm}, case {case['case_id']}")
                    # For a cached prefill this includes both cache and quantization error.
                    error = np.abs(scores.astype(np.float64) / FIXED_SCALE**2 - reference)
                    selected = int(ids[scores.argmax()])
                    rows.append({
                        "case": case["case_id"],
                        "prompt": prompt,
                        "record": case["record"],
                        "placement": case["placement"],
                        "arm": arm,
                        "rows_sent": rows_sent,
                        "projection_rows": rows_sent or vocab,
                        "upload_bytes": upload,
                        "download_bytes": download,
                        "peer_bytes": 0,
                        "projection_ns": elapsed,
                        "local_gather_ns": local_ns,
                        "prefill_pair_ns": case["prefill_cached_ns" if cached else "prefill_full_ns"],
                        "exact_ring": 1,
                        "reference_top1": argmax_ref,
                        "output_top1": selected,
                        "target": int(case["target_ids"][prompt]),
                        "max_abs_logit_error": float(error.max()),
                    })
    return rows


def summarize(rows):
    summary = {}
    for arm, _, _, _ in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        paired_prefill = [row["prefill_pair_ns"] for row in group if row["prompt"] == 0]
        timings = [row["projection_ns"] for row in group]
        summary[arm] = {
            "prompts": len(group),
            "exact_ring": sum(row["exact_ring"] for row in group),
            "fp32_argmax": sum(row["output_top1"] == row["reference_top1"] for row in group),
            "target_exact": sum(row["output_top1"] == row["target"] for row in group),
            "fp32_target_exact": sum(row["reference_top1"] == row["target"] for row in group),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in group),
            **{key: group[0][key] for key in ("projection_rows", "rows_sent", "upload_bytes",
                                             "download_bytes", "peer_bytes")},
            "median_projection_us": round(statistics.median(timings) / 1000, 2),
            "p95_projection_us": round(statistics.quantiles(timings, n=20, method="inclusive")[18] / 1000, 2),
            "median_local_gather_us": round(statistics.median(row["local_gather_ns"] for row in group) / 1000, 2),
            "median_prefill_pair_ms": round(statistics.median(paired_prefill) / 1e6, 3),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The projection workers need only NumPy, with no model or tokenizer.
    import torch
    import transformers
    from local_lm_quality import MODEL, REVISION, WEIGHT_SHA256
    from pythia_head_inputs import load_inputs

    torch.set_num_threads(2)
    matrix, cases = load_inputs(args.checkpoint)
    rows = run(matrix, cases)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "pythia_head_cases.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "vocab": matrix.shape[0], "width": matrix.shape[1],
        "fixed_scale": FIXED_SCALE, "summary": summarize(rows),
    }
    (args.output / "pythia_head_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()
