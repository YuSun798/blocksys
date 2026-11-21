"""Matched square-circuit benchmark with two TCP loopback party processes."""
import argparse
import csv
import json
import multiprocessing as mp
import platform
import random
import secrets
import socket
import statistics
import struct
import time
from collections import defaultdict
from pathlib import Path

MASK = (1 << 64) - 1
WORD = struct.Struct("!Q")
HEADER = struct.Struct("!I")
ARMS = ("full", "islands", "rewrite", "joint", "algebra")
WIDTHS = (4, 16, 64, 128)
TIMEOUT = 10


def encode(values):
    return b"".join(WORD.pack(value & MASK) for value in values)


def receive(channel, size):
    data = bytearray(size)
    view = memoryview(data)
    while view:
        count = channel.recv_into(view)
        if count == 0:
            raise EOFError("peer closed before complete message")
        view = view[count:]
    return data


def words(channel, count):
    return [item[0] for item in WORD.iter_unpack(receive(channel, count * WORD.size))]


def tcp_pair():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.create_connection(listener.getsockname(), timeout=TIMEOUT)
        server, _ = listener.accept()
    for endpoint in (client, server):
        endpoint.settimeout(TIMEOUT)
        endpoint.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return client, server


def party(index, control, other):
    with control, other:
        control.settimeout(TIMEOUT)
        other.settimeout(TIMEOUT)
        while True:
            count, = HEADER.unpack(receive(control, HEADER.size))
            if count == 0:
                return
            preprocessing = words(control, 2 * count)
            masks = preprocessing[::2]
            squares = preprocessing[1::2]
            control.sendall(b"R")
            online = words(control, 2 * count)
            shares = online[::2]
            weights = online[1::2]

            # At these widths each opening fits in a single small send.
            diffs = [(x - a) & MASK for x, a in zip(shares, masks)]
            packet = encode(diffs)
            if index == 0:
                other.sendall(packet)
                received = words(other, count)
            else:
                received = words(other, count)
                other.sendall(packet)
            result = 0
            for own, remote, a, aa, weight in zip(diffs, received, masks, squares, weights):
                d = (own + remote) & MASK
                result += weight * (aa + 2 * d * a + (d * d if index == 0 else 0))
            control.sendall(encode((result, len(packet))))


def shares(value):
    first = secrets.randbits(64)
    return first, (value - first) & MASK


def split(values):
    left, right = zip(*(shares(value) for value in values))
    return left, right


def plan(terms, private, public_template, arm):
    """Produce protected square operands, public weights, and client offset."""
    m = len(terms)
    if public_template and arm == "algebra":
        count, total, second = 0, 0, 0
        for p in terms:
            count += 1
            total += p
            second += p * p
        return [private], [count], second + 2 * total * private

    # The typed rewrite applies only when the template operands are public.
    # Each entry is (square operand, coefficient, depends on private input).
    if public_template and arm in ("rewrite", "joint"):
        gates = [(p, 1, False) for p in terms] + [(private, m, True)]
        offset = 2 * private * sum(terms)
    else:
        gates = [(p + private, 1, True) for p in terms]
        offset = 0
    if arm in ("islands", "joint"):
        offset += sum(w * p * p for p, w, private_dependent in gates
                      if not private_dependent)
        gates = [gate for gate in gates if gate[2]]
    return [gate[0] for gate in gates], [gate[1] for gate in gates], offset


def evaluate(channels, operands, weights, offset):
    prep_start = time.perf_counter_ns()
    count = len(operands)
    masks = [secrets.randbits(64) for _ in operands]
    triples = split(value for a in masks for value in (a, a * a))
    prep_packets = [HEADER.pack(count) + encode(values) for values in triples]
    for channel, packet in zip(channels, prep_packets):
        channel.sendall(packet)
    for channel in channels:
        if receive(channel, 1) != b"R":
            raise ValueError("unexpected preprocessing acknowledgment")
    prep_ns = time.perf_counter_ns() - prep_start

    online_start = time.perf_counter_ns()
    input_packets = [
        encode(value for pair in zip(values, weights) for value in pair)
        for values in split(operands)
    ]
    for channel, packet in zip(channels, input_packets):
        channel.sendall(packet)
    outputs = [words(channel, 2) for channel in channels]
    answer = (offset + sum(value for value, _ in outputs)) & MASK
    online_ns = time.perf_counter_ns() - online_start
    peer_bytes = sum(sent for _, sent in outputs)
    if peer_bytes != 2 * count * WORD.size:
        raise ValueError("unexpected opening payload length")
    # Two result words. Exclude the extra per-party instrumentation words.
    client_bytes = sum(map(len, input_packets)) + 2 * WORD.size
    return answer, peer_bytes, client_bytes, sum(map(len, prep_packets)) + 2, online_ns, prep_ns


def run(trials, warmup):
    ctx = mp.get_context("spawn")
    p0_link, p1_link = tcp_pair()
    client0, control0 = tcp_pair()
    client1, control1 = tcp_pair()
    clients = client0, client1
    workers = [
        ctx.Process(target=party, args=(0, control0, p0_link)),
        ctx.Process(target=party, args=(1, control1, p1_link)),
    ]
    rng = random.Random(2027)
    rows = []
    try:
        for worker in workers:
            worker.start()
        for endpoint in (control0, control1, p0_link, p1_link):
            endpoint.close()
        for template in ("public", "private"):
            for width in WIDTHS:
                for trial in range(-warmup, trials):
                    terms = [rng.randrange(-1000, 1001) for _ in range(width)]
                    private = rng.randrange(-1000, 1001)
                    expected = sum((value + private) ** 2 for value in terms) & MASK
                    order = list(ARMS)
                    rng.shuffle(order)
                    for arm in order:
                        request_start = time.perf_counter_ns()
                        operands, weights, offset = plan(terms, private, template == "public", arm)
                        plan_ns = time.perf_counter_ns() - request_start
                        measured = evaluate(clients, operands, weights, offset)
                        cold_ns = time.perf_counter_ns() - request_start
                        answer, peer, client, offline, online_ns, prep_ns = measured
                        if answer != expected:
                            raise ValueError(f"wrong output: {template}, width {width}, {arm}")
                        if trial >= 0:
                            rows.append({
                                "template": template, "width": width, "trial": trial, "arm": arm,
                                "secure_products": len(operands), "peer_bytes": peer,
                                "client_bytes": client, "offline_bytes": offline,
                                "online_ns": online_ns, "preprocessing_ns": prep_ns,
                                "plan_ns": plan_ns, "cold_request_ns": cold_ns,
                                "exact": int(answer == expected),
                            })
        for channel in clients:
            channel.sendall(HEADER.pack(0))
    finally:
        for endpoint in (client0, client1, control0, control1, p0_link, p1_link):
            endpoint.close()
        for worker in workers:
            if worker.pid is not None:
                worker.join(TIMEOUT)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                if worker.exitcode:
                    raise RuntimeError(f"party process failed with exit code {worker.exitcode}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["template"], row["width"], row["arm"]].append(row)
    return [
        {
            "template": template, "width": width, "arm": arm, "trials": len(group),
            **{key: group[0][key] for key in ("secure_products", "peer_bytes", "client_bytes", "offline_bytes")},
            "exact": sum(row["exact"] for row in group),
            "online_median_us": round(statistics.median(row["online_ns"] for row in group) / 1000, 2),
            "online_p95_us": round(statistics.quantiles(
                [row["online_ns"] for row in group], n=20, method="inclusive"
            )[18] / 1000, 2),
            "cold_median_us": round(statistics.median(row["cold_request_ns"] for row in group) / 1000, 2),
            "preprocessing_median_us": round(statistics.median(
                row["preprocessing_ns"] for row in group
            ) / 1000, 2),
        }
        for (template, width, arm), group in sorted(groups.items())
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2 or args.warmup < 0:
        parser.error("use at least 2 trials and nonnegative warm-up")
    results = run(args.trials, args.warmup)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "loopback_cases.csv", results)
    write_csv(args.output / "loopback_summary.csv", summarize(results))
    metadata = {"platform": platform.platform(), "python": platform.python_version(),
                "trials_per_cell": args.trials, "warmup_per_cell": args.warmup,
                "transport": "TCP loopback, separate multiprocessing workers",
                "ring_bits": 64, "party_count": 2, "trusted_dealer": True}
    (args.output / "loopback_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(results)} exact executions on {platform.platform()} with Python {platform.python_version()}")
