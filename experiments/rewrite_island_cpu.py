"""Paired microbenchmark for exact rewriting and selective 2PC arithmetic.

A deterministic in-process emulator uses dealer-provided independent triples.
It counts logical peer-to-peer openings and excludes transport and preprocessing.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

RING_BITS = 64
MOD = 1 << RING_BITS
OPENING_BYTES = 2 * 2 * RING_BITS // 8


def share(value, rng):
    first = rng.randrange(MOD)
    return first, (value - first) % MOD


def multiply(left, right, rng):
    a, b = rng.randrange(MOD), rng.randrange(MOD)
    sa, sb, sc = share(a, rng), share(b, rng), share(a * b, rng)
    d = sum((x - y) % MOD for x, y in zip(left, sa)) % MOD
    e = sum((x - y) % MOD for x, y in zip(right, sb)) % MOD
    return tuple((sc[i] + d * sb[i] + e * sa[i] + (d * e if i == 0 else 0))
                 % MOD for i in range(2))


def evaluate(public_terms, private, rewrite, islands, rng):
    # The unrevised program exposes each square after an opaque task-local sum.
    # The exact rewrite expands (p+x)^2 in the ring, leaving only x^2 secret.
    # A contract establishes equality for all ring inputs by distributivity.
    if rewrite:
        secure_terms = [private] if islands else [*public_terms, private]
        clear = (sum(p * p for p in public_terms) if islands else 0)
        clear += 2 * private * sum(public_terms)
    else:
        secure_terms = [p + private for p in public_terms]
        clear = 0

    squares = []
    for term in secure_terms:
        shared = share(term, rng)
        squares.append(sum(multiply(shared, shared, rng)) % MOD)

    if rewrite:
        # The final square is x^2 and can be reused within the same request.
        squares[-1] *= len(public_terms)
    return (clear + sum(squares)) % MOD, len(secure_terms)


def run():
    rows = defaultdict(lambda: {"cases": 0, "mismatches": 0, "multiplications": 0,
                                "peer_online_bytes": 0, "batched_rounds": 0})
    data = random.Random(2027)
    for width in (1, 4, 16, 64):
        for case in range(100):
            terms = [data.randrange(-1000, 1001) for _ in range(width)]
            private = data.randrange(-1000, 1001)
            reference = sum((p + private) ** 2 for p in terms) % MOD
            for rewrite in (False, True):
                for islands in (False, True):
                    result, products = evaluate(
                        terms, private, rewrite, islands,
                        random.Random((width << 16) + 4 * case + 2 * rewrite + islands),
                    )
                    row = rows[f"width={width}/rewrite={int(rewrite)}/islands={int(islands)}"]
                    row["cases"] += 1
                    row["mismatches"] += int(result != reference)
                    row["multiplications"] += products
                    row["peer_online_bytes"] += products * OPENING_BYTES
                    row["batched_rounds"] += int(products > 0)
    if any(row["mismatches"] for row in rows.values()):
        raise SystemExit("ring equality check failed")
    return {"ring_bits": RING_BITS, "results": dict(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2) + "\n")
