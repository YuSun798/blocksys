"""Replay measured composed-request payloads through an explicit link cost model."""
import argparse
import csv
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("full", "qk_only", "rows_only", "joint", "private_pool")
RTT_MS = (1, 10, 40)
MBPS = (10, 100, 1000)


def estimate_ms(row, rtt_ms, mbps, shared):
    # Dealer send/ack, QK upload, two ordered peer openings, QK return,
    # head upload and head return are eight one-way dependencies.
    peer = int(row["qk_peer_bytes"])
    client = sum(int(row[key]) for key in ("qk_dealer_bytes", "qk_client_bytes", "head_bytes"))
    # Client phases have symmetric payloads to two workers. Independent links
    # can serve these in parallel; the two peer openings are ordered in code.
    serial_bytes = peer + (client if shared else client / 2)
    return int(row["elapsed_ns"]) / 1e6 + 4 * rtt_ms + 0.008 * serial_bytes / mbps


def replay(source):
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        group = indexed.setdefault((row["pair"], row["variant"]), {})
        if row["arm"] in group:
            raise ValueError("duplicate trial arm")
        group[row["arm"]] = row
    if len(indexed) != 96 or any(set(group) != set(ARMS) for group in indexed.values()):
        raise ValueError("expected 96 prompts with all five policies")
    scenarios = []
    for shared in (True, False):
        for rtt in RTT_MS:
            for mbps in MBPS:
                trials = [
                    {arm: estimate_ms(group[arm], rtt, mbps, shared) for arm in ARMS}
                    for group in indexed.values()
                ]
                scenarios.append({
                    "link": "shared" if shared else "independent",
                    "rtt_ms": rtt, "mbps": mbps,
                    "median_ms": {arm: round(statistics.median(t[arm] for t in trials), 3)
                                  for arm in ARMS},
                    "median_paired_full_over_joint": round(
                        statistics.median(t["full"] / t["joint"] for t in trials), 3),
                })
    return {"prompts": len(indexed), "scenarios": scenarios}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS / "pythia_composed_cases.csv")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = json.dumps(replay(args.source), indent=2) + "\n"
    target = RESULTS / "network_sensitivity.json"
    if args.check:
        if target.read_text() != summary:
            raise SystemExit("Archived network replay is stale")
    else:
        target.write_text(summary)
