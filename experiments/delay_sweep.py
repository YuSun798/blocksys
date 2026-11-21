"""Controlled propagation-delay test on TCP loopback; no host traffic shaping."""
import argparse
import csv
import json
import statistics
from pathlib import Path

DELAYS_MS = (0, 5, 20)
ARMS = ("full", "joint")


def summarize(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    values = {}
    for row in rows:
        key = (int(row["rtt_ms"]), int(row["pair"]), int(row["variant"]), row["arm"])
        if key in values or row["reference"] != row["protected"] or int(row["elapsed_ns"]) <= 0:
            raise ValueError("invalid controlled-delay trial")
        values[key] = row
    expected = {(rtt, pair, variant, arm) for rtt in DELAYS_MS for pair in range(48)
                for variant in range(2) for arm in ARMS}
    if values.keys() != expected:
        raise ValueError("incomplete controlled-delay archive")
    summary = []
    for rtt in DELAYS_MS:
        latencies = {arm: [] for arm in ARMS}
        ratios = []
        increments = {arm: [] for arm in ARMS}
        for pair in range(48):
            for variant in range(2):
                for arm in ARMS:
                    current = values[rtt, pair, variant, arm]
                    baseline = values[0, pair, variant, arm]
                    joint = values[rtt, pair, variant, "joint"]
                    payload = ("qk_products", "qk_peer_bytes", "qk_dealer_bytes",
                               "qk_client_bytes", "head_bytes", "rows_sent")
                    if any(current[field] != baseline[field] for field in payload):
                        raise ValueError("delay changed the payload for a fixed policy")
                    fields = ("score_sha256", "target", "protected")
                    if any(current[field] != baseline[field] or current[field] != joint[field] for field in fields):
                        raise ValueError("delay or policy changed a paired result")
                    elapsed = int(current["elapsed_ns"]) / 1e6
                    latencies[arm].append(elapsed)
                    increments[arm].append(elapsed - int(baseline["elapsed_ns"]) / 1e6)
                ratios.append(latencies["full"][-1] / latencies["joint"][-1])
        summary.append({
            "rtt_ms": rtt,
            "prompts": len(ratios),
            "paired_ratio": round(statistics.median(ratios), 3),
            "median_ms": {arm: round(statistics.median(latencies[arm]), 3) for arm in ARMS},
            "paired_increment_ms": {arm: round(statistics.median(increments[arm]), 3) for arm in ARMS},
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = args.output / "delay_cases.csv"
    summary = args.output / "delay_summary.json"
    if not args.check:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for a new experiment")
        from pythia_composed import run

        args.output.mkdir(parents=True, exist_ok=True)
        with raw.open("w", newline="") as stream:
            writer = None
            for rtt in DELAYS_MS:
                for row in run(args.checkpoint, one_way_s=rtt / 2000, arms=ARMS):
                    row = {"rtt_ms": rtt, **row}
                    if writer is None:
                        writer = csv.DictWriter(stream, row.keys(), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(row)
    report = json.dumps(summarize(raw), indent=2) + "\n"
    if args.check:
        if summary.read_text() != report:
            raise SystemExit("archived delay statistics are stale")
    else:
        summary.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
