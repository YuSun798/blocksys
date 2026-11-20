"""Rerun the matched ablation and verify measured counts and LNCS page limit."""
import argparse
import json
import re
import subprocess
from pathlib import Path

from rewrite_island_cpu import run

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pdf", type=Path)
parser.add_argument("--causal-summary", type=Path, default=root / "results/summary.json")
args = parser.parse_args()

ablation = run()
if ablation != json.loads((root / "results/rewrite_island.json").read_text()):
    raise SystemExit("Matched-ablation results differ from archived results.")
for width in (1, 4, 16, 64):
    for rewrite, islands, expected in (
        (0, 0, width), (0, 1, width), (1, 0, width + 1), (1, 1, 1),
    ):
        row = ablation["results"][f"width={width}/rewrite={rewrite}/islands={islands}"]
        if row != {"cases": 100, "mismatches": 0, "multiplications": expected * 100,
                   "peer_online_bytes": expected * 3200, "batched_rounds": 100}:
            raise SystemExit(f"Unexpected ablation result at {width, rewrite, islands}: {row}")

causal = json.loads(args.causal_summary.read_text())
expected_causal = {
    "total_cases": 720,
    "total_position_comparisons": 11520,
    "causal_closure_false_negatives": 0,
    "causal_closure_false_positives": 0,
    "changed_activations_misclassified_by_naive_token_labels": 2133,
    "required_local_fallback_tasks": 400,
    "wrong_version_cache_rejections": 800,
}
for field, expected in expected_causal.items():
    if causal[field] != expected:
        raise SystemExit(f"Unexpected causal-test result for {field}: {causal[field]}")

metadata = subprocess.check_output(["pdfinfo", args.pdf], text=True)
pages = int(re.search(r"^Pages:\s+(\d+)", metadata, re.MULTILINE)[1])
if pages > 14:
    raise SystemExit(f"LNCS manuscript exceeds the 14-page limit: {pages} pages.")
print(f"PASS: {pages} pages, 1,600 matched evaluations, and 720 causal cases.")
