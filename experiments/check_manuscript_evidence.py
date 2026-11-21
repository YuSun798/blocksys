"""Rerun the matched ablation and verify measured counts and LNCS page limit."""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from loopback_2pc import ARMS, WIDTHS, summarize
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

loopback_meta = json.loads((root / "results/loopback_metadata.json").read_text())
with (root / "results/loopback_cases.csv").open(newline="") as source:
    loopback = [{name: value if name in ("template", "arm") else int(value)
                 for name, value in row.items()} for row in csv.DictReader(source)]
if loopback_meta["trials_per_cell"] != 100 or loopback_meta["warmup_per_cell"] != 10:
    raise SystemExit("Unexpected loopback trial design.")
expected_cells = {(template, width, arm, trial)
                  for template in ("public", "private") for width in WIDTHS
                  for arm in ARMS for trial in range(loopback_meta["trials_per_cell"])}
for row in loopback:
    template, width, arm, trial = (row[key] for key in ("template", "width", "arm", "trial"))
    key = template, width, arm, trial
    if key not in expected_cells:
        raise SystemExit(f"Unexpected or duplicate loopback case: {key}")
    expected_cells.remove(key)
    squares = width
    if template == "public":
        if arm == "rewrite":
            squares += 1
        elif arm in ("joint", "algebra"):
            squares = 1
    if (row["exact"], row["secure_products"], row["peer_bytes"],
        row["client_bytes"], row["offline_bytes"]) != (
            1, squares, 16 * squares, 32 * squares + 16, 32 * squares + 10):
        raise SystemExit(f"Unexpected correctness or traffic count: {key}")
    if min(row["online_ns"], row["preprocessing_ns"], row["plan_ns"]) <= 0 or (
            row["cold_request_ns"] < row["online_ns"] + row["preprocessing_ns"] + row["plan_ns"]):
        raise SystemExit(f"Unexpected timing fields: {key}")
if expected_cells:
    raise SystemExit("Archived loopback experiment is incomplete.")
with (root / "results/loopback_summary.csv").open(newline="") as source:
    archived_summary = list(csv.DictReader(source))
if [{key: str(value) for key, value in row.items()} for row in summarize(loopback)] != archived_summary:
    raise SystemExit("Loopback timing summary differs from archived cases.")

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

pythia = json.loads((root / "results/pythia_probe.json").read_text())
probe = pythia["cases"]
if len(probe) != 48 or pythia["tv_threshold"] != 1e-5:
    raise SystemExit("Unexpected Pythia probe design.")
for placement, changed_count in (("early", 290), ("late", 82)):
    group = [row for row in probe if row["placement"] == placement]
    if (len(group) != 24 or any(row["max_prefix_tv"] != 0 for row in group)
            or sum(row["suffix_positions_changed"] for row in group) != changed_count
            or sum(row["sequence_tokens"] - row["public_prefix_tokens"] - 1
                   for row in group) != changed_count):
        raise SystemExit(f"Unexpected Pythia placement result: {placement}")
if (sum(row["prefix_reuse_argmax_matches"] for row in probe) != 1928
        or sum(2 * row["sequence_tokens"] for row in probe) != 1928
        or round(max(row["prefix_reuse_max_tv"] for row in probe), 5) != 0.00110):
    raise SystemExit("Unexpected Pythia prefix reuse result.")

metadata = subprocess.check_output(["pdfinfo", args.pdf], text=True)
pages = int(re.search(r"^Pages:\s+(\d+)", metadata, re.MULTILINE)[1])
if pages > 14:
    raise SystemExit(f"LNCS manuscript exceeds the 14-page limit: {pages} pages.")
paper = subprocess.check_output(["pdftotext", "-layout", args.pdf, "-"], text=True)
if re.search(r"\b(?:TODO|TBD|lorem ipsum|contribution title|title suppressed)\b|\?\?",
             paper, re.IGNORECASE):
    raise SystemExit("The compiled PDF contains a placeholder or unresolved reference.")
for heading in ("Compile on the trusted client", "Disclosure of Interests"):
    if heading not in paper:
        raise SystemExit(f"Expected manuscript content missing: {heading}")
if "The authors declare no competing interests relevant to this work." not in " ".join(paper.split()):
    raise SystemExit("The competing-interests statement is missing.")
abstract = re.search(r"Abstract\.\s*(.*?)\s*Keywords:", paper, re.DOTALL)
if abstract is None:
    raise SystemExit("Could not locate the LNCS abstract.")
words = len(re.findall(r"[\w]+(?:[-,.][\w]+)*", re.sub(r"(\w)-\s*\n\s*", r"\1", abstract[1])))
if not 150 <= words <= 250:
    raise SystemExit(f"Abstract has {words} words. The template range is 150–250.")
print(f"PASS: {pages} pages, {words}-word abstract, {len(loopback):,} measured TCP executions, 1,600 emulator evaluations, 720 causal cases, and 48 pretrained prompt pairs.")
