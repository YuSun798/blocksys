"""Rerun the matched ablation and verify measured counts and LNCS page limit."""
import argparse
import csv
import hashlib
import json
import re
import statistics
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

head = json.loads((root / "results/pythia_head_summary.json").read_text())
head_cases = root / "results/pythia_head_cases.csv"
if hashlib.sha256(head_cases.read_bytes()).hexdigest() != head["raw_sha256"]:
    raise SystemExit("Head trial archive checksum does not match.")
with head_cases.open(newline="") as source:
    head_rows = list(csv.DictReader(source))
head_arms = ("full", "reuse", "rows", "joint", "private_pool")
head_keys = {(str(case), str(prompt), arm) for case in range(48)
             for prompt in range(2) for arm in head_arms}
if len(head_rows) != 480 or {(row["case"], row["prompt"], row["arm"]) for row in head_rows} != head_keys:
    raise SystemExit("Head archive must contain all 480 unique cases.")
if (head["vocab"], head["width"], head["fixed_scale"]) != (50304, 512, 65536):
    raise SystemExit("Unexpected pretrained projection dimensions or fixed-point scale.")
for arm in head_arms:
    public_rows = 4 if arm in ("rows", "joint") else 0
    scores_returned = public_rows or head["vocab"]
    upload = 2 * (4 + 4 * public_rows + 8 * head["width"])
    download = 2 * 8 * scores_returned
    group = [row for row in head_rows if row["arm"] == arm]
    if any(tuple(int(row[key]) for key in (
            "rows_sent", "projection_rows", "upload_bytes", "download_bytes", "peer_bytes", "exact_ring"
            )) != (public_rows, scores_returned, upload, download, 0, 1)
           or row["output_top1"] != row["reference_top1"] for row in group):
        raise SystemExit(f"Head correctness, disclosure or traffic mismatch: {arm}")
    expected = {
        "exact_ring": 96, "fp32_argmax": 96, "target_exact": 88, "fp32_target_exact": 88,
        "prompts": len(group), "projection_rows": scores_returned, "rows_sent": public_rows,
        "upload_bytes": upload, "download_bytes": download, "peer_bytes": 0,
        "max_abs_logit_error": max(float(row["max_abs_logit_error"]) for row in group),
        "median_projection_us": round(statistics.median(int(row["projection_ns"])
                                                         for row in group) / 1000, 2),
        "median_prefill_pair_ms": round(statistics.median(int(row["prefill_pair_ns"])
                                    for row in group if row["prompt"] == "0") / 1e6, 3),
    }
    if (sum(row["output_top1"] == row["target"] for row in group) != 88
            or expected["max_abs_logit_error"] >= 0.00621
            or any(head["summary"][arm][key] != value for key, value in expected.items())):
        raise SystemExit(f"Head accuracy, drift or median summary mismatch: {arm}")

qk = json.loads((root / "results/pythia_qk_summary.json").read_text())
qk_cases = root / "results/pythia_qk_cases.csv"
if hashlib.sha256(qk_cases.read_bytes()).hexdigest() != qk["raw_sha256"]:
    raise SystemExit("Attention-score archive checksum does not match.")
if (qk["transport"], qk["prompt_pairs_per_placement"], qk["layer_index"],
        qk["head_index"], qk["scale"]) != ("TCP loopback", 24, 1, 0, 65536):
    raise SystemExit("Unexpected attention-score design.")
with qk_cases.open(newline="") as source:
    qk_rows = list(csv.DictReader(source))
qk_keys = {(str(pair), placement, str(prompt), arm) for pair in range(24)
           for placement in ("early", "late") for prompt in range(2)
           for arm in ("all_shared", "selective")}
if (len(qk_rows) != 192 or
        {(row["pair_id"], row["placement"], row["variant"], row["arm"])
         for row in qk_rows} != qk_keys or
        sum(int(row["causal_scores"]) for row in qk_rows) != 40976):
    raise SystemExit("Attention-score archive is incomplete.")
for row in qk_rows:
    case = probe[2 * int(row["pair_id"]) + (row["placement"] == "late")]
    length, prefix = int(row["tokens"]), int(row["public_prefix_tokens"])
    residual = length if row["arm"] == "all_shared" else length - prefix
    products = 64 * residual * (residual + 1) // 2
    scores = length * (length + 1) // 2
    if ((length, prefix, row["record"], row["placement"]) != (
            case["sequence_tokens"], case["public_prefix_tokens"], case["record"], case["placement"])
            or tuple(int(row[key]) for key in (
                "exact_ring", "causal_scores", "beaver_multiplications", "peer_payload_bytes",
                "client_upload_bytes", "client_output_bytes", "dealer_bytes"
            )) != (1, scores, products, 32 * products, 32 * 64 * length,
                   16 * scores, 48 * products + 26)
            or float(row["max_abs_score_error"]) > 9.73e-5
            or float(row["reference_attention_error"]) > 1e-6):
        raise SystemExit("Attention-score correctness or traffic mismatch.")
for placement in ("early", "late"):
    for arm in ("all_shared", "selective"):
        group = [row for row in qk_rows if (row["placement"], row["arm"]) == (placement, arm)]
        for key in ("beaver_multiplications", "peer_payload_bytes", "dealer_bytes", "online_us"):
            if statistics.median(float(row[key]) for row in group) != qk["summary"][placement][arm]["median_" + key]:
                raise SystemExit(f"Attention-score median mismatch: {placement}/{arm}/{key}")

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
print(f"PASS: {pages} pages, {words}-word abstract, {len(loopback):,} measured TCP executions, 1,600 emulator evaluations, 720 causal cases, 48 pretrained prompt pairs, 480 protected projections, and 192 attention-score tiles.")
