"""Paired zero-shot layout sensitivity on the official GLUE SST-2 dev split."""
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from cross_layer_rewrite import digest_words
from cross_layer_scale import (
    CHECKPOINTS, HEAD_WIDTH, VOCAB_SIZE, checkpoint_artifacts, evaluate_arm, validate_checkpoint,
)
from pythia_qk_2pc import HEAD, LAYER, workers
from pythia_secure_head import projection_workers, quantize

DATASET_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
ZIP_SHA256 = "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
DEV_SHA256 = "54a9c12c77e402fc14d6a66a506f24d999f596b0176eb777c972cccc2f9cc4b0"
MEMBER = "SST-2/dev.tsv"
SEED = 20271122
PER_CLASS = 64
DEV_ROWS = 872
LABELS = ("negative", "positive")
LAYOUTS = ("original", "rewritten")
HEADER = "Task: Classify the sentiment of a movie review as negative or positive.\n"
NOTE = "Note: The label should describe the overall opinion expressed by the review.\n"
FIELD = "Review:"
QUERY = "\nSentiment:"
PAYLOAD_KEYS = ("qk_products", "qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes", "rows_sent")


@dataclass(frozen=True)
class Example:
    index: int
    sentence: str
    label: int


@dataclass(frozen=True)
class Prompt:
    ids: tuple
    private_prefix: int
    private_token_count: int
    private_char_start: int


def load_data(path, full_dev=False):
    if hashlib.sha256(path.read_bytes()).hexdigest() != ZIP_SHA256:
        raise ValueError("input is not the pinned GLUE SST-2 official archive")
    with zipfile.ZipFile(path) as archive:
        content = archive.read(MEMBER)
    if hashlib.sha256(content).hexdigest() != DEV_SHA256:
        raise ValueError("official dev.tsv hash mismatch")
    header, *lines = content.decode("utf-8").splitlines()
    if header != "sentence\tlabel" or len(lines) != DEV_ROWS:
        raise ValueError("unexpected SST-2 dev row count or schema")
    examples = []
    for index, line in enumerate(lines):
        sentence, label = line.rsplit("\t", 1)
        sentence = sentence.strip()
        label = int(label)
        if not sentence or "\t" in sentence or label not in (0, 1):
            raise ValueError(f"malformed SST-2 dev row {index}")
        examples.append(Example(index, sentence, label))
    if full_dev:
        return examples

    rng = random.Random(SEED)
    sampled = [example for label in (0, 1) for example in
               rng.sample([row for row in examples if row.label == label], PER_CLASS)]
    rng.shuffle(sampled)
    return sampled


def encode_prompt(tokenizer, sentence, layout):
    before = HEADER + (NOTE if layout == "rewritten" else "") + FIELD
    text = before + " " + sentence + ("\n" + NOTE.rstrip("\n") if layout == "original" else "") + QUERY
    start = len(before)
    finish = start + len(sentence) + 1
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    private = [i for i, (left, right) in enumerate(encoded.offset_mapping)
               if right > start and left < finish]
    if not private or private != list(range(private[0], private[-1] + 1)):
        raise ValueError("private sentence must occupy a nonempty consecutive token span")
    return Prompt(tuple(encoded.input_ids), private[0], len(private), start)


def prepare(tokenizer, examples):
    if not tokenizer.is_fast:
        raise ValueError("offset-mapped disclosure requires a fast tokenizer")
    variants = [tokenizer.encode(" " + label, add_special_tokens=False) for label in LABELS]
    if any(len(ids) != 1 for ids in variants) or variants[0] == variants[1]:
        raise ValueError("candidate labels are not distinct single tokens")
    ids = np.asarray([item[0] for item in variants], dtype=np.int64)
    prepared = []
    for example in examples:
        before, after = prompts = tuple(encode_prompt(tokenizer, example.sentence, layout) for layout in LAYOUTS)
        if sorted(before.ids) != sorted(after.ids) or before.ids == after.ids:
            raise ValueError(f"matched token inventory/placement did not hold at dev row {example.index}")
        if (before.private_token_count != after.private_token_count
                or before.ids[before.private_prefix:before.private_prefix + before.private_token_count]
                != after.ids[after.private_prefix:after.private_prefix + after.private_token_count]
                or before.private_prefix >= after.private_prefix):
            raise ValueError(f"rewrite changed the private token span at dev row {example.index}")
        prepared.append((example, prompts))
    return ids, prepared


def projected_counts(length, prefix, hidden_width):
    suffix = length - prefix
    protected_products = HEAD_WIDTH * suffix * (suffix + 1) // 2
    rows = len(LABELS)
    return {
        "qk_products": protected_products,
        "qk_peer_bytes": 32 * protected_products,
        "qk_dealer_bytes": 48 * protected_products + 26,
        "qk_client_bytes": 32 * HEAD_WIDTH * length + 8 * length * (length + 1),
        "head_bytes": 2 * (4 + 4 * rows + 8 * hidden_width + 8 * rows),
        "rows_sent": rows,
    }


@torch.inference_mode()
def run(checkpoint, key, examples, protected_count, transport):
    spec = CHECKPOINTS[key]
    validate_checkpoint(checkpoint, spec)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    ids, prepared = prepare(tokenizer, examples)
    body = model.base_model
    selected = body.layers[LAYER].attention
    if selected.head_size != HEAD_WIDTH:
        raise ValueError("selected attention head width differs from the traffic model")
    matrix = model.get_output_embeddings().weight.numpy()
    if matrix.shape != (VOCAB_SIZE, spec.hidden_width):
        raise ValueError("selected output matrix differs from checkpoint metadata")
    projection = matrix[ids]
    native_scores = {}
    cases = {}
    for example, prompts in prepared:
        for layout, prompt in zip(LAYOUTS, prompts):
            tokens = torch.tensor([prompt.ids])
            hidden = body(tokens, use_cache=False).last_hidden_state[0, -1].numpy()
            logits = projection @ hidden
            native_scores[example.index, layout] = logits
            cases[example.index, layout] = {
                "model": key, "sample_id": example.index, "label": example.label, "layout": layout,
                "sentence_sha256": hashlib.sha256(example.sentence.encode("utf-8")).hexdigest(),
                "prompt_sha256": digest_words(prompt.ids),
                "token_multiset_sha256": digest_words(sorted(prompt.ids)),
                "candidate_ids_sha256": digest_words(ids),
                "negative_token": int(ids[0]), "positive_token": int(ids[1]),
                "tokens": len(prompt.ids), "private_prefix_tokens": prompt.private_prefix,
                "private_token_count": prompt.private_token_count,
                "private_char_start": prompt.private_char_start,
                "native_choice": int(logits.argmax()), "native_margin": float(logits[1] - logits[0]),
                **{f"projected_{k}": v for k, v in projected_counts(
                    len(prompt.ids), prompt.private_prefix, spec.hidden_width,
                ).items()},
                "protected_choice": "", "protected_margin": "",
                "max_abs_logit_error": "", "protected_exact_ring": "", "protected_elapsed_ns": "",
                **{f"measured_{key}": "" for key in PAYLOAD_KEYS},
            }

    rng = random.Random(SEED + 1)
    if protected_count:
        fixed_weight = quantize(matrix)
        fixed_projection = fixed_weight[ids]
        max_weight = int(np.abs(fixed_weight).max())
        with workers(transport) as qk_channels, projection_workers(fixed_weight, transport=transport) as head_channels:
            # One discarded request per layout; randomize both layouts of every recorded pair.
            exposed = [prepared[0], *prepared[:protected_count]]
            for i, (example, prompts) in enumerate(exposed):
                for position in rng.sample((0, 1), 2):
                    layout, prompt = LAYOUTS[position], prompts[position]
                    inputs = (layout, prompt.private_prefix, torch.tensor([prompt.ids]), "", None,
                              projection, fixed_projection, ids)
                    choice, logits, _, elapsed, counts = evaluate_arm(
                        "operator_only" if layout == "original" else "joint",
                        inputs, body, selected, max_weight, VOCAB_SIZE, qk_channels, head_channels,
                    )
                    if counts != projected_counts(len(prompt.ids), prompt.private_prefix, spec.hidden_width):
                        raise ValueError("measured protocol payload differs from its prediction")
                    if i:
                        cases[example.index, layout].update(
                            protected_choice=int(np.where(ids == choice)[0][0]),
                            protected_margin=float(logits[1] - logits[0]),
                            max_abs_logit_error=float(np.abs(logits - native_scores[example.index, layout]).max()),
                            protected_exact_ring=1,
                            protected_elapsed_ns=elapsed,
                            **{f"measured_{key}": counts[key] for key in PAYLOAD_KEYS},
                        )
    return list(cases.values())


def summarize(rows):
    arms = {layout: [r for r in rows if r["layout"] == layout] for layout in LAYOUTS}
    pairs = list(zip(*(arms[layout] for layout in LAYOUTS)))
    result = {}
    for layout, group in arms.items():
        measured = [r for r in group if r["protected_exact_ring"] == 1]
        result[layout] = {
            "prompts": len(group),
            "native_correct": sum(r["native_choice"] == r["label"] for r in group),
            "mean_tokens": statistics.mean(r["tokens"] for r in group),
            "mean_prefix_tokens": statistics.mean(r["private_prefix_tokens"] for r in group),
            **{f"mean_projected_{key}": statistics.mean(r[f"projected_{key}"] for r in group)
               for key in PAYLOAD_KEYS},
            "protected_prompts": len(measured),
            "protected_correct": sum(r["protected_choice"] == r["label"] for r in measured),
            "protected_native_agreement": sum(r["protected_choice"] == r["native_choice"] for r in measured),
        }
        if measured:
            result[layout].update(
                protected_max_abs_logit_error=max(r["max_abs_logit_error"] for r in measured),
                median_protected_elapsed_ms=statistics.median(r["protected_elapsed_ns"] for r in measured) / 1e6,
            )
    result["paired"] = {
        "same_native_choices": sum(a["native_choice"] == b["native_choice"] for a, b in pairs),
        "correct_to_incorrect": sum(a["native_choice"] == a["label"] != b["native_choice"] for a, b in pairs),
        "incorrect_to_correct": sum(a["native_choice"] != a["label"] == b["native_choice"] for a, b in pairs),
        "same_protected_choices": sum(
            a["protected_choice"] == b["protected_choice"] for a, b in pairs if a["protected_exact_ring"] == 1
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--checkpoint-70m", type=Path)
    parser.add_argument("--checkpoint-160m", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-dev", action="store_true",
                        help="use all 872 official examples in source-row order")
    parser.add_argument("--protected-count", type=int, default=0)
    parser.add_argument("--transport", choices=("tcp", "unix"), default="tcp")
    args = parser.parse_args()
    checkpoints = {key: path for key, path in (
        ("70m", args.checkpoint_70m), ("160m", args.checkpoint_160m),
    ) if path is not None}
    size = DEV_ROWS if args.full_dev else 2 * PER_CLASS
    if not checkpoints or not 0 <= args.protected_count <= size:
        parser.error(f"provide a checkpoint and 0–{size} protected examples per model")
    examples = load_data(args.dataset_zip, args.full_dev)
    rows = []
    summary = {}
    for key, checkpoint in checkpoints.items():
        results = run(checkpoint, key, examples, args.protected_count, args.transport)
        rows.extend(results)
        summary[key] = summarize(results)
        print(f"{key}: {len(results)} native-layout evaluations, "
              f"{2 * args.protected_count} protected-layout evaluations", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "sst2_layout_cases.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "dataset": {
            "name": "GLUE SST-2 official development", "source_url": DATASET_URL,
            "dataset_zip_sha256": ZIP_SHA256, "member": MEMBER,
            "dev_tsv_sha256": DEV_SHA256, "split_rows": DEV_ROWS,
            "sample_seed": None if args.full_dev else SEED,
            "examples_per_label": None if args.full_dev else PER_CLASS,
            "sample_ids": [ex.index for ex in examples],
            "normalization": "UTF-8 official row sentence stripped with str.strip()",
        },
        "layout_contract": {"header": HEADER, "field": FIELD, "public_note": NOTE, "query": QUERY,
                            "labels": list(LABELS)},
        "models": {
            key: {"model": spec.model, "revision": spec.revision, "hidden_width": spec.hidden_width,
                  "artifact_sha256": checkpoint_artifacts(spec)}
            for key in checkpoints for spec in (CHECKPOINTS[key],)
        },
        "head_index": HEAD, "layer_index": LAYER, "head_width": HEAD_WIDTH,
        "threads": 2, "torch": str(torch.__version__), "transformers": transformers_version,
        "numpy": np.__version__, "platform": platform.platform(),
        "protected_examples_per_model": args.protected_count,
        "protected_sample_ids": [ex.index for ex in examples[:args.protected_count]],
        "protected_transport": (("TCP loopback" if args.transport == "tcp" else "AF_UNIX socketpair")
                                if args.protected_count else None),
        "discarded_protected_warmups_per_model_per_layout": int(args.protected_count > 0),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "summary": summary,
    }
    if args.full_dev:
        report["dataset"]["selection"] = "full-dev"
    (args.output / "sst2_layout_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
