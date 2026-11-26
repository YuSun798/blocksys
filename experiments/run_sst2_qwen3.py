"""Evaluate the frozen SST-2 layout pair in a second model family, in plaintext."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
from collections import Counter
from pathlib import Path

from check_cross_layer_evidence import DIGEST
from check_sst2_layout import (
    CONTRACT, DATA_URL, DEV_ROWS, DEV_SHA, LAYOUTS, MEMBER, PER_LABEL, SEED, ZIP_SHA,
    digest_words, pinned_rows, sha, transition_stats,
)

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ARTIFACTS = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
LABEL_IDS = (8225, 6785)
PREFIXES = (17, 31)
THREADS = 2
CASES = "sst2_qwen3_cases.csv"
SUMMARY = "sst2_qwen3_summary.json"


def verify_checkpoint(checkpoint):
    for name, expected in ARTIFACTS.items():
        digest = hashlib.sha256()
        with (checkpoint / name).open("rb") as stream:
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{name} does not match {MODEL} revision {REVISION}")


def summarize_pairs(pairs):
    labels = Counter(int(original["label"]) for original, _ in pairs)
    correct = [tuple(int(row["native_choice"]) == int(row["label"]) for row in pair) for pair in pairs]
    transitions = transition_stats(correct)
    paired = {key: value for key, value in transitions.items()
              if key not in ("original_correct", "rewritten_correct")}
    paired["same_native_choices"] = sum(a["native_choice"] == b["native_choice"] for a, b in pairs)
    return {
        "examples": len(pairs),
        "label_counts": {str(k): v for k, v in sorted(labels.items())},
        "majority_correct": max(labels.values()),
        **{layout: {
            "native_correct": transitions[f"{layout}_correct"],
            "mean_tokens": statistics.mean(int(pair[i]["tokens"]) for pair in pairs),
            "mean_prefix_tokens": statistics.mean(int(pair[i]["private_prefix_tokens"]) for pair in pairs),
        } for i, layout in enumerate(LAYOUTS)},
        "paired": paired,
    }


def summarize(rows):
    pairs = list(zip(rows[::2], rows[1::2]))
    rng = random.Random(SEED)
    sample = [pair for label in (0, 1) for pair in rng.sample(
        [item for item in pairs if int(item[0]["label"]) == label], PER_LABEL,
    )]
    rng.shuffle(sample)
    return {"full_dev": summarize_pairs(pairs), "frozen_balanced_128": summarize_pairs(sample)}


def evaluate(checkpoint, dataset):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    from run_sst2_layout import FIELD, HEADER, LABELS, NOTE, QUERY, load_data, prepare

    contract = {"header": HEADER, "field": FIELD, "public_note": NOTE, "query": QUERY, "labels": list(LABELS)}
    if contract != CONTRACT:
        raise ValueError("the shared SST-2 prompt contract changed")
    verify_checkpoint(checkpoint)
    examples = load_data(dataset, full_dev=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ids, prepared = prepare(tokenizer, examples)
    if tuple(ids) != LABEL_IDS:
        raise ValueError("the two Qwen candidate tokens changed")

    inputs = []
    approved = [set(), set()]
    for example, pair in prepared:
        for i, (layout, prompt) in enumerate(zip(LAYOUTS, pair)):
            start = prompt.private_prefix
            span = prompt.ids[start:start + prompt.private_token_count]
            decoded_span = tokenizer.decode(span, clean_up_tokenization_spaces=False)
            sentence_span = " " + example.sentence
            if start != PREFIXES[i] or decoded_span not in (sentence_span, sentence_span + "\n"):
                raise ValueError(f"unexpected private boundary at source row {example.index}")
            approved[i].add(prompt.ids[:start])
            before = HEADER + (NOTE if layout == "rewritten" else "") + FIELD
            text = before + sentence_span + ("\n" + NOTE.rstrip("\n") if layout == "original" else "") + QUERY
            for label, token in zip(LABELS, LABEL_IDS):
                continuation = tokenizer.encode(text + " " + label, add_special_tokens=False)
                if tuple(continuation) != (*prompt.ids, token):
                    raise ValueError(f"label crosses the query boundary at source row {example.index}")
            inputs.append((prompt.ids, {
                "sample_id": example.index, "label": example.label, "layout": layout,
                "sentence_sha256": sha(example.sentence.encode("utf-8")),
                "prompt_sha256": digest_words(prompt.ids),
                "token_multiset_sha256": digest_words(sorted(prompt.ids)),
                "private_span_sha256": digest_words(span),
                "candidate_ids_sha256": digest_words(LABEL_IDS),
                "negative_token": LABEL_IDS[0], "positive_token": LABEL_IDS[1],
                "tokens": len(prompt.ids), "private_prefix_tokens": start,
                "private_token_count": prompt.private_token_count,
                "private_char_start": prompt.private_char_start,
                "private_last_token_includes_newline": int(decoded_span.endswith("\n")),
            }))
    if any(len(prefix) != 1 for prefix in approved):
        raise ValueError("approved tokens vary with the private review")

    torch.set_num_threads(THREADS)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    config = model.config
    if (config.model_type, config.vocab_size, config.hidden_size, config.head_dim,
        config.num_attention_heads, config.num_key_value_heads, config.num_hidden_layers) != (
        "qwen3", 151936, 1024, 128, 16, 8, 28,
    ):
        raise ValueError("unexpected Qwen3 checkpoint architecture")
    body = model.base_model
    projection = model.get_output_embeddings().weight.detach().numpy()[ids]
    if projection.shape != (2, 1024) or projection.dtype != np.float32:
        raise ValueError("unexpected Qwen3 candidate projection")
    print(f"Qwen3 prepared {len(inputs)} layout forwards, two CPU threads", flush=True)
    rows = []
    with torch.inference_mode():
        for index, (tokens, record) in enumerate(inputs, 1):
            states = body(torch.tensor([tokens]), use_cache=False).last_hidden_state
            logits = projection @ states[0, -1].numpy()
            negative, positive = (float(score) for score in logits)
            rows.append({
                **record,
                "native_choice": int(logits.argmax()), "negative_logit": negative, "positive_logit": positive,
                "native_margin": positive - negative,
            })
            if index % 256 == 0:
                print(f"Completed {index}/{len(inputs)}", flush=True)
    execution = {
        "mode": "plaintext FP32 prefill", "device": "CPU", "threads": THREADS, "batch_size": 1,
        "attention": "eager", "projection": "two fixed output rows on the last hidden state",
        "candidate_ids": list(LABEL_IDS), "approved_prefix_tokens": dict(zip(LAYOUTS, PREFIXES)),
        "approved_prefix_sha256": {layout: digest_words(next(iter(prefix)))
                                   for layout, prefix in zip(LAYOUTS, approved)},
        "boundary_newlines_per_layout": sum(row["private_last_token_includes_newline"] for row in rows[::2]),
        "torch": str(torch.__version__), "transformers": transformers_version, "numpy": np.__version__,
        "platform": platform.platform(),
    }
    return rows, execution


def model_manifest():
    return {"name": MODEL, "revision": REVISION, "artifact_sha256": ARTIFACTS}


def dataset_manifest():
    return {
        "name": "GLUE SST-2 official development", "source_url": DATA_URL,
        "archive_sha256": ZIP_SHA, "member": MEMBER, "dev_sha256": DEV_SHA,
        "selection": "all 872 official dev rows in source order", "contract": CONTRACT,
        "frozen_balanced_subset": {"seed": SEED, "examples_per_label": PER_LABEL},
    }


def write_archive(output, rows, execution):
    output.mkdir(parents=True, exist_ok=True)
    cases = output / CASES
    with cases.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "model": model_manifest(), "dataset": dataset_manifest(), "execution": execution,
        "cases_sha256": sha(cases.read_bytes()), "results": summarize(rows),
    }
    (output / SUMMARY).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def verify_archive(output, dataset=None):
    output = Path(output)
    report = json.loads((output / SUMMARY).read_text())
    cases = output / CASES
    if report["model"] != model_manifest() or report["dataset"] != dataset_manifest():
        raise ValueError("checkpoint or dataset identity changed")
    if sha(cases.read_bytes()) != report["cases_sha256"]:
        raise ValueError("Qwen3 case archive digest mismatch")
    with cases.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 * DEV_ROWS:
        raise ValueError("expected both layouts for every official dev row")
    official = pinned_rows(dataset, full_dev=True) if dataset else None
    for index, (a, b) in enumerate(zip(rows[::2], rows[1::2])):
        for row, layout, prefix in zip((a, b), LAYOUTS, PREFIXES):
            if (int(row["sample_id"]), row["layout"], int(row["private_prefix_tokens"])) != (index, layout, prefix):
                raise ValueError(f"invalid source index, layout or prefix at row {index}")
            if (int(row["negative_token"]), int(row["positive_token"])) != LABEL_IDS or (
                row["candidate_ids_sha256"] != digest_words(LABEL_IDS)
            ):
                raise ValueError(f"candidate IDs changed at source row {index}")
            negative, positive, margin = (float(row[col]) for col in ("negative_logit", "positive_logit", "native_margin"))
            if (not all(map(math.isfinite, (negative, positive, margin))) or margin != positive - negative
                or int(row["native_choice"]) != int(margin > 0)):
                raise ValueError(f"inconsistent logits, margin or choice at source row {index}")
            if not 0 < int(row["private_token_count"]) <= int(row["tokens"]) - prefix:
                raise ValueError(f"invalid private token count at source row {index}")
            start = len(CONTRACT["header"] + CONTRACT["field"])
            if layout == "rewritten":
                start += len(CONTRACT["public_note"])
            if int(row["private_char_start"]) != start or row["private_last_token_includes_newline"] not in ("0", "1"):
                raise ValueError(f"invalid private character boundary at source row {index}")
            digests = ("sentence_sha256", "prompt_sha256", "token_multiset_sha256", "private_span_sha256")
            if not all(DIGEST(row[field]) for field in digests):
                raise ValueError(f"invalid record fingerprint at source row {index}")
            if official:
                sentence, label = official[index]
                if label != int(row["label"]) or sha(sentence.encode("utf-8")) != row["sentence_sha256"]:
                    raise ValueError(f"archive doesn't match official source row {index}")
        for field in ("label", "sentence_sha256", "tokens", "token_multiset_sha256", "private_span_sha256",
                      "private_token_count", "private_last_token_includes_newline"):
            if a[field] != b[field]:
                raise ValueError(f"unmatched {field} at source row {index}")
        if a["prompt_sha256"] == b["prompt_sha256"]:
            raise ValueError(f"layout didn't change prompt order at source row {index}")
    execution = report["execution"]
    expected_execution = {
        "mode": "plaintext FP32 prefill", "device": "CPU", "threads": THREADS, "batch_size": 1,
        "attention": "eager", "candidate_ids": list(LABEL_IDS), "approved_prefix_tokens": dict(zip(LAYOUTS, PREFIXES)),
        "boundary_newlines_per_layout": sum(int(row["private_last_token_includes_newline"]) for row in rows[::2]),
    }
    if any(execution[key] != expected for key, expected in expected_execution.items()):
        raise ValueError("execution summary differs from archived rows")
    prefix_hashes = execution["approved_prefix_sha256"]
    if set(prefix_hashes) != set(LAYOUTS) or not all(DIGEST(value) for value in prefix_hashes.values()):
        raise ValueError("approved prefix fingerprints are invalid")
    if report["results"] != summarize(rows) or report["results"]["full_dev"]["label_counts"] != {"0": 428, "1": 444}:
        raise ValueError("Qwen3 paired statistics or label distribution mismatch")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-zip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="check the result archive without loading the model")
    args = parser.parse_args()
    if args.verify:
        report = verify_archive(args.output, args.dataset_zip)
    else:
        if not args.checkpoint or not args.dataset_zip:
            parser.error("evaluation requires --checkpoint and --dataset-zip")
        report = write_archive(args.output, *evaluate(args.checkpoint, args.dataset_zip))
    print(json.dumps(report["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
