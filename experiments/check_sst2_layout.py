"""Check the pinned GLUE SST-2 layout diagnostic and its paired evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from check_cross_layer_evidence import DIGEST
from check_cross_layer_scale import checkpoint_width

ZIP_SHA = "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
DEV_SHA = "54a9c12c77e402fc14d6a66a506f24d999f596b0176eb777c972cccc2f9cc4b0"
MEMBER = "SST-2/dev.tsv"
DEV_ROWS = 872
FULL_DEV = "full-dev"
SEED = 20271122
PER_LABEL = 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned_rows(path, full_dev=False):
    if sha(path.read_bytes()) != ZIP_SHA:
        raise ValueError("expected the pinned original GLUE SST-2 ZIP")
    with zipfile.ZipFile(path) as source:
        dev = source.read(MEMBER)
    if sha(dev) != DEV_SHA:
        raise ValueError("official dev member digest does not match")
    parsed = list(csv.DictReader(io.StringIO(dev.decode("utf-8")), delimiter="\t"))
    labels = [int(row["label"]) for row in parsed]
    if Counter(labels) != {0: 428, 1: 444}:
        raise ValueError("official dev split must contain 872 binary examples")
    if full_dev:
        sample = range(DEV_ROWS)
    else:
        rng = random.Random(SEED)
        sample = [i for label in (0, 1) for i in
                  rng.sample([i for i, value in enumerate(labels) if value == label], PER_LABEL)]
        rng.shuffle(sample)
    return {i: (parsed[i]["sentence"].strip(), labels[i]) for i in sample}


def digest_words(items):
    return sha(struct.pack(f"<{len(items)}q", *items))


def difference_interval(before, after):
    """Percentile interval over resampled dataset rows, paired by source index."""
    differences = [int(y) - int(x) for x, y in zip(before, after)]
    rng = random.Random(SEED + 1)
    count = len(differences)
    samples = [sum(rng.choices(differences, k=count)) / count for _ in range(10_000)]
    quantiles = statistics.quantiles(samples, n=40, method="inclusive")
    return [quantiles[0], quantiles[-1]]


def transition_stats(rows):
    """Compute paired deltas from gold-correctness booleans rather than run counts."""
    before, after = zip(*rows)
    return {
        "original_correct": sum(before),
        "rewritten_correct": sum(after),
        "correct_to_incorrect": sum(a and not b for a, b in rows),
        "incorrect_to_correct": sum(not a and b for a, b in rows),
        "paired_accuracy_difference": (sum(after) - sum(before)) / len(rows),
        "paired_bootstrap_95_interval": difference_interval(before, after),
    }


LAYOUTS = ("original", "rewritten")
MODEL_NAMES = {m: f"EleutherAI/pythia-{m}-deduped" for m in ("70m", "160m")}
DATA_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
SAMPLE_SHA = "da7729b5655ea981c0da9cfff331bbd3e7dd30eddaac58772f34a1a1f517baea"
FULL_DEV_SHA = "017dee7b05ee71f2df834d1dc21bbdad22f4d2928423122c24b27024653b62c8"
TOKENIZER_SHA = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
CONTRACT = {
    "header": "Task: Classify the sentiment of a movie review as negative or positive.\n",
    "field": "Review:",
    "public_note": "Note: The label should describe the overall opinion expressed by the review.\n",
    "query": "\nSentiment:",
    "labels": ["negative", "positive"],
}
PAYLOAD = ("qk_products", "qk_peer_bytes", "qk_dealer_bytes", "qk_client_bytes", "head_bytes", "rows_sent")
DIGEST_COLS = ("sentence_sha256", "prompt_sha256", "token_multiset_sha256", "candidate_ids_sha256")
INT_COLS = (
    "sample_id", "label", "negative_token", "positive_token", "tokens", "private_prefix_tokens",
    "private_token_count", "private_char_start", "native_choice",
)
PROTECTED = (
    "protected_choice", "protected_margin", "max_abs_logit_error", "protected_exact_ring",
    "protected_elapsed_ns", *(f"measured_{field}" for field in PAYLOAD),
)


def model_widths(metadata):
    if not metadata["models"] or not set(metadata["models"]) <= set(MODEL_NAMES):
        raise ValueError("diagnostic contains an unsupported checkpoint")
    widths = {}
    for model, manifest in metadata["models"].items():
        if manifest["model"] != MODEL_NAMES[model]:
            raise ValueError(f"checkpoint keyed incorrectly: {model}")
        widths[model] = checkpoint_width({
            **manifest, "weight_sha256": manifest["artifact_sha256"]["model.safetensors"],
        })
    return widths


def read_metadata(directory):
    metadata = json.loads((directory / "sst2_layout_summary.json").read_text())
    dataset = metadata["dataset"]
    selection = dataset.get("selection")
    if selection not in (None, FULL_DEV):
        raise ValueError("unsupported SST-2 selection")
    full_dev = selection == FULL_DEV
    expected = {
        "name": "GLUE SST-2 official development", "source_url": DATA_URL,
        "dataset_zip_sha256": ZIP_SHA, "member": MEMBER, "dev_tsv_sha256": DEV_SHA,
        "split_rows": DEV_ROWS, "sample_seed": None if full_dev else SEED,
        "examples_per_label": None if full_dev else PER_LABEL,
        "normalization": "UTF-8 official row sentence stripped with str.strip()",
    }
    if any(dataset[field] != value for field, value in expected.items()):
        raise ValueError("dataset manifest does not describe the pinned SST-2 selection")
    sample_ids = dataset["sample_ids"]
    if full_dev:
        if sample_ids != list(range(DEV_ROWS)):
            raise ValueError("full dev split must include every official row in order")
    elif len(sample_ids) != 2 * PER_LABEL or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("dataset manifest does not have 128 distinct IDs")
    if metadata["layout_contract"] != CONTRACT or (
            metadata["head_index"], metadata["layer_index"], metadata["head_width"]) != (0, 1, 64):
        raise ValueError("unexpected experimental layout or QK dimensions")
    protected = metadata["protected_examples_per_model"]
    allowed_transports = ("TCP loopback", "AF_UNIX socketpair") if protected else (None,)
    if (not 0 <= protected <= len(sample_ids)
            or metadata["protected_sample_ids"] != sample_ids[:protected]
            or metadata["protected_transport"] not in allowed_transports
            or metadata["discarded_protected_warmups_per_model_per_layout"] != int(protected > 0)):
        raise ValueError("protected scope is inconsistent with the deterministic sampling order")
    return metadata, model_widths(metadata)


def read_cases(path, metadata, widths):
    if sha(path.read_bytes()) != metadata["raw_sha256"]:
        raise ValueError("raw SST-2 archive hash does not match its manifest")
    actual = {}
    protected_ids = set(metadata["protected_sample_ids"])
    with path.open(newline="") as stream:
        for entry in csv.DictReader(stream):
            row = {**entry, **{field: int(entry[field]) for field in INT_COLS}}
            source = row["sample_id"]
            model, layout = row["model"], row["layout"]
            idx = model, source, layout
            if idx in actual or model not in widths or layout not in LAYOUTS:
                raise ValueError(f"duplicate or unsupported SST-2 row {idx}")
            if not all(DIGEST(row[field]) for field in DIGEST_COLS):
                raise ValueError(f"invalid fingerprint at {idx}")
            margin = float(row["native_margin"])
            if not math.isfinite(margin) or row["native_choice"] != int(margin > 0):
                raise ValueError(f"FP32 choice inconsistent with stored margin at {idx}")
            length, prefix, sentence_tokens = (row[name] for name in (
                "tokens", "private_prefix_tokens", "private_token_count"))
            header = CONTRACT["header"] + (CONTRACT["public_note"] if layout == "rewritten" else "")
            if (not 0 < prefix < length or not 0 < sentence_tokens <= length - prefix
                    or row["private_char_start"] != len(header + CONTRACT["field"])):
                raise ValueError(f"inconsistent private boundary at {idx}")
            candidates = row["negative_token"], row["positive_token"]
            if len(set(candidates)) != 2 or digest_words(candidates) != row["candidate_ids_sha256"]:
                raise ValueError(f"incorrect binary candidate scope at {idx}")
            suffix = length - prefix
            products = 64 * suffix * (suffix + 1) // 2
            expected = (products, 32 * products, 48 * products + 26,
                        2048 * length + 8 * length * (length + 1), 16 * widths[model] + 56, 2)
            projections = tuple(int(row[f"projected_{field}"]) for field in PAYLOAD)
            if projections != expected:
                raise ValueError(f"wrong projected QK, dealer, client or head bytes at {idx}")
            if source in protected_ids:
                choice = int(row["protected_choice"])
                protected_margin = float(row["protected_margin"])
                error = float(row["max_abs_logit_error"])
                elapsed = int(row["protected_elapsed_ns"])
                if (not math.isfinite(protected_margin) or choice != int(protected_margin > 0)
                        or not math.isfinite(error) or error < 0 or elapsed <= 0
                        or row["protected_exact_ring"] != "1"
                        or tuple(int(row[f"measured_{field}"]) for field in PAYLOAD) != expected):
                    raise ValueError(f"wrong measured answer, timing or payload at {idx}")
                row.update(protected_choice=choice, max_abs_logit_error=error,
                           protected_elapsed_ns=elapsed, protected_exact_ring=1)
            elif any(row[field] != "" for field in PROTECTED):
                raise ValueError(f"projected row includes unregistered protected results at {idx}")
            actual[idx] = row
    expected = {(m, i, layout) for m in widths for i in metadata["dataset"]["sample_ids"]
                for layout in LAYOUTS}
    if set(actual) != expected:
        raise ValueError("required model/validation example/layout cells are incomplete")
    return actual


def check_input_pairs(actual, metadata, examples=None, tokenizer_path=None):
    sample_ids = metadata["dataset"]["sample_ids"]
    models = list(metadata["models"])
    reference = models[0]
    source_hash = hashlib.sha256()
    candidates = None
    checked_prompts = 0
    if tokenizer_path is not None:
        if examples is None:
            raise ValueError("re-tokenizing archived prompts requires the original dataset zip")
        if sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA:
            raise ValueError("expected the pinned Pythia tokenizer.json")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for idx in sample_ids:
        before, after = (actual[reference, idx, layout] for layout in LAYOUTS)
        label_fields = ("label", "sentence_sha256", "token_multiset_sha256", "candidate_ids_sha256",
                        "tokens", "private_token_count")
        if (any(before[field] != after[field] for field in label_fields)
                or before["prompt_sha256"] == after["prompt_sha256"]
                or before["private_prefix_tokens"] >= after["private_prefix_tokens"]):
            raise ValueError(f"rewritten layout changed inventory or failed to expand prefix: {idx}")
        source_hash.update(struct.pack("<2q", idx, before["label"]))
        source_hash.update(bytes.fromhex(before["sentence_sha256"]))
        current_candidates = before["negative_token"], before["positive_token"]
        if candidates is None:
            candidates = current_candidates
        if candidates != current_candidates:
            raise ValueError(f"candidate rows vary with validation input at {idx}")
        if examples is not None:
            sentence, label = examples[idx]
            if before["label"] != label or before["sentence_sha256"] != sha(sentence.encode("utf-8")):
                raise ValueError(f"recorded label/review does not match official SST-2 dev row {idx}")
        private_ids = []
        for row in (before, after):
            layout = row["layout"]
            fields = (*label_fields, "prompt_sha256", "private_prefix_tokens", "private_char_start")
            if any(actual[model, idx, layout][field] != row[field]
                   for model in models for field in fields):
                raise ValueError(f"models did not share token inputs and labels at {idx}/{layout}")
            if tokenizer_path is None:
                continue
            header = CONTRACT["header"]
            note = CONTRACT["public_note"]
            start = header + (note if layout == "rewritten" else "") + CONTRACT["field"]
            tail = "\n" + note.rstrip("\n") if layout == "original" else ""
            encoding = tokenizer.encode(start + " " + sentence + tail + CONTRACT["query"],
                                        add_special_tokens=False)
            low, high = len(start), len(start) + 1 + len(sentence)
            span = [i for i, (left, right) in enumerate(encoding.offsets)
                    if right > low and left < high]
            if (not span or span != list(range(span[0], span[-1] + 1))
                    or tuple(row[k] for k in ("tokens", "private_prefix_tokens", "private_token_count",
                                               "prompt_sha256", "token_multiset_sha256")) != (
                        len(encoding.ids), span[0], len(span), digest_words(encoding.ids),
                        digest_words(sorted(encoding.ids)))):
                raise ValueError(f"independent tokenizer reproduction differs at {idx}/{layout}")
            private_ids.append(encoding.ids[span[0]:span[-1] + 1])
            checked_prompts += 1
        if private_ids and private_ids[0] != private_ids[1]:
            raise ValueError(f"secret token span changed between layouts at {idx}")
    expected_hash = FULL_DEV_SHA if metadata["dataset"].get("selection") == FULL_DEV else SAMPLE_SHA
    if source_hash.hexdigest() != expected_hash:
        raise ValueError("selected IDs, sentence fingerprints or labels differ from the pinned selection")
    if tokenizer_path is not None:
        encoded = [tokenizer.encode(" " + name, add_special_tokens=False).ids for name in CONTRACT["labels"]]
        if encoded != [[token] for token in candidates]:
            raise ValueError("binary completion token IDs do not match the pinned tokenizer")
    return checked_prompts


def verify_summary(actual, metadata, model):
    sample_ids = metadata["dataset"]["sample_ids"]
    pairs = [tuple(actual[model, idx, layout] for layout in LAYOUTS) for idx in sample_ids]
    summary = {}
    for position, layout in enumerate(LAYOUTS):
        group = [row[position] for row in pairs]
        protected = [row for row in group if row["protected_exact_ring"] == 1]
        expected = {
            "prompts": len(group),
            "native_correct": sum(row["native_choice"] == row["label"] for row in group),
            "mean_tokens": statistics.mean(row["tokens"] for row in group),
            "mean_prefix_tokens": statistics.mean(row["private_prefix_tokens"] for row in group),
            **{f"mean_projected_{field}": statistics.mean(int(row[f"projected_{field}"]) for row in group)
               for field in PAYLOAD},
            "protected_prompts": len(protected),
            "protected_correct": sum(row["protected_choice"] == row["label"] for row in protected),
            "protected_native_agreement": sum(row["protected_choice"] == row["native_choice"]
                                              for row in protected),
        }
        if protected:
            expected["protected_max_abs_logit_error"] = max(row["max_abs_logit_error"] for row in protected)
            expected["median_protected_elapsed_ms"] = statistics.median(
                row["protected_elapsed_ns"] for row in protected) / 1e6
        summary[layout] = expected
    accuracy = [(a["native_choice"] == a["label"], b["native_choice"] == b["label"]) for a, b in pairs]
    delta = transition_stats(accuracy)
    summary["paired"] = {
        "same_native_choices": sum(a["native_choice"] == b["native_choice"] for a, b in pairs),
        "correct_to_incorrect": delta["correct_to_incorrect"],
        "incorrect_to_correct": delta["incorrect_to_correct"],
        "same_protected_choices": sum(a["protected_choice"] == b["protected_choice"]
                                      for a, b in pairs if a["protected_exact_ring"] == 1),
    }
    if summary != metadata["summary"][model]:
        raise ValueError(f"archived per-layout summary disagrees with raw rows for {model}")
    return {**delta, "same_native_choices": summary["paired"]["same_native_choices"],
            "protected_native_agreement": {k: summary[k]["protected_native_agreement"] for k in LAYOUTS}}


def check(directory, dataset=None, tokenizer=None, require_full_tcp=False):
    metadata, models = read_metadata(directory)
    sample_ids = metadata["dataset"]["sample_ids"]
    if require_full_tcp and (set(models) != set(MODEL_NAMES)
                             or metadata["protected_examples_per_model"] != len(sample_ids)
                             or metadata["protected_transport"] != "TCP loopback"):
        raise ValueError("paper evidence requires both pinned models and TCP for every selected example")
    full_dev = metadata["dataset"].get("selection") == FULL_DEV
    examples = pinned_rows(dataset, full_dev=full_dev) if dataset is not None else None
    if examples is not None and list(examples) != sample_ids:
        raise ValueError("recorded review IDs differ from the pinned selection")
    actual = read_cases(directory / "sst2_layout_cases.csv", metadata, models)
    checked_prompts = check_input_pairs(actual, metadata, examples, tokenizer)
    return {
        "records": len(sample_ids),
        "independently_retokenized_prompts": checked_prompts,
        "native_layout_results": len(actual),
        "measured_protected_layout_results": 2 * len(models) * metadata["protected_examples_per_model"],
        "protected_transport": metadata["protected_transport"],
        "models": {model: verify_summary(actual, metadata, model) for model in models},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset-zip", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="pinned Pythia tokenizer.json")
    parser.add_argument("--require-full-tcp", action="store_true",
                        help="reject partial or non-TCP archives used for development")
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.dataset_zip, args.tokenizer, args.require_full_tcp), indent=2))
