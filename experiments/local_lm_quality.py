"""Pretrained Pythia diagnostic for token dependence and exact-prefix reuse.

Install PyTorch 2.8.0 and Transformers 4.52.4, then pass the pinned checkpoint
directory with --checkpoint. Runs on CPU and uses only fixed, synthetic records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "EleutherAI/pythia-70m-deduped"
REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
WEIGHT_SHA256 = "3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd"
TV_THRESHOLD = 1e-5
RECORDS = (
    ("Order", "City", "The shipping address has been verified.", ("Paris", "London", "Berlin", "Rome")),
    ("Travel", "Destination", "The trip is scheduled for next week.", ("Paris", "London", "Berlin", "Rome")),
    ("Mail", "Recipient", "The envelope is ready for delivery.", ("John", "Mary", "David", "James")),
    ("Clinic", "Name", "The appointment is listed for Tuesday.", ("John", "Mary", "David", "James")),
    ("Stock", "Color", "The item is available at the main store.", ("red", "blue", "green", "black")),
    ("Rental", "Vehicle", "The reservation is valid for one day.", ("car", "truck", "bike", "van")),
    ("Kitchen", "Fruit", "The ingredients are ready for the next meal.", ("apple", "orange", "banana", "lemon")),
    ("Library", "Subject", "The returned book is ready to shelve.", ("history", "science", "music", "art")),
    ("Roster", "Team", "Practice is scheduled for the morning.", ("red", "blue", "green", "black")),
    ("Garden", "Tree", "The soil has been prepared for planting.", ("oak", "pine", "maple", "ash")),
    ("Receipt", "Total", "The purchase is complete.", ("10", "20", "30", "40")),
    ("Weather", "Season", "The report covers the usual conditions.", ("winter", "summer", "spring", "fall")),
)


def verify_checkpoint(checkpoint):
    with (checkpoint / "model.safetensors").open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Checkpoint weights do not match the pinned Pythia revision")


def matched_prompts(tokenizer):
    for record, field, public_note, options in RECORDS:
        for left, right in zip(options[::2], options[1::2]):
            header = f"{record} record.\n"
            slot = f"{field}: {{}}\n"
            context = public_note + "\n"
            query = f"{field}:"
            options_by_placement = {}
            for placement, body in (("early", slot + context), ("late", context + slot)):
                template = header + body + query
                ids = [tokenizer.encode(template.format(value), add_special_tokens=False)
                       for value in (left, right)]
                if len(ids[0]) != len(ids[1]):
                    raise ValueError(f"Unequal treatment token lengths in {record}, {left}/{right}")
                changed = [i for i, (a, b) in enumerate(zip(*ids)) if a != b]
                if len(changed) != 1 or changed[0] in (0, len(ids[0]) - 1):
                    raise ValueError(f"Expected one interior changed token in {record}, {left}/{right}")
                options_by_placement[placement] = changed[0], ids
            early, late = options_by_placement.values()
            if any(sorted(a) != sorted(b) for a, b in zip(early[1], late[1])):
                raise ValueError(f"Moving the private field changed token inventory in {record}")
            for placement, (secret, ids) in options_by_placement.items():
                yield record, placement, (left, right), secret, torch.tensor(ids)


@torch.inference_mode()
def evaluate(checkpoint):
    verify_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    rows = []
    for record, placement, values, secret, ids in matched_prompts(tokenizer):
        logits = model(ids, use_cache=False).logits
        probs = logits.softmax(-1)
        tv = (probs[0] - probs[1]).abs().sum(-1) / 2
        targets = logits.argmax(-1)
        prefix = model(ids[:1, :secret], use_cache=True)
        prefix.past_key_values.batch_repeat_interleave(repeats=2)
        tail = model(ids[:, secret:], past_key_values=prefix.past_key_values, use_cache=True)
        split_logits = torch.cat((prefix.logits.expand(2, -1, -1), tail.logits), dim=1)
        split_tv = (probs - split_logits.softmax(-1)).abs().sum(-1) / 2
        if tv[:secret].max().item() > TV_THRESHOLD:
            raise ValueError(f"Equal public prefixes diverged in {record}, {placement}, {values}")
        rows.append({
            "record": record, "placement": placement, "values": values,
            "sequence_tokens": ids.size(1), "public_prefix_tokens": secret,
            "max_prefix_tv": tv[:secret].max().item(),
            "suffix_positions_changed": (tv[secret + 1:] > TV_THRESHOLD).sum().item(),
            "final_tv": tv[-1].item(),
            "final_top1_changes": int(targets[0, -1] != targets[1, -1]),
            "final_field_exact": (targets[:, -1] == ids[:, secret]).sum().item(),
            "prefix_reuse_max_tv": split_tv.max().item(),
            "prefix_reuse_max_logit_error": (logits - split_logits).abs().max().item(),
            "prefix_reuse_argmax_matches": (targets == split_logits.argmax(-1)).sum().item(),
        })

    by_placement = {}
    for label in ("early", "late"):
        subset = [row for row in rows if row["placement"] == label]
        by_placement[label] = {
            "prompt_pairs": len(subset),
            "avg_public_prefix_fraction": statistics.mean(
                row["public_prefix_tokens"] / row["sequence_tokens"] for row in subset),
            "nominal_public_suffix_positions": sum(
                row["sequence_tokens"] - row["public_prefix_tokens"] - 1 for row in subset),
            "nominal_public_suffix_positions_changed": sum(row["suffix_positions_changed"] for row in subset),
            "median_final_tv": statistics.median(row["final_tv"] for row in subset),
            "final_top1_changes": sum(row["final_top1_changes"] for row in subset),
            "field_exact": sum(row["final_field_exact"] for row in subset),
            "field_examples": 2 * len(subset),
            "max_prefix_tv": max(row["max_prefix_tv"] for row in subset),
        }
    return {
        "model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
        "implementation": "Transformers eager CPU float32",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "threads": torch.get_num_threads(), "tv_threshold": TV_THRESHOLD,
        "by_placement": by_placement,
        "prefix_reuse_argmax_matches": sum(row["prefix_reuse_argmax_matches"] for row in rows),
        "prefix_reuse_argmax_total": sum(2 * row["sequence_tokens"] for row in rows),
        "prefix_reuse_max_logit_error": max(row["prefix_reuse_max_logit_error"] for row in rows),
        "prefix_reuse_max_tv": max(row["prefix_reuse_max_tv"] for row in rows),
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = evaluate(args.checkpoint)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
