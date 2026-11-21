"""Trusted-client hidden states and public LM-head rows for the Pythia probe."""
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_lm_quality import RECORDS, matched_prompts, verify_checkpoint


def answer_tokens(tokenizer):
    tokens = {}
    for record, _, _, values in RECORDS:
        choices = [tokenizer.encode(" " + value, add_special_tokens=False) for value in values]
        if any(len(ids) != 1 for ids in choices) or len({ids[0] for ids in choices}) != 4:
            raise ValueError(f"Expected four distinct single-token candidates for {record}")
        tokens[record] = torch.tensor([ids[0] for ids in choices])
    return tokens


def complete_prefill(model, ids, _):
    return model(ids, use_cache=False).last_hidden_state[:, -1, :]


def cached_prefill(model, ids, public_len):
    if not torch.equal(ids[0, :public_len], ids[1, :public_len]):
        raise ValueError("Prefix mismatch in a paired request")
    prefix = model(ids[:1, :public_len], use_cache=True)
    prefix.past_key_values.batch_repeat_interleave(repeats=2)
    return model(ids[:, public_len:], past_key_values=prefix.past_key_values,
                 use_cache=True).last_hidden_state[:, -1, :]


@torch.inference_mode()
def load_inputs(checkpoint):
    """Return FP32 public head weights and matched pair cases for two-party scoring."""
    verify_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    head = model.get_output_embeddings().weight
    candidates = answer_tokens(tokenizer)
    prompts = list(matched_prompts(tokenizer))
    causal_body = model.base_model
    # Warm both execution paths once before timing.
    _, _, _, first_public_len, first_ids = prompts[0]
    complete_prefill(causal_body, first_ids, first_public_len)
    cached_prefill(causal_body, first_ids, first_public_len)

    cases = []
    for case_id, (record, placement, _, public_len, ids) in enumerate(prompts):
        candidate_ids = candidates[record]
        target_ids = ids[:, public_len]
        if not torch.isin(target_ids, candidate_ids).all():
            raise ValueError(f"Missing private field value in public candidate set for {record}")
        head_rows = head[candidate_ids]
        case = {
            "case_id": case_id, "record": record, "placement": placement,
            "candidate_ids": candidate_ids.numpy().copy(), "target_ids": target_ids.numpy().copy(),
            "prefix_tokens": public_len, "sequence_tokens": ids.shape[1],
        }
        arms = [("full", complete_prefill), ("cached", cached_prefill)]
        # Pairs are grouped by placement in matched_prompts. Rotate which arm is
        # timed first across records so both placements see both orders.
        if (case_id // 2) % 2:
            arms.reverse()
        for label, prefill in arms:
            start = time.perf_counter_ns()
            hidden = prefill(causal_body, ids, public_len)
            case[f"prefill_{label}_ns"] = time.perf_counter_ns() - start
            case[f"hidden_{label}"] = np.ascontiguousarray(hidden.numpy())
            score_label = "logits_fp32" if label == "full" else "logits_cached_fp32"
            case[score_label] = (hidden @ head_rows.T).numpy()
        cases.append(case)
    return np.ascontiguousarray(head.detach().numpy()), cases
