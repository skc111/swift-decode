"""CPU-testable output accounting; the GPU adapter supplies prime/step/sync.

prime returns the first output token. step returns NEW tokens following it,
not the committed input token again. A speculative step may return a batch.
"""
from dataclasses import dataclass
import statistics
import time


@dataclass
class Step:
    tokens: list[int]
    accepted: int = 0
    proposed: int = 0


def measure(prime, step, synchronize, max_new, stop_ids=(), clock=time.perf_counter):
    if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 1:
        raise ValueError("max_new must be a positive integer")
    stop_ids = set(stop_ids)
    synchronize()
    start = clock()
    first = int(prime())
    synchronize()
    first_ready = clock()
    output = [first]
    events = []
    computed = accepted = proposed = 0
    finish = first_ready
    while len(output) < max_new and output[-1] not in stop_ids:
        batch = step()
        if not batch.tokens:
            raise RuntimeError("step produced no tokens; refusing a non-progressing loop")
        if not 0 <= batch.accepted <= batch.proposed:
            raise ValueError("invalid speculative acceptance counts")
        computed += len(batch.tokens)
        accepted += batch.accepted
        proposed += batch.proposed
        emitted = 0
        for token in batch.tokens:
            output.append(int(token))
            emitted += 1
            if len(output) == max_new or output[-1] in stop_ids:
                break
        synchronize()
        finish = clock()
        events.append({"elapsed_s": finish - start, "emitted": emitted,
                       "computed": len(batch.tokens), "accepted": batch.accepted,
                       "proposed": batch.proposed})
    decode = finish - first_ready
    total = finish - start
    post_first = len(output) - 1
    stats = {
        "new_tokens": len(output),
        "first_token_s": first_ready - start,
        "decode_s": decode,
        "total_s": total,
        "decode_output_tok_s": post_first / decode if post_first and decode > 0 else None,
        "output_tok_s": len(output) / total if total > 0 else None,
        "tpot_s": decode / post_first if post_first else None,
        "steps": len(events),
        "emitted_tokens_per_step": post_first / len(events) if events else None,
        "accepted_drafts": accepted,
        "proposed_drafts": proposed,
        "acceptance_rate": accepted / proposed if proposed else None,
        "discarded_tail_tokens": computed - post_first,
        "stop_reason": "eos" if output[-1] in stop_ids else "length",
    }
    return {"token_ids": output, "stats": stats, "step_events": events}


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def summarize(rows):
    groups = {}
    for row in rows:
        if row["phase"] != "measure":
            continue
        groups.setdefault((row["mode"], row["prompt_id"]), []).append(row)
    result = []
    metrics = ("first_token_s", "decode_output_tok_s", "output_tok_s", "tpot_s",
               "acceptance_rate", "peak_allocated_bytes", "peak_reserved_bytes")
    for (mode, prompt), repeats in sorted(groups.items()):
        item = {"mode": mode, "prompt_id": prompt, "measured_rounds": len(repeats)}
        for metric in metrics:
            values = [r["stats"][metric] for r in repeats if r["stats"].get(metric) is not None]
            item[metric] = ({"median": statistics.median(values), "min": min(values),
                             "max": max(values)} if values else None)
        result.append(item)
    return result


def compare_outputs(rows, modes, prompt_ids, repeats, *, references=None):
    """Check every measured row against graph round 0 or saved per-prompt references."""
    selected = [r for r in rows if r["phase"] == "measure"]
    by_key = {}
    for row in selected:
        key = (row["mode"], row["prompt_id"], row["round"])
        if key in by_key:
            raise ValueError(f"duplicate result {key}")
        by_key[key] = row
    expected = {(m, p, r) for m in modes for p in prompt_ids for r in range(repeats)}
    if set(by_key) != expected:
        raise ValueError(f"incomplete results: missing={expected - set(by_key)}, extra={set(by_key) - expected}")
    checks = []
    reference_mode = "graph" if "graph" in modes else modes[0]
    if references is not None and set(references) != set(prompt_ids):
        raise ValueError("saved references do not cover exactly the experiment's prompts")
    for prompt in prompt_ids:
        ref = by_key[(reference_mode, prompt, 0)] if references is None else references[prompt]
        for mode in modes:
            for repeat in range(repeats):
                row = by_key[(mode, prompt, repeat)]
                same_input = row["prompt_token_ids"] == ref["prompt_token_ids"]
                difference = first_difference(ref["token_ids"], row["token_ids"])
                checks.append({"mode": mode, "prompt_id": prompt, "round": repeat,
                               "same_input": same_input, "first_difference": difference,
                               "equal": same_input and difference is None})
    return {"all_equal": all(c["equal"] for c in checks), "reference": reference_mode, "checks": checks}
