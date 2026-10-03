"""Replay one failed MTP gate prompt and inspect the first differing target logits.

Uses the gate's saved input IDs and settings with one model loaded. Both CUDA
graphs belong to that same engine, so agreement here does not replace a gate
whose modes run in separate processes. This is a diagnostic, never a benchmark.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .measurement import first_difference
from .provenance import (ROOT, check_gpu_environment, checkpoint_manifest, file_hash,
                         object_hash, runtime_fingerprint, snapshot_sources, source_fingerprint)
from .single_gpu import write_json


def inspect_batch(reference, start, tokens, predictions):
    """Map new tokens to target verify rows; the committed input is already output."""
    visible = tokens[:max(0, len(reference) - start)]
    difference = first_difference(reference[start:start + len(visible)], visible)
    return {"output_start": start, "emitted_tokens": visible,
            "target_argmax": predictions, "collector_matches_target": tokens == predictions,
            "first_difference": start + difference if difference is not None else None}


def load_case(gate_dir, prompt_id=None):
    gate_dir = Path(gate_dir).expanduser().resolve()
    cfg = json.loads((gate_dir / "configuration.json").read_text())
    summary = json.loads((gate_dir / "summary.json").read_text())
    if summary.get("stage") != "gate" or summary.get("configuration_sha256") != object_hash(cfg):
        raise ValueError("expected a gate with an intact configuration record")
    comparison = summary["output_comparison"]
    if comparison.get("reference") != "graph":
        raise ValueError("this diagnostic requires graph as the gate reference")
    failures = [c for c in comparison["checks"] if c["mode"] == "mtp" and not c["equal"]]
    if prompt_id is None:
        if not failures:
            raise ValueError("the gate has no failed MTP comparison")
        prompt_id = failures[0]["prompt_id"]
    matches = [c for c in failures if c["prompt_id"] == prompt_id]
    if not matches:
        raise ValueError(f"no failed MTP comparison for {prompt_id}")
    failed = matches[0]
    rows = {}
    for mode, repeat in (("graph", 0), ("mtp", failed["round"])):
        candidates = [json.loads(line) for line in (gate_dir / f"{mode}.jsonl").read_text().splitlines()]
        selected = [r for r in candidates if r["phase"] == "measure"
                    and r["prompt_id"] == prompt_id and r["round"] == repeat]
        if len(selected) != 1:
            raise ValueError(f"expected one saved {mode} row for {prompt_id}, round {repeat}")
        rows[mode] = selected[0]
    if rows["graph"]["prompt_token_ids"] != rows["mtp"]["prompt_token_ids"]:
        raise ValueError("saved prompt IDs differ; resolve the input mismatch first")
    if cfg["respect_eos"]:
        raise ValueError("this diagnostic currently requires the fixed-length gate")
    if any(len(r["token_ids"]) != cfg["tokens"] for r in rows.values()):
        raise ValueError("saved output length differs from the gate configuration")
    difference = first_difference(rows["graph"]["token_ids"], rows["mtp"]["token_ids"])
    if difference is None or difference != failed["first_difference"]:
        raise ValueError("saved outputs do not reproduce the gate's recorded first difference")
    return cfg, summary, rows


def validate_sources(fingerprints):
    # New diagnostic files are allowed; every source present during the old gate
    # must still match. In particular this checks the engine and the GPU adapter.
    for name, digest in fingerprints.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or file_hash(source) != digest:
            raise ValueError(f"source changed since this gate: {name}; diagnose with its original code")


def logits_brief(logits, tokenizer):
    import torch
    values = logits.float()
    finite = bool(torch.isfinite(values).all())
    result = {"finite": finite, "argmax": int(values.argmax())}
    if finite:
        scores, ids = values.topk(min(5, values.numel()))
        result["top1_top2_gap"] = float(scores[0] - scores[1]) if len(scores) > 1 else None
        result["top5"] = [{"id": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
                          for i, v in zip(ids, scores)]
    return result


def explain_difference(index, reference, actual, ref_logits, mtp_logits, tokenizer, step=None):
    import torch
    left, right = ref_logits.float(), mtp_logits.float()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return {"output_index_zero_based": index, "step": step,
            "common_prefix_tail": tokenizer.decode(reference[max(0, index - 16):index]),
            "graph_token": {"id": reference[index], "text": tokenizer.decode([reference[index]])},
            "mtp_token": {"id": actual, "text": tokenizer.decode([actual])},
            "graph_logits": logits_brief(left, tokenizer),
            "mtp_target_logits": logits_brief(right, tokenizer),
            "max_abs_logit_difference": float((left - right).abs().max()) if finite else None,
            "mean_abs_logit_difference": float((left - right).abs().mean()) if finite else None,
            "candidate_logits": [
                {"id": token, "graph": float(left[token]) if finite else None,
                 "mtp_target": float(right[token]) if finite else None}
                for token in sorted({reference[index], actual})]}


def replay(adapter, prompt_ids, count):
    """Keep raw logits on CPU; speculative logits are retained only at the first mismatch."""
    engine = adapter.engine
    adapter.mode = "graph"
    print("Replaying graph reference...", flush=True)
    reference = [adapter.prime(prompt_ids)]
    raw_logits = [engine.w.lm_head(engine.last_hidden)[-1].detach().cpu()]
    for _ in range(count - 1):
        reference.extend(adapter.step().tokens)
        raw_logits.append(engine.logits[0].detach().cpu())

    adapter.mode = "mtp"
    print("Replaying MTP and checking each emitted token against target argmax...", flush=True)
    actual = [adapter.prime(prompt_ids)]
    first_logits = engine.w.lm_head(engine.spec_hidden[:1])[0].detach().cpu()
    first = None
    if actual[0] != reference[0]:
        first = explain_difference(0, reference, actual[0], raw_logits[0], first_logits, adapter.tokenizer)
    steps = []
    while len(actual) < count:
        start, position, committed = len(actual), engine.state.pos, int(engine.tok[0])
        batch = adapter.step()
        if not batch.tokens:
            raise RuntimeError("MTP returned an empty batch")
        target_logits = engine.spec_logits[:batch.accepted + 1].detach().cpu()
        predictions = target_logits.argmax(-1).tolist()
        record = inspect_batch(reference, start, batch.tokens, predictions)
        record.update(step=len(steps), accepted=batch.accepted, proposed=batch.proposed,
                      drafts=engine.drafts[:adapter.k].tolist(), committed_input=committed,
                      committed_matches_previous_output=committed == actual[-1],
                      position_before=position, position_after=engine.state.pos,
                      position_device=int(engine.state.pos_t[0]), slot=int(engine.state.slot[0]))
        record["state_bookkeeping_matches"] = (record["position_after"] == position + batch.accepted + 1
                                               == record["position_device"] and record["slot"] == batch.accepted)
        actual.extend(record["emitted_tokens"])
        steps.append(record)
        index = record["first_difference"]
        if first is None and index is not None:
            row = index - start
            first = explain_difference(index, reference, actual[index], raw_logits[index], target_logits[row],
                                       adapter.tokenizer, {**record, "verify_row": row})
    return {"graph_token_ids": reference, "mtp_token_ids": actual, "steps": steps,
            "first_difference": first,
            "collector_matches_target": int(first_logits.argmax()) == actual[0]
                and all(s["collector_matches_target"] for s in steps),
            "state_bookkeeping_matches": all(s["state_bookkeeping_matches"]
                and s["committed_matches_previous_output"] for s in steps)}


def run(args):
    cfg, summary, saved = load_case(args.gate_dir, args.prompt)
    validate_sources(cfg["sources"])
    if checkpoint_manifest(cfg["model"]["path"]) != cfg["model"]:
        raise ValueError("checkpoint metadata/weight size or mtime changed since the gate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = Path(args.output or ROOT / "runs" / f"diagnose-mtp-{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "gate_dir": str(Path(args.gate_dir).resolve()),
              "prompt_id": saved["graph"]["prompt_id"], "options": vars(args),
              "gate_configuration_sha256": summary["configuration_sha256"],
              "saved_first_difference": first_difference(saved["graph"]["token_ids"], saved["mtp"]["token_ids"]),
              "note": "Diagnostic only. Both paths share one loaded engine. Does not pass or modify a gate, "
                      "measure performance, or establish independent HF/BF16 correctness."}
    print(f"Diagnostic results: {output}", flush=True)
    try:
        sources = source_fingerprint()
        write_json(output / "configuration.json", {"gate_configuration": cfg, "diagnostic_sources": sources})
        snapshot_sources(output / "sources", sources)
        env = check_gpu_environment()
        write_json(output / "environment.json", env)
        if not env["passed"] or runtime_fingerprint(env) != summary["runtime_sha256"]:
            raise ValueError("GPU / driver / Python stack differs from the gate or fails its environment check")
        from .gpu import Adapter
        import torch
        options = argparse.Namespace(stage="gate", model=cfg["model"]["path"], backend=cfg["backend"],
                                     kv=cfg["kv"], mtp_depth=cfg["mtp_depth"], tokens=cfg["tokens"],
                                     max_len=cfg["max_len"], chunk=cfg["chunk"])
        slots = max(cfg["mtp_depth"], 7 if "dflash" in cfg["modes"] else 0)
        with torch.inference_mode():
            adapter = Adapter(options, "mtp", slots)
            result = replay(adapter, saved["graph"]["prompt_token_ids"], cfg["tokens"])
        report.update(result)
        report["replay_vs_saved_first_difference"] = {
            mode: first_difference(result[f"{mode}_token_ids"], saved[mode]["token_ids"])
            for mode in ("graph", "mtp")}
        from tokenrush import quant
        report["rows_for_one"] = quant.ROWS_FOR_ONE
        report["rows_kernel_autotune"] = [
            {"key": str(key), "config": str(config)}
            for key, config in getattr(quant._int4_gemm_rows_kernel, "cache", {}).items()]
        report["status"] = "complete"
        console = {key: report[key] for key in ("prompt_id", "saved_first_difference",
                   "replay_vs_saved_first_difference", "collector_matches_target", "state_bookkeeping_matches",
                   "rows_for_one", "first_difference")}
        print(json.dumps(console, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print(f"Diagnostic complete; gate status unchanged. Full report: {output / 'diagnosis.json'}", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "diagnosis.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", required=True, help="existing gate results; read-only")
    parser.add_argument("--prompt", help="default: first failing MTP prompt")
    parser.add_argument("--output", help="new diagnostic directory; existing directories are refused")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
