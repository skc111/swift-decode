"""Locate a saved MTP divergence before or inside the final quantized projection.

Replays only through the first mismatch. Restores the saved Triton row-kernel
autotune choices, then projects frozen raw/speculative hidden states at M=1 and
M=K+1. A final probe gives the batched head the singleton's launch configuration.
All interventions are process-local diagnostics; no gate or engine file is edited.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .diagnose_mtp import load_case, logits_brief, validate_sources
from .measurement import first_difference
from .provenance import (ROOT, check_gpu_environment, checkpoint_manifest, runtime_fingerprint,
                         snapshot_sources, source_fingerprint)
from .single_gpu import write_json


def restore_rows_cache(quant, entries):
    """Reuse only known upstream configs; never evaluate code from the saved report."""
    choices = {str(config): config for config in quant._ROWS_CONFIGS}
    restored = {}
    for item in entries:
        key = ast.literal_eval(item["key"])
        if (not isinstance(key, tuple) or len(key) < 3
                or any(type(n) is not int or n < 1 for n in key[:3])
                or not all(isinstance(dtype, str) for dtype in key[3:])):
            raise ValueError("unrecognized saved row-kernel cache key")
        if item["config"] not in choices:
            raise ValueError("saved row-kernel config is not available in this pinned Triton environment")
        if key in restored:
            raise ValueError("duplicate saved row-kernel cache key")
        restored[key] = choices[item["config"]]
    if not restored:
        raise ValueError("the saved diagnostic has no row-kernel autotune choices")
    cache = quant._int4_gemm_rows_kernel.cache
    cache.update(restored)
    return len(restored)


@contextmanager
def shared_head_config(quant, vocab, hidden, rows):
    """Temporarily change only the batched lm_head's autotune-cache entry."""
    cache = quant._int4_gemm_rows_kernel.cache
    single = [key for key in cache if key[:3] == (vocab, hidden, 1)]
    if len(single) != 1:
        raise ValueError("expected one singleton lm_head config in the autotune cache")
    single = single[0]
    multi = (*single[:2], rows, *single[3:])
    if multi not in cache:
        raise ValueError("batched lm_head config is missing from the autotune cache")
    original = cache[multi]
    info = {"singleton": str(cache[single]), "batched_original": str(original),
            "batched_probe": str(cache[single])}
    cache[multi] = cache[single]
    try:
        yield info
    finally:
        cache[multi] = original


def probe_plan(diagnosis, saved):
    if diagnosis.get("status") != "complete":
        raise ValueError("the saved diagnostic did not complete")
    for mode in ("graph", "mtp"):
        if diagnosis[f"{mode}_token_ids"] != saved[mode]["token_ids"]:
            raise ValueError(f"saved diagnostic did not reproduce the gate's {mode} output")
    difference = diagnosis.get("first_difference") or {}
    step = difference.get("step")
    if step is None:
        raise ValueError("this probe requires a mismatch during a speculative step, after prefill")
    index, start, row = difference["output_index_zero_based"], step["output_start"], step["verify_row"]
    actual = first_difference(saved["graph"]["token_ids"], saved["mtp"]["token_ids"])
    if (index != actual or start < 1 or index != start + row or not 0 <= row <= step["accepted"]
            or step["committed_input"] != saved["mtp"]["token_ids"][start - 1]):
        raise ValueError("saved mismatch does not align with the committed input and verify row")
    if diagnosis["steps"][step["step"]] != {k: v for k, v in step.items() if k != "verify_row"}:
        raise ValueError("saved mismatch step differs from the diagnostic step trace")
    return {"index": index, "step": step["step"], "row": row}


def tensor_difference(left, right):
    import torch
    left, right = left.detach().float().cpu(), right.detach().float().cpu()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return {"equal": bool(torch.equal(left, right)), "finite": finite,
            "different_values": int((left != right).sum()), "values": left.numel(),
            "max_abs": float((left - right).abs().max()) if finite else None,
            "mean_abs": float((left - right).abs().mean()) if finite else None}


def replay_hiddens(adapter, diagnosis, saved, plan):
    engine = adapter.engine
    prompt = saved["graph"]["prompt_token_ids"]
    adapter.mode = "mtp"
    output = [adapter.prime(prompt)]
    if output != saved["mtp"]["token_ids"][:1]:
        raise RuntimeError("MTP first token no longer reproduces the saved diagnostic")
    print(f"Replaying MTP through speculative step {plan['step']} (zero based)...", flush=True)
    for step_index in range(plan["step"] + 1):
        expected = diagnosis["steps"][step_index]
        if len(output) != expected["output_start"] or engine.state.pos != expected["position_before"]:
            raise RuntimeError(f"MTP state before step {step_index} differs from the saved diagnostic")
        batch = adapter.step()
        if (batch.tokens != expected["emitted_tokens"] or batch.accepted != expected["accepted"]
                or engine.drafts[:adapter.k].tolist() != expected["drafts"]
                or engine.state.pos != expected["position_after"]):
            raise RuntimeError(f"MTP step {step_index} no longer reproduces the saved diagnostic")
        output.extend(batch.tokens)
    hidden_mtp = engine.spec_hidden[:adapter.k + 1].clone()
    captured_logits = engine.spec_logits[plan["row"]].detach().cpu()

    # Eager single-token decode exposes its actual post-norm hidden. Its tokens
    # must reproduce the saved graph prefix before the projection probe proceeds.
    adapter.mode = "eager"
    print(f"Replaying eager reference through output index {plan['index']}...", flush=True)
    output = [adapter.prime(prompt)]
    for _ in range(plan["index"]):
        output.extend(adapter.step().tokens)
    if output != saved["graph"]["token_ids"][:plan["index"] + 1]:
        raise RuntimeError("eager reference no longer reproduces the saved graph prefix")
    return engine.last_hidden.clone(), hidden_mtp, captured_logits


def project_frozen_hiddens(adapter, hidden_raw, hidden_mtp, captured_logits, row):
    from tokenrush import quant
    head = adapter.engine.w.lm_head
    count = hidden_mtp.shape[0]
    raw_batch = hidden_raw.repeat(count, 1)
    mtp_single = hidden_mtp[row:row + 1].contiguous()
    logits = {"raw_hidden_single": head(hidden_raw)[0].detach().cpu(),
              "raw_hidden_batch": head(raw_batch)[row].detach().cpu(),
              "mtp_hidden_single": head(mtp_single)[0].detach().cpu(),
              "mtp_hidden_batch": head(hidden_mtp)[row].detach().cpu()}
    original_reproduced = tensor_difference(logits["mtp_hidden_batch"], captured_logits)
    if not original_reproduced["equal"]:
        raise RuntimeError("reprojecting the saved speculative hidden does not reproduce captured target logits")
    with shared_head_config(quant, head.shape[0], head.shape[1], count) as configs:
        logits["raw_hidden_batch_shared_config"] = head(raw_batch)[row].detach().cpu()
        logits["mtp_hidden_batch_shared_config"] = head(hidden_mtp)[row].detach().cpu()
    pairs = (("raw_hidden_single", "raw_hidden_batch"), ("mtp_hidden_single", "mtp_hidden_batch"),
             ("raw_hidden_single", "raw_hidden_batch_shared_config"),
             ("mtp_hidden_single", "mtp_hidden_batch_shared_config"),
             ("raw_hidden_single", "mtp_hidden_single"))
    return {"head_configs": configs,
            "hidden_difference": tensor_difference(hidden_raw[0], mtp_single[0]),
            "captured_mtp_logits_reproduced": original_reproduced,
            "logit_differences": {f"{left} vs {right}": tensor_difference(logits[left], logits[right])
                                  for left, right in pairs},
            "projections": {name: logits_brief(values, adapter.tokenizer) for name, values in logits.items()}}


def run(args):
    directory = Path(args.diagnosis_dir).expanduser().resolve()
    diagnosis = json.loads((directory / "diagnosis.json").read_text())
    cfg, summary, saved = load_case(diagnosis["gate_dir"], diagnosis["prompt_id"])
    if diagnosis["gate_configuration_sha256"] != summary["configuration_sha256"]:
        raise ValueError("diagnostic and gate configurations differ")
    if cfg["backend"] != "triton":
        raise ValueError("this probe is specifically for the Triton backend")
    plan = probe_plan(diagnosis, saved)
    validate_sources(cfg["sources"])
    diagnostic_cfg = json.loads((directory / "configuration.json").read_text())
    validate_sources(diagnostic_cfg["diagnostic_sources"])
    if checkpoint_manifest(cfg["model"]["path"]) != cfg["model"]:
        raise ValueError("checkpoint record changed since the gate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = Path(args.output or ROOT / "runs" / f"probe-mtp-head-{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "diagnosis_dir": str(directory), "plan": plan,
              "note": "Frozen-hidden projection diagnostic, not a gate or benchmark. "
                      "Saved autotune choices are restored before replay. Only the final eager head probe "
                      "temporarily changes the batched lm_head config; existing captured graphs are unchanged."}
    print(f"Head probe results: {output}", flush=True)
    try:
        sources = source_fingerprint()
        snapshot_sources(output / "sources", sources)
        write_json(output / "configuration.json", {"gate": cfg, "sources": sources, "options": vars(args)})
        env = check_gpu_environment()
        write_json(output / "environment.json", env)
        if not env["passed"] or runtime_fingerprint(env) != summary["runtime_sha256"]:
            raise ValueError("environment differs from the gate or fails its environment check")
        import os
        os.environ["TOKENRUSH_BACKEND"] = "triton"
        from tokenrush import quant
        report["restored_autotune_entries"] = restore_rows_cache(quant, diagnosis["rows_kernel_autotune"])
        from .gpu import Adapter
        import torch
        options = argparse.Namespace(stage="gate", model=cfg["model"]["path"], backend=cfg["backend"],
                                     kv=cfg["kv"], mtp_depth=cfg["mtp_depth"], tokens=cfg["tokens"],
                                     max_len=cfg["max_len"], chunk=cfg["chunk"])
        with torch.inference_mode():
            adapter = Adapter(options, "mtp", max(cfg["mtp_depth"], 7 if "dflash" in cfg["modes"] else 0))
            raw, mtp, captured = replay_hiddens(adapter, diagnosis, saved, plan)
            report.update(project_frozen_hiddens(adapter, raw, mtp, captured, plan["row"]))
        report["status"] = "complete"
        console = {key: report[key] for key in ("plan", "head_configs", "hidden_difference", "logit_differences")}
        console["projections"] = {name: {**values, "top5": values.get("top5", [])[:2]}
                                  for name, values in report["projections"].items()}
        print(json.dumps(console, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print(f"Probe complete; gate status unchanged. Full report: {output / 'probe.json'}", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "probe.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", required=True, help="completed diagnose_mtp results; read-only")
    parser.add_argument("--output", help="new probe directory; existing directories are refused")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
