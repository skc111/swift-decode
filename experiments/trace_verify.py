"""Trace the first arithmetic difference between sequential and batched verification.

Use identical tokens and a restored target-state snapshot. Scan the saved MTP
blocks through its first output mismatch, then test GDN tiling and row-kernel
configuration separately and together. This diagnostic never passes a gate.
"""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

from .diagnose_mtp import load_case, validate_sources
from .probe_mtp_head import probe_plan, restore_rows_cache, tensor_difference
from .provenance import (ROOT, check_gpu_environment, checkpoint_manifest, runtime_fingerprint,
                         snapshot_sources, source_fingerprint)
from .single_gpu import write_json


def block_inputs(diagnosis, saved, step_index, depth):
    step = diagnosis["steps"][step_index]
    start = step["output_start"]
    prompt_length = len(saved["graph"]["prompt_token_ids"])
    if (not 1 <= start < len(saved["mtp"]["token_ids"])
            or step["step"] != step_index
            or not 0 <= step["accepted"] <= depth
            or step["position_before"] != prompt_length + start - 1
            or len(step["drafts"]) != depth or step["proposed"] != depth
            or step["committed_input"] != saved["mtp"]["token_ids"][start - 1]
            or saved["graph"]["token_ids"][:start] != saved["mtp"]["token_ids"][:start]):
        raise ValueError("saved block does not share the reference's committed prefix")
    emitted = step["target_argmax"]
    if (len(emitted) != step["accepted"] + 1
            or emitted != saved["mtp"]["token_ids"][start:start + len(emitted)]
            or emitted[:-1] != step["drafts"][:step["accepted"]]):
        raise ValueError("saved target rows do not align with the accepted drafts and bonus token")
    return [step["committed_input"], *step["drafts"]]


@contextmanager
def common_row_configs(quant, shapes, rows, enabled):
    """Match target projections to their singleton configs; leave draft shapes alone."""
    cache = quant._int4_gemm_rows_kernel.cache
    pending = []
    if enabled:
        for key, original in list(cache.items()):
            if key[2] == rows and key[:2] in shapes:
                singleton = (*key[:2], 1, *key[3:])
                if singleton not in cache:
                    raise ValueError(f"missing singleton configuration for {key[:2]}")
                pending.append((key, original, cache[singleton]))
        if {key[:2] for key, _, _ in pending} != shapes:
            raise ValueError("not every target projection has a cached batched configuration")
    for key, _, replacement in pending:
        cache[key] = replacement
    try:
        yield [{"shape": list(key[:2]), "original": str(old), "probe": str(new)}
               for key, old, new in pending]
    finally:
        for key, original, _ in pending:
            cache[key] = original


class StateSnapshot:
    """Only live recurrent/KV data are meaningful; keep the entire conv ring for restore."""
    def __init__(self, state):
        self.pos, self.slot = state.pos, state.slot_h
        self.rec = state.rec[state.slot_h].clone()
        self.conv = state.conv.clone()
        self.k = state.k[:, :, :state.pos].clone()
        self.v = state.v[:, :, :state.pos].clone()

    def restore(self, state):
        state.rec.zero_()
        state.rec[self.slot].copy_(self.rec)
        state.conv.copy_(self.conv)
        state.k[:, :, :self.pos].copy_(self.k)
        state.v[:, :, :self.pos].copy_(self.v)
        state.pos = self.pos
        state.pos_t.fill_(self.pos)
        state.slot_h = self.slot
        state.slot.fill_(self.slot)

    def compare(self, state):
        return {"position_equal": self.pos == state.pos == int(state.pos_t[0]),
                "live_recurrent": tensor_difference(self.rec, state.rec[state.slot_h]),
                "conv_ring": tensor_difference(self.conv, state.conv),
                "key_prefix": tensor_difference(self.k, state.k[:, :, :self.pos]),
                "value_prefix": tensor_difference(self.v, state.v[:, :, :self.pos])}


class OperationTrace:
    """Observe eager operators without changing their arithmetic (unless BV is requested).

    CPU copies intentionally synchronize. Split-K partials are [S,T,N], so move
    axis 1 to the front before aligning a batch with T sequential forwards.
    """
    def __init__(self, engine, whole_gdn=False):
        self.engine, self.whole_gdn = engine, whole_gdn
        self.values, self.linears, self.norms, self.gdn_layers = {}, {}, {}, {}
        self.layer = "unknown"
        self.linears[id(engine.w.lm_head)] = "lm_head"
        self.norms[id(engine.w.final_norm)] = "final_norm"
        for i, layer in enumerate(engine.w.layers):
            name = f"layer.{i}"
            self.norms[id(layer.ln1)], self.norms[id(layer.ln2)] = name + ".norm1", name + ".norm2"
            for part in ("gate_up", "down"):
                self.linears[id(getattr(layer, part))] = name + "." + part
            mixer = layer.mixer
            for part in ("in_qkvz", "out", "qkv", "o"):
                if hasattr(mixer, part):
                    self.linears[id(getattr(mixer, part))] = name + "." + part
            if hasattr(mixer, "in_ba"):
                self.gdn_layers[id(mixer.in_ba)] = name

    def save(self, name, value, row_axis=0):
        self.values.setdefault(name, []).append(value.detach().movedim(row_axis, 0).cpu().clone())

    def tensors(self):
        import torch
        return {name: torch.cat(parts, dim=0) for name, parts in self.values.items()}

    def __enter__(self):
        from tokenrush import fused, quant
        self.stack = ExitStack()

        def linear_wrapper(original, kind, output_axis):
            def wrapped(weight, x, *args, **kwargs):
                name = self.linears.get(id(weight))
                if name:
                    self.save(name + f".{kind}.input", x)
                result = original(weight, x, *args, **kwargs)
                if name:
                    self.save(name + f".{kind}.output", result, output_axis)
                return result
            return wrapped

        for method, kind, axis in (("__call__", "linear", 0), ("partials", "partials", 1)):
            self.stack.enter_context(patch.object(quant.QLinear, method,
                linear_wrapper(getattr(quant.QLinear, method), kind, axis)))

        original_norm = fused.add_rmsnorm
        def norm(x, h, weight, *args, **kwargs):
            name = self.norms[id(weight)]
            self.layer = name.rsplit(".", 1)[0]
            self.save(name + ".input", x)
            if h is not None:
                self.save(name + ".residual_input", h, 1 if h.ndim == 3 else 0)
            result = original_norm(x, h, weight, *args, **kwargs)
            self.save(name + ".residual_output", result[0])
            self.save(name + ".normalized", result[1])
            return result
        self.stack.enter_context(patch.object(fused, "add_rmsnorm", norm))

        original_gdn = fused.gdn_step_fused
        def gdn(qkvz, x, ba_w, *args, **kwargs):
            name = self.gdn_layers[id(ba_w)] + ".gdn"
            self.save(name + ".qkvz", qkvz)
            self.save(name + ".input", x)
            if self.whole_gdn:
                kwargs["BV"] = self.engine.cfg.gdn_v_dim
            result = original_gdn(qkvz, x, ba_w, *args, **kwargs)
            self.save(name + ".output", result)
            return result
        self.stack.enter_context(patch.object(fused, "gdn_step_fused", gdn))

        def fused_wrapper(original, kind):
            def wrapped(x, *args, **kwargs):
                name = self.layer + "." + kind
                self.save(name + ".input", x)
                result = original(x, *args, **kwargs)
                self.save(name + ".output", result)
                return result
            return wrapped
        for method in ("attn_prep", "attn_decode_fused", "silu_mul"):
            self.stack.enter_context(patch.object(fused, method,
                fused_wrapper(getattr(fused, method), method)))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def compare_traces(reference, actual, rows):
    import torch
    if list(reference) != list(actual):
        raise ValueError("operator traces differ in structure or order")
    differences = []
    for name, left in reference.items():
        right = actual[name]
        if left.shape != right.shape or not 1 <= rows <= left.shape[0]:
            raise ValueError(f"unaligned operator rows at {name}: {left.shape} vs {right.shape}")
        # Rejected tails must be present in the batch, but are not committed to
        # this saved run. Do not select a block solely for a rejected-tail difference.
        left, right = left[:rows], right[:rows]
        if not torch.equal(left, right):
            differences.append({"operation": name, "rows": [
                {"row": row, **tensor_difference(left[row], right[row])}
                for row in range(left.shape[0]) if not torch.equal(left[row], right[row])]})
    return {"compared_rows": rows, "observed_operations": len(reference), "different_operations": len(differences),
            "first_difference": differences[0] if differences else None, "differences": differences}


def traced_forward(engine, snapshot, tokens, sequential, whole_gdn=False):
    import torch
    snapshot.restore(engine.state)
    logits, hidden = [], []
    with OperationTrace(engine, whole_gdn) as trace:
        chunks = tokens.split(1) if sequential else (tokens,)
        for chunk in chunks:
            logits.append(engine.forward(chunk, all_logits=True).detach().cpu())
            hidden.append(engine.last_hidden.detach().cpu())
        values = trace.tensors()
    return values, torch.cat(logits), torch.cat(hidden)


def trace_blocks(adapter, diagnosis, saved, plan, report):
    import torch
    from tokenrush import quant
    engine, prompt = adapter.engine, saved["graph"]["prompt_token_ids"]
    adapter.mode = "eager"
    raw_first = adapter.prime(prompt)
    raw_hidden = engine.last_hidden.clone()
    snapshot = StateSnapshot(engine.state)
    adapter.mode = "mtp"
    mtp_first = adapter.prime(prompt)
    report["prefill"] = snapshot.compare(engine.state)
    report["prefill"].update(first_tokens=[raw_first, mtp_first],
                             hidden=tensor_difference(raw_hidden, engine.spec_hidden[:1]))
    prefill_equal = (raw_first == mtp_first and report["prefill"]["position_equal"]
                    and all(value["equal"] for value in report["prefill"].values() if isinstance(value, dict)))
    del snapshot, raw_hidden
    if not prefill_equal:
        report["finding"] = "Raw and MTP prefill already differ; investigate prefill before decode arithmetic."
        return
    shapes = {engine.w.lm_head.shape}
    for layer in engine.w.layers:
        shapes.update((layer.gate_up.shape, layer.down.shape))
        for name in ("in_qkvz", "out", "qkv", "o"):
            if hasattr(layer.mixer, name):
                shapes.add(getattr(layer.mixer, name).shape)
    report["scanned_blocks"] = []
    for step in range(plan["step"] + 1):
        ids = block_inputs(diagnosis, saved, step, adapter.k)
        tokens = torch.tensor(ids, device=engine.device, dtype=torch.long)
        start = diagnosis["steps"][step]["output_start"]
        adapter.mode = "eager"
        prefix = [adapter.prime(prompt)]
        for _ in range(start - 1):
            prefix.extend(adapter.step().tokens)
        if prefix != saved["graph"]["token_ids"][:start]:
            raise RuntimeError("raw committed prefix no longer matches the saved gate")
        snapshot = StateSnapshot(engine.state)
        print(f"Tracing saved block {step}, target position {snapshot.pos}, tokens {ids}...", flush=True)
        reference, ref_logits, ref_hidden = traced_forward(engine, snapshot, tokens, sequential=True)
        actual, logits, hidden = traced_forward(engine, snapshot, tokens, sequential=False)
        rows = diagnosis["steps"][step]["accepted"] + 1
        comparison = compare_traces(reference, actual, rows)
        comparison.update(logits=tensor_difference(ref_logits[:rows], logits[:rows]),
                          hidden=tensor_difference(ref_hidden[:rows], hidden[:rows]),
                          sequential_argmax=ref_logits[:rows].argmax(-1).tolist(),
                          batched_argmax=logits[:rows].argmax(-1).tolist())
        report["scanned_blocks"].append({"step": step, "position": snapshot.pos,
                                         "first_difference": comparison["first_difference"]})
        del actual
        if not comparison["different_operations"]:
            del reference, snapshot
            continue
        report["selected_block"] = {"step": step, "position": snapshot.pos, "tokens": ids,
                                    "saved_target_argmax": diagnosis["steps"][step]["target_argmax"],
                                    "state_source": "raw reference; shared by every variant",
                                    "compared_rows": rows}
        report["variants"] = {"original": comparison}
        for name, whole_gdn, common_rows in (("whole_gdn", True, False), ("common_rows", False, True),
                                            ("whole_gdn_and_common_rows", True, True)):
            print(f"Tracing controlled variant: {name}...", flush=True)
            with common_row_configs(quant, shapes, len(ids), common_rows) as configs:
                actual, logits, hidden = traced_forward(engine, snapshot, tokens, False, whole_gdn)
            result = compare_traces(reference, actual, rows)
            result.update(logits=tensor_difference(ref_logits[:rows], logits[:rows]),
                          hidden=tensor_difference(ref_hidden[:rows], hidden[:rows]),
                          batched_argmax=logits[:rows].argmax(-1).tolist(), changed_row_configs=configs,
                          gdn_bv=engine.cfg.gdn_v_dim if whole_gdn else "upstream default")
            report["variants"][name] = result
            del actual
        report["finding"] = "First differing operator and controlled variants recorded; not a correctness gate."
        return
    report["finding"] = "No observed operator-output difference in emitted rows from shared raw states through this saved block. " \
                        "This does not establish equality of speculative state contents after commits."


def run(args):
    directory = Path(args.diagnosis_dir).expanduser().resolve()
    diagnosis = json.loads((directory / "diagnosis.json").read_text())
    cfg, summary, saved = load_case(diagnosis["gate_dir"], diagnosis["prompt_id"])
    if cfg["backend"] != "triton" or cfg["kv"] != "bf16":
        raise ValueError("this arithmetic trace currently requires Triton and BF16 KV")
    if diagnosis["gate_configuration_sha256"] != summary["configuration_sha256"]:
        raise ValueError("diagnostic and gate configurations differ")
    plan = probe_plan(diagnosis, saved)
    validate_sources(cfg["sources"])
    validate_sources(json.loads((directory / "configuration.json").read_text())["diagnostic_sources"])
    if checkpoint_manifest(cfg["model"]["path"]) != cfg["model"]:
        raise ValueError("checkpoint record changed since the gate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = Path(args.output or ROOT / "runs" / f"trace-verify-{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "diagnosis_dir": str(directory), "plan": plan,
              "note": "Teacher-forced eager arithmetic trace from identical raw-prefix states, not a gate or benchmark. "
                      "The batched block uses saved drafts, including rejected tail tokens. No new model is downloaded."}
    print(f"Verify trace results: {output}", flush=True)
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
        torch.set_num_threads(1)  # CPU trace comparisons; this is not a timing run.
        report["cpu_analysis_threads"] = 1
        options = argparse.Namespace(stage="gate", model=cfg["model"]["path"], backend=cfg["backend"],
                                     kv=cfg["kv"], mtp_depth=cfg["mtp_depth"], tokens=cfg["tokens"],
                                     max_len=cfg["max_len"], chunk=cfg["chunk"])
        with torch.inference_mode():
            adapter = Adapter(options, "mtp", max(cfg["mtp_depth"], 7 if "dflash" in cfg["modes"] else 0))
            trace_blocks(adapter, diagnosis, saved, plan, report)
        report["status"] = "complete"
        console = {key: value for key, value in report.items() if key != "variants"}
        console["variants"] = {name: {key: value for key, value in result.items()
                                      if key not in ("differences", "changed_row_configs")}
                               for name, result in report.get("variants", {}).items()}
        print(json.dumps(console, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print(f"Trace complete; gate status unchanged. Full report: {output / 'trace.json'}", flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "trace.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", required=True, help="completed diagnose_mtp results; read-only")
    parser.add_argument("--output", help="new output directory")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
