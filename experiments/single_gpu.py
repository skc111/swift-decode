"""One-card, one-request-at-a-time reproduction. --plan works without torch.

Run gate first, then bench with --gate-dir pointing at a matching successful gate.
No model downloads, package installs, or GPU power/clock changes are performed.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .measurement import compare_outputs, measure, summarize
from .provenance import (ROOT, check_gpu_environment, checkpoint_manifest, command,
                         environment, object_hash, runtime_fingerprint,
                         snapshot_sources, source_fingerprint)

MODES = ("eager", "graph", "mtp", "dflash")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_prompts(path):
    prompts = json.loads(Path(path).read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty JSON list")
    seen = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or set(prompt) != {"id", "text"}:
            raise ValueError("each prompt must contain exactly id and text")
        if not all(isinstance(prompt[k], str) and prompt[k].strip() for k in ("id", "text")):
            raise ValueError("prompt id/text must be nonempty strings")
        if prompt["id"] in seen:
            raise ValueError(f"duplicate prompt id: {prompt['id']}")
        seen.add(prompt["id"])
    return prompts


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-env", action="store_true", help="check pinned packages and a CUDA matmul, then exit")
    p.add_argument("--plan", action="store_true", help="print the plan without CUDA, downloads or output files")
    p.add_argument("--stage", choices=("gate", "bench"), default="gate")
    p.add_argument("--model", help="local packed checkpoint directory; never a Hub repo id")
    p.add_argument("--draft-model", help="local DFlash2 directory, required only for dflash mode")
    p.add_argument("--modes", default="eager,graph,mtp")
    p.add_argument("--backend", choices=("triton", "marlin"), default="triton")
    p.add_argument("--kv", choices=("bf16", "fp8"), default="bf16")
    p.add_argument("--mtp-depth", type=int, default=3, help="fixed draft depth, not upstream adaptive 3:4")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompts", default=str(ROOT / "experiments/prompts.json"))
    p.add_argument("--respect-eos", action="store_true", help="default: fixed output length, ignore EOS for workload control")
    p.add_argument("--gate-dir", help="required for bench: a successful gate of the same code/model/workload")
    p.add_argument("--output", help="new output directory; existing directories are never overwritten")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per mode process")
    p.add_argument("--worker-mode", choices=MODES, help=argparse.SUPPRESS)
    return p


def validate(args):
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError(f"--modes must be unique names from {MODES}")
    if "graph" not in modes or len(modes) < 2:
        raise ValueError("include graph and at least one other mode for a meaningful comparison")
    if args.worker_mode and args.worker_mode not in modes:
        raise ValueError("worker mode is not in this experiment's modes")
    if not args.model:
        raise ValueError("--model is required")
    if "dflash" in modes and not args.draft_model:
        raise ValueError("--draft-model is required for dflash")
    if not 1 <= args.mtp_depth <= 7:
        raise ValueError("--mtp-depth must be between 1 and 7")
    if any(getattr(args, k) < 1 for k in ("tokens", "max_len", "chunk", "repeats", "timeout")):
        raise ValueError("tokens, max-len, chunk, repeats and timeout must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.stage == "bench" and not args.worker_mode and not args.gate_dir and not args.plan:
        raise ValueError("bench requires --gate-dir; run the correctness gate first")
    return modes


def configuration(args, modes, prompts):
    return {"protocol": 1, "sources": source_fingerprint(),
            "model": checkpoint_manifest(args.model),
            "draft": checkpoint_manifest(args.draft_model, packed=False) if "dflash" in modes else None,
            "modes": sorted(modes), "prompts": prompts, "backend": args.backend, "kv": args.kv,
            "mtp_depth": args.mtp_depth, "tokens": args.tokens, "max_len": args.max_len,
            "chunk": args.chunk, "respect_eos": args.respect_eos,
            "temperature": 0.0, "seed": 0, "draft_vocab_limit": 131072}


def check_gate(path, signature, runtime=None):
    gate = json.loads((Path(path) / "summary.json").read_text())
    if gate.get("stage") != "gate" or gate.get("status") != "passed":
        raise ValueError("the supplied gate did not pass")
    if gate.get("configuration_sha256") != signature:
        raise ValueError("gate configuration differs: code, checkpoint, prompts or generation parameters changed")
    if runtime is not None and gate.get("runtime_sha256") != runtime:
        raise ValueError("gate environment differs: rerun the gate on this GPU / driver / Python stack")


def child_command(args, mode, output):
    cmd = [sys.executable, "-u", "-m", "experiments.single_gpu", "--worker-mode", mode,
           "--stage", args.stage, "--model", str(Path(args.model).expanduser().resolve()),
           "--modes", args.modes, "--backend", args.backend, "--kv", args.kv,
           "--mtp-depth", str(args.mtp_depth), "--tokens", str(args.tokens),
           "--max-len", str(args.max_len), "--chunk", str(args.chunk),
           "--warmup", str(args.warmup), "--repeats", str(args.repeats),
           "--prompts", str(Path(args.prompts).expanduser().resolve()), "--output", str(output)]
    if args.draft_model:
        cmd += ["--draft-model", str(Path(args.draft_model).expanduser().resolve())]
    if args.respect_eos:
        cmd.append("--respect-eos")
    return cmd


def worker(args, modes, prompts):
    from .gpu import Adapter
    import torch
    slots = max(args.mtp_depth if "mtp" in modes else 0, 7 if "dflash" in modes else 0)
    with torch.inference_mode():
        adapter = Adapter(args, args.worker_mode, slots)
        prepared = [(p["id"], adapter.encode(p["text"])) for p in prompts]
        # Each process loads one mode so engines/graphs cannot survive into the
        # next measurement and make the peak-memory comparison misleading.
        output = Path(args.output)
        write_json(output / f"{args.worker_mode}.environment.json", environment())
        with (output / f"{args.worker_mode}.jsonl").open("x") as f:
            for phase, count in (("warmup", args.warmup), ("measure", args.repeats)):
                for repeat in range(count):
                    for name, ids in prepared:
                        torch.cuda.reset_peak_memory_stats()
                        result = measure(lambda: adapter.prime(ids), adapter.step, torch.cuda.synchronize,
                                         args.tokens, adapter.stop_ids if args.respect_eos else ())
                        result.update(mode=args.worker_mode, prompt_id=name, prompt_token_ids=ids,
                                      phase=phase, round=repeat)
                        result["stats"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                        result["text"] = adapter.tokenizer.decode(result["token_ids"])
                        f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        f.flush()
                        print(f"{args.worker_mode} {phase} {repeat + 1} {name}: {result['stats']}", flush=True)


def run_child(cmd, log_path, timeout):
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def suite(args, modes, prompts):
    cfg = configuration(args, modes, prompts)
    signature = object_hash(cfg)
    if args.stage == "bench":
        check_gate(args.gate_dir, signature)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = Path(args.output or ROOT / "runs" / f"{args.stage}-{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "configuration.json", cfg)
    manifest = {"stage": args.stage, "status": "running", "started_utc": stamp,
                "configuration_sha256": signature, "options": vars(args), "jobs": [],
                "gate_dir": str(Path(args.gate_dir).resolve()) if args.gate_dir else None}
    write_json(output / "suite.json", manifest)
    print(f"Results: {output}", flush=True)
    try:
        snapshot_sources(output / "sources", cfg["sources"])
        diff = command(["git", "diff", "HEAD"])
        write_json(output / "git_diff.json", diff)
        (output / "tracked_changes.patch").write_text(diff.get("stdout", ""))
        env = check_gpu_environment()
        write_json(output / "environment.json", env)
        if not env["passed"]:
            raise RuntimeError("environment check failed: " + "; ".join(env["errors"]))
        runtime = runtime_fingerprint(env)
        manifest["runtime_sha256"] = runtime
        if args.stage == "bench":
            check_gate(args.gate_dir, signature, runtime)
        rows = []
        for mode in modes:
            cmd = child_command(args, mode, output)
            job = {"mode": mode, "command": cmd, "status": "running"}
            manifest["jobs"].append(job)
            write_json(output / "suite.json", manifest)
            print(f"=== {mode} === (log: {output / (mode + '.log')})", flush=True)
            started = time.monotonic()
            try:
                rc = run_child(cmd, output / f"{mode}.log", args.timeout)
            except BaseException:
                job["status"] = "interrupted"
                raise
            job.update(returncode=rc, elapsed_s=time.monotonic() - started,
                       status="complete" if rc == 0 else "failed")
            write_json(output / "suite.json", manifest)
            if rc != 0:
                raise RuntimeError(f"{mode} failed with exit {rc}; see {mode}.log")
            rows.extend(json.loads(line) for line in (output / f"{mode}.jsonl").read_text().splitlines())
        comparison = compare_outputs(rows, modes, [p["id"] for p in prompts], args.repeats)
        passed = comparison["all_equal"]
        status = ("passed" if passed else "failed") if args.stage == "gate" else "complete"
        summary = {"stage": args.stage, "status": status, "configuration_sha256": signature,
                   "runtime_sha256": runtime,
                   "output_comparison": comparison,
                   "performance": summarize(rows) if args.stage == "bench" else [],
                   "note": "Single-stream engine measurements, not serving throughput or population percentiles. "
                           "Gate uses consistent=True and is not a speed measurement or an independent HF oracle. "
                           "Bench uses normal kernels; any output differences are recorded, not hidden."}
        write_json(output / "summary.json", summary)
        if args.stage == "bench":
            with (output / "comparison.csv").open("w", newline="") as f:
                fields = ("mode", "prompt_id", "measured_rounds", "decode_output_tok_s_median",
                          "first_token_ms_median", "tpot_ms_median", "peak_allocated_GiB_median")
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in summary["performance"]:
                    median = lambda key, scale=1: r[key]["median"] * scale if r[key] else None
                    writer.writerow(dict(zip(fields, (r["mode"], r["prompt_id"], r["measured_rounds"],
                        median("decode_output_tok_s"), median("first_token_s", 1000), median("tpot_s", 1000),
                        median("peak_allocated_bytes", 1 / 2**30)))))
        manifest["status"] = status
        if args.stage == "gate" and not passed:
            print("FAIL: greedy outputs differ; see summary.json. Benchmark is not authorized by this gate.", flush=True)
        elif args.stage == "gate":
            print("PASS: greedy outputs match across modes and measured rounds.", flush=True)
        elif args.stage == "bench" and not passed:
            print("WARNING: normal-kernel outputs differ; inspect output_comparison before making claims.", flush=True)
        return 1 if status == "failed" else 0
    except BaseException as e:
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "suite.json", manifest)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.check_env:
            report = check_gpu_environment()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["passed"] else 1
        modes = validate(args)
        prompts = read_prompts(args.prompts)
        if args.plan:
            print(json.dumps({"stage": args.stage, "modes": modes, "backend": args.backend,
                              "prompts": [p["id"] for p in prompts], "tokens_per_request": args.tokens,
                              "warmup": args.warmup, "repeats": args.repeats,
                              "measured_requests": len(modes) * len(prompts) * args.repeats,
                              "downloads": False, "gpu_used": False,
                              "requires_passing_gate": args.stage == "bench"}, indent=2))
            return 0
        if args.worker_mode:
            worker(args, modes, prompts)
            return 0
        return suite(args, modes, prompts)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
