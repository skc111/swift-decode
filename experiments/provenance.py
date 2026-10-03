"""Small, read-only environment and checkpoint records. No network requests."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"returncode": None, "error": str(e)}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_fingerprint():
    paths = [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in ("tokenrush", "experiments"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h", ".cuh", ".json")
                     and "build" not in p.parts)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def snapshot_sources(destination, fingerprints):
    """Keep uncommitted/untracked experiment code too, not just git diff."""
    for name, expected in fingerprints.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError("source snapshot must stay within the repository")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"source changed while preparing the run: {name}")
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def checkpoint_manifest(path, packed=True):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"local checkpoint directory not found: {path}; no automatic downloads")
    for name in (("config.json", "tokenrush.json") if packed else ("config.json",)):
        if not (path / name).is_file():
            raise ValueError(f"checkpoint missing {name}: {path}")
    weights = sorted(path.glob("model-*.safetensors" if packed else "*.safetensors"))
    if not weights:
        raise ValueError(f"checkpoint has no expected safetensors weights: {path}")
    if any(p.stat().st_size == 0 for p in weights):
        raise ValueError("checkpoint contains an empty weight file")
    if packed:
        meta = json.loads((path / "tokenrush.json").read_text())
        if "shards" in meta and len(weights) != meta["shards"]:
            raise ValueError("packed checkpoint shard count does not match tokenrush.json")
    metadata = {p.name: file_hash(p) for p in sorted(path.iterdir())
                if p.is_file() and p.suffix in (".json", ".jinja", ".txt")}
    return {"path": str(path), "metadata_sha256": metadata,
            "weight_files": [{"name": p.name, "bytes": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns} for p in weights],
            "note": "Weight bytes are NOT hashed here. Size/mtime is not a content checksum; keep the pinned download revision."}


def environment():
    packages = {}
    for name in ("torch", "triton", "transformers", "flash-linear-attention", "nvidia-nccl-cu13", "cuda-bindings"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        fla_source = json.loads(importlib.metadata.distribution("flash-linear-attention")
                                .read_text("direct_url.json") or "null")
    except importlib.metadata.PackageNotFoundError:
        fla_source = None
    limits = {}
    for name in ("memory.max", "cpu.max", "cpuset.cpus.effective"):
        p = Path("/sys/fs/cgroup") / name
        if p.is_file():
            limits[name] = p.read_text().strip()
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "cgroup_v2_visible_limits": limits,
            "fla_source": fla_source,
            "runtime_env": {k: os.environ.get(k) for k in
                            ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST")},
            "git_commit": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--porcelain"]),
            "nvcc": command(["nvcc", "--version"]),
            "nvidia_smi": command(["nvidia-smi"]),
            "gpu_driver_query": command(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"])}


def runtime_fingerprint(report):
    """Exclude changing timestamps/utilization; bind the gate to this runtime."""
    return object_hash({k: report.get(k) for k in
                       ("python", "platform", "packages", "fla_source", "runtime_env", "gpu",
                        "gpu_driver_query", "nvcc")})


def check_gpu_environment():
    import tomllib
    report = environment()
    errors = []
    locked = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    expected = {name: p["version"] for name, p in locked.items()}
    for name, installed in report["packages"].items():
        if installed != expected.get(name):
            errors.append(f"{name}: installed={installed}, locked={expected.get(name)}")
    fla_commit = (report["fla_source"] or {}).get("vcs_info", {}).get("commit_id")
    expected_commit = locked["flash-linear-attention"]["source"]["git"].rsplit("#", 1)[-1]
    if fla_commit != expected_commit:
        errors.append(f"flash-linear-attention commit: installed={fla_commit}, locked={expected_commit}")
    if report["gpu_driver_query"]["returncode"] != 0:
        errors.append("cannot record GPU UUID / driver via nvidia-smi")
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        errors.append("requires Linux and Python 3.12")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(0)
        capability = torch.cuda.get_device_capability(0)
        report["gpu"] = {"name": torch.cuda.get_device_name(0), "capability": list(capability),
                         "torch_cuda": torch.version.cuda, "visible_devices": torch.cuda.device_count(),
                         "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable"))}
        if capability != (12, 0):
            errors.append(f"this reproduction targets sm_120, got {capability}")
        x = torch.ones((16, 16), device="cuda")
        report["cuda_matmul_pass"] = bool(((x @ x) == 16).all().item())
        torch.cuda.synchronize()
        if not report["cuda_matmul_pass"]:
            errors.append("CUDA matmul smoke test failed")
    except Exception as e:
        errors.append(f"CUDA smoke test: {type(e).__name__}: {e}")
    report["errors"] = errors
    report["passed"] = not errors
    report["note"] = "This only checks package versions and a CUDA matmul; it is NOT the model/kernel correctness gate."
    return report
