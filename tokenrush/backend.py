"""Backend selection without importing torch or building a CUDA extension."""
import os

BACKENDS = ("marlin", "triton", "tinygemm", "dequant")


def select_backend(probe_marlin):
    requested = os.environ.get("TOKENRUSH_BACKEND", "auto")
    if requested == "auto":
        return "marlin" if probe_marlin() else "triton"
    if requested not in BACKENDS:
        raise ValueError(f"TOKENRUSH_BACKEND={requested!r}; expected auto or one of {BACKENDS}")
    # Do not replace an explicit choice with the auto-probe's fallback.
    return requested
