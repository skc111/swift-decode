"""CPU-readable numerical policy shared by gate, benchmark, and GPU workers."""


def kernel_mode(args):
    return "consistent" if args.stage == "gate" else getattr(args, "bench_kernels", "normal")


def engine_options(args, draft_depth, gate_slots):
    consistent = kernel_mode(args) == "consistent"
    # Match the gate's recurrent-slot allocation as well as its kernel policy.
    # This also keeps the short-prefill fused-path threshold identical.
    return {"consistent": consistent, "max_spec": gate_slots if consistent else draft_depth}
