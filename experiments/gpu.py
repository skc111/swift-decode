"""Adapter to the upstream engine. Imported only by a GPU worker process."""
import os

from .measurement import Step
from .numerics import engine_options


class Adapter:
    def __init__(self, args, mode, gate_slots):
        # Set this BEFORE importing model/quant: upstream otherwise probes and
        # compiles Marlin even for a requested Triton run.
        os.environ["TOKENRUSH_BACKEND"] = args.backend
        import torch
        from tokenrush.model import Engine
        from tokenrush.weights import load_packed
        from tokenrush.sample import sample
        from transformers import AutoTokenizer

        self.torch, self.sample, self.args, self.mode = torch, sample, args, mode
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        if args.backend == "marlin":
            from tokenrush.marlin import ext
            ext()  # Fail loudly; do not silently fall back to Triton.
        cfg, weights, mtp_tensors = load_packed(args.model, backend=args.backend, with_mtp=mode == "mtp")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        self.k = args.mtp_depth if mode == "mtp" else 7 if mode == "dflash" else 0
        self.execution_options = engine_options(args, self.k, gate_slots)
        self.engine = Engine(cfg, weights, max_len=args.max_len, **self.execution_options,
                             kv_dtype=torch.bfloat16 if args.kv == "bf16" else torch.float8_e4m3fn)
        self.engine.sampling.set(temperature=0.0)
        self.stop_ids = {x for x in (*cfg.eos_ids, self.tokenizer.eos_token_id) if x is not None}
        self.draft = None
        if mode != "eager":
            self.engine.capture()
        if mode == "mtp":
            from tokenrush.mtp import MTPHead, build_mtp
            self.draft = MTPHead(cfg, build_mtp(cfg, mtp_tensors, "cuda", int4=True), weights.embed,
                                 weights.lm_head, args.max_len, kv_dtype=self.engine.state.kv_dtype)
            self.engine.attach_mtp(self.draft, draft_vocab=torch.arange(min(131072, cfg.vocab)))
            self.engine.capture_spec(self.k)
        elif mode == "dflash":
            from tokenrush.dflash import DFlashDraft, load_dflash
            self.draft = DFlashDraft(load_dflash(args.draft_model, int4=True), weights.embed,
                                     weights.lm_head, cfg.hidden, args.max_len)
            if self.draft.w.block_size - 1 != self.k:
                raise ValueError("this experiment expects the DFlash2 checkpoint with block_size=8")
            self.engine.attach_dflash(self.draft, draft_vocab=torch.arange(min(131072, cfg.vocab)))
            self.engine.capture_spec_dflash()

    def encode(self, text):
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if not ids:
            raise ValueError("empty tokenized prompt")
        # Reserve scratch positions needed by the final speculative verify step.
        if len(ids) + self.args.tokens + self.k > self.args.max_len:
            raise ValueError("prompt + output + speculative scratch exceeds --max-len")
        return ids

    def prime(self, ids):
        from tokenrush.spec import prime_dflash, prime_spec
        torch, engine = self.torch, self.engine
        torch.manual_seed(0)
        if self.mode == "mtp":
            first, _ = prime_spec(engine, self.draft, ids, chunk=self.args.chunk)
        elif self.mode == "dflash":
            first, _ = prime_dflash(engine, self.draft, ids, chunk=self.args.chunk)
        else:
            engine.reset()
            tokens = torch.tensor(ids, device=engine.device, dtype=torch.long)
            logits = engine.prefill(tokens, chunk=self.args.chunk)
            first = self.sample(logits[-1:], engine.sampling)[0]
            engine.tok.copy_(first.view(1))
        self.last = first.view(1)
        return int(first)

    def step(self):
        engine = self.engine
        if self.mode == "eager":
            logits = engine.decode(self.last)
            self.last = self.sample(logits[-1:], engine.sampling)
            return Step([int(self.last[0])])
        if self.mode == "graph":
            return Step([int(engine.step()[0])])
        n = engine.spec_step(self.k) if self.mode == "mtp" else engine.spec_step_dflash()
        # The input/committed token was already emitted. These are its successors:
        # accepted drafts, then the target's mismatch/bonus token (already valid).
        return Step(engine.drafts[:n].tolist() + [int(engine.tok[0])], accepted=n, proposed=self.k)
