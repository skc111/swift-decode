from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import trace_verify as t


class BlockAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.saved = {"graph": {"prompt_token_ids": [1, 2, 3], "token_ids": [10, 20, 21, 50, 60]},
                      "mtp": {"token_ids": [10, 20, 22, 50, 60]}}
        self.diagnosis = {"steps": [{"step": 0, "output_start": 1, "position_before": 3,
                                    "committed_input": 10, "accepted": 1, "proposed": 3,
                                    "drafts": [20, 999, 998], "target_argmax": [20, 22]}]}

    def test_uses_actual_drafts_including_rejected_tail(self):
        self.assertEqual(t.block_inputs(self.diagnosis, self.saved, 0, 3), [10, 20, 999, 998])

    def test_rejects_inconsistent_position_committed_input_or_depth(self):
        for change in ({"position_before": 4}, {"committed_input": 99}, {"proposed": 2},
                       {"drafts": [20]}, {"step": 1}, {"accepted": 4},
                       {"output_start": 0}, {"output_start": 99}):
            diagnosis = deepcopy(self.diagnosis)
            diagnosis["steps"][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "committed prefix"):
                t.block_inputs(diagnosis, self.saved, 0, 3)

    def test_rejects_a_prefix_after_the_saved_divergence(self):
        self.diagnosis["steps"][0].update(output_start=3, position_before=5, committed_input=22)
        with self.assertRaisesRegex(ValueError, "committed prefix"):
            t.block_inputs(self.diagnosis, self.saved, 0, 3)

    def test_rejects_misaligned_accepted_rows_or_bonus(self):
        for change in ({"target_argmax": [20]}, {"target_argmax": [20, 99]},
                       {"drafts": [99, 999, 998]}):
            diagnosis = deepcopy(self.diagnosis)
            diagnosis["steps"][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "target rows"):
                t.block_inputs(diagnosis, self.saved, 0, 3)


class TargetConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.single = (248320, 5120, 1, "torch.bfloat16")
        self.multi = (248320, 5120, 4, "torch.bfloat16")
        self.other_single = (16384, 5120, 1, "torch.bfloat16")
        self.other_multi = (16384, 5120, 4, "torch.bfloat16")
        self.draft = (131072, 5120, 4, "torch.bfloat16")
        self.cache = {self.single: "single", self.multi: "multi",
                      self.other_single: "other-single", self.other_multi: "other-multi", self.draft: "draft"}
        self.quant = SimpleNamespace(_int4_gemm_rows_kernel=SimpleNamespace(cache=self.cache))
        self.shapes = {self.single[:2], self.other_single[:2]}

    def test_matches_all_target_shapes_and_restores_after_failure(self):
        before = dict(self.cache)
        with self.assertRaisesRegex(RuntimeError, "failed probe"):
            with t.common_row_configs(self.quant, self.shapes, 4, True) as changes:
                self.assertEqual(len(changes), 2)
                self.assertEqual(self.cache[self.multi], before[self.single])
                self.assertEqual(self.cache[self.other_multi], before[self.other_single])
                self.assertEqual(self.cache[self.draft], before[self.draft])
                raise RuntimeError("failed probe")
        self.assertEqual(self.cache, before)

    def test_missing_singleton_or_batched_config_is_atomic(self):
        for missing in (self.other_single, self.other_multi):
            with self.subTest(missing=missing):
                original = self.cache.pop(missing)
                before = dict(self.cache)
                with self.assertRaises(ValueError):
                    with t.common_row_configs(self.quant, self.shapes, 4, True):
                        self.fail("must validate every shape before changing any config")
                self.assertEqual(self.cache, before)
                self.cache[missing] = original

    def test_disabled_control_leaves_even_incomplete_cache_untouched(self):
        del self.cache[self.other_single]
        before = dict(self.cache)
        with t.common_row_configs(self.quant, self.shapes, 4, False) as changes:
            self.assertEqual(changes, [])
            self.assertEqual(self.cache, before)
        self.assertEqual(self.cache, before)


class TraceHookTests(unittest.TestCase):
    def test_hooks_preserve_results_and_restore_after_exception(self):
        # Exercise the wrapper protocol without importing Torch or GPU modules.
        class Weight:
            def __call__(self, x):
                return x

            def partials(self, x):
                return x

        passthrough = lambda x, *args, **kwargs: x
        fused = SimpleNamespace(add_rmsnorm=lambda x, *args, **kwargs: (x, x),
                                gdn_step_fused=passthrough, attn_prep=passthrough,
                                attn_decode_fused=passthrough, silu_mul=passthrough)
        quant = SimpleNamespace(QLinear=Weight)
        engine = SimpleNamespace(w=SimpleNamespace(lm_head=Weight(), final_norm=object(), layers=[]))
        originals = dict(vars(fused))
        original_call, original_partials = Weight.__call__, Weight.partials
        observed = []
        trace = t.OperationTrace(engine)
        trace.save = lambda name, value, row_axis=0: observed.append((name, value, row_axis))
        x = object()
        with patch.dict("sys.modules", {"tokenrush": SimpleNamespace(quant=quant, fused=fused)}):
            with self.assertRaisesRegex(RuntimeError, "failed trace"):
                with trace:
                    self.assertIs(engine.w.lm_head(x), x)
                    self.assertIs(engine.w.lm_head.partials(x), x)
                    self.assertEqual(fused.add_rmsnorm(x, None, engine.w.final_norm), (x, x))
                    raise RuntimeError("failed trace")
        self.assertIn(("lm_head.partials.output", x, 1), observed)
        self.assertIs(Weight.__call__, original_call)
        self.assertIs(Weight.partials, original_partials)
        for name, original in originals.items():
            self.assertIs(getattr(fused, name), original)


class TraceAlignmentTests(unittest.TestCase):
    class Rows(tuple):
        @property
        def shape(self):
            return (len(self), len(self[0]))

        def __getitem__(self, key):
            result = super().__getitem__(key)
            return type(self)(result) if isinstance(key, slice) else result

    def setUp(self):
        self.reference = {"input": self.Rows(((1,), (2,), (3,))),
                          "output": self.Rows(((10,), (20,), (30,)))}
        # Only comparisons and row selection are under test; no simulated GPU arithmetic.
        self.torch = SimpleNamespace(equal=lambda left, right: left == right)

    def test_skips_rejected_tail_and_reports_emitted_row(self):
        actual = {**self.reference, "output": self.Rows(((10,), (20,), (999,)))}
        with patch.dict("sys.modules", {"torch": self.torch}), \
                patch.object(t, "tensor_difference", return_value={"equal": False}):
            self.assertIsNone(t.compare_traces(self.reference, actual, 2)["first_difference"])
            actual["output"] = self.Rows(((10,), (999,), (999,)))
            result = t.compare_traces(self.reference, actual, 2)
        self.assertEqual(result["different_operations"], 1)
        self.assertEqual(result["first_difference"], {"operation": "output", "rows": [{"row": 1, "equal": False}]})

    def test_refuses_changed_operator_order_shapes_or_invalid_row_limit(self):
        cases = [(dict(reversed(list(self.reference.items()))), 2),
                 ({**self.reference, "output": self.Rows(((10,), (20,)))}, 2),
                 (self.reference, 0), (self.reference, 4)]
        with patch.dict("sys.modules", {"torch": self.torch}):
            for actual, rows in cases:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    t.compare_traces(self.reference, actual, rows)


if __name__ == "__main__":
    unittest.main()
