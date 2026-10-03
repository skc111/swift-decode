from copy import deepcopy
from types import SimpleNamespace
import unittest

from experiments import probe_mtp_head as p


def fake_quant():
    single = (248320, 5120, 1, "torch.bfloat16", "torch.bfloat16")
    multi = (*single[:2], 4, *single[3:])
    other = (131072, 5120, 1, "torch.bfloat16", "torch.bfloat16")
    return SimpleNamespace(_ROWS_CONFIGS=["config-256", "config-512"],
                           _int4_gemm_rows_kernel=SimpleNamespace(cache={
                               single: "config-256", multi: "config-512", other: "config-512"})), single, multi, other


class AutotuneInterventionTests(unittest.TestCase):
    def test_restores_known_configs_without_evaluating_code(self):
        quant, single, _, _ = fake_quant()
        self.assertEqual(p.restore_rows_cache(quant, [{"key": repr(single), "config": "config-512"}]), 1)
        self.assertEqual(quant._int4_gemm_rows_kernel.cache[single], "config-512")
        with self.assertRaises((ValueError, SyntaxError)):
            p.restore_rows_cache(quant, [{"key": "__import__('os').getcwd()", "config": "config-512"}])

    def test_unknown_config_leaves_cache_intact(self):
        quant, single, multi, _ = fake_quant()
        before = dict(quant._int4_gemm_rows_kernel.cache)
        with self.assertRaisesRegex(ValueError, "not available"):
            p.restore_rows_cache(quant, [{"key": repr(single), "config": "config-512"},
                                         {"key": repr(multi), "config": "unknown"}])
        self.assertEqual(quant._int4_gemm_rows_kernel.cache, before)

    def test_empty_and_duplicate_records_are_rejected(self):
        quant, single, _, _ = fake_quant()
        entry = {"key": repr(single), "config": "config-256"}
        for entries in ([], [entry, entry]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                p.restore_rows_cache(quant, entries)

    def test_changes_only_batched_head_and_restores_after_exception(self):
        quant, single, multi, other = fake_quant()
        before = dict(quant._int4_gemm_rows_kernel.cache)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with p.shared_head_config(quant, 248320, 5120, 4) as info:
                self.assertEqual(info["batched_original"], "config-512")
                self.assertEqual(quant._int4_gemm_rows_kernel.cache[multi], before[single])
                self.assertEqual(quant._int4_gemm_rows_kernel.cache[other], before[other])
                raise RuntimeError("probe failure")
        self.assertEqual(quant._int4_gemm_rows_kernel.cache, before)

    def test_missing_batch_config_fails_without_changing_singleton(self):
        quant, _, multi, _ = fake_quant()
        del quant._int4_gemm_rows_kernel.cache[multi]
        before = dict(quant._int4_gemm_rows_kernel.cache)
        with self.assertRaisesRegex(ValueError, "missing"):
            with p.shared_head_config(quant, 248320, 5120, 4):
                self.fail("must not reach the probe")
        self.assertEqual(quant._int4_gemm_rows_kernel.cache, before)


class ProbeAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.saved = {"graph": {"token_ids": [10, 20, 21, 50]}, "mtp": {"token_ids": [10, 20, 22, 50]}}
        step = {"step": 0, "output_start": 1, "committed_input": 10, "accepted": 1}
        self.diagnosis = {"status": "complete", "graph_token_ids": [10, 20, 21, 50],
                          "mtp_token_ids": [10, 20, 22, 50], "steps": [step],
                          "first_difference": {"output_index_zero_based": 2, "step": {**step, "verify_row": 1}}}

    def test_uses_the_verify_row_that_actually_changed_the_target_token(self):
        self.assertEqual(p.probe_plan(self.diagnosis, self.saved), {"index": 2, "step": 0, "row": 1})

    def test_refuses_nonreproducing_diagnostic(self):
        self.diagnosis["mtp_token_ids"][2] = 99
        with self.assertRaisesRegex(ValueError, "did not reproduce"):
            p.probe_plan(self.diagnosis, self.saved)

    def test_refuses_misaligned_committed_token_or_verify_row(self):
        for change in ({"committed_input": 99}, {"verify_row": 0}, {"accepted": 0}):
            diagnosis = deepcopy(self.diagnosis)
            diagnosis["first_difference"]["step"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "does not align"):
                p.probe_plan(diagnosis, self.saved)


if __name__ == "__main__":
    unittest.main()
