import copy
import unittest
from unittest.mock import Mock

from experiments.measurement import Step, compare_outputs, first_difference, measure, summarize


def run_steps(steps, length, stops=(), first=1):
    prime = Mock(return_value=first)
    step = Mock(side_effect=steps)
    sync = Mock()
    clock = Mock(side_effect=range(len(steps) + 2))
    result = measure(prime, step, sync, length, stops, clock)
    return result, prime, step, sync


def row(mode="graph", prompt="p", repeat=0, tokens=None, phase="measure"):
    return {"mode": mode, "prompt_id": prompt, "round": repeat, "phase": phase,
            "prompt_token_ids": [10, 11], "token_ids": [1, 2, 3] if tokens is None else tokens,
            "stats": {"decode_output_tok_s": 100, "first_token_s": 0.1,
                      "output_tok_s": 30, "tpot_s": 0.01,
                      "acceptance_rate": None, "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}}


class MeasurementTests(unittest.TestCase):
    def test_raw_does_not_execute_a_step_after_the_last_token(self):
        r, prime, step, sync = run_steps([Step([2]), Step([3]), Step([4])], 4)
        self.assertEqual(r["token_ids"], [1, 2, 3, 4])
        self.assertEqual(prime.call_count, 1)
        self.assertEqual(step.call_count, 3)
        self.assertEqual(sync.call_count, 5)
        self.assertEqual(r["stats"]["new_tokens"], 4)
        self.assertEqual(r["stats"]["decode_output_tok_s"], 1)
        self.assertEqual(r["stats"]["decode_s"], 3)
        self.assertEqual(r["stats"]["first_token_s"], 1)

    def test_one_token_has_no_decode_rate(self):
        r, _, step, _ = run_steps([], 1)
        step.assert_not_called()
        self.assertIsNone(r["stats"]["decode_output_tok_s"])
        self.assertIsNone(r["stats"]["tpot_s"])
        self.assertEqual(r["stats"]["decode_s"], 0)

    def test_speculative_tail_is_counted_but_not_emitted(self):
        r, _, step, _ = run_steps([Step([2, 3, 4, 5], accepted=3, proposed=3)], 3)
        self.assertEqual(r["token_ids"], [1, 2, 3])
        self.assertEqual(step.call_count, 1)
        self.assertEqual(r["stats"]["discarded_tail_tokens"], 2)
        self.assertEqual(r["stats"]["decode_output_tok_s"], 2)
        self.assertEqual(r["stats"]["acceptance_rate"], 1)
        self.assertEqual(r["step_events"][0]["computed"], 4)
        self.assertEqual(r["step_events"][0]["emitted"], 2)

    def test_rejected_drafts_still_make_progress(self):
        r, _, _, _ = run_steps([Step([2], 0, 3), Step([3], 0, 3)], 3)
        self.assertEqual(r["token_ids"], [1, 2, 3])
        self.assertEqual(r["stats"]["acceptance_rate"], 0)
        self.assertEqual(r["stats"]["proposed_drafts"], 6)

    def test_first_token_eos_does_not_decode(self):
        r, _, step, _ = run_steps([], 10, stops=[1])
        step.assert_not_called()
        self.assertEqual(r["stats"]["stop_reason"], "eos")

    def test_eos_in_a_speculative_batch_cuts_the_tail(self):
        r, _, step, _ = run_steps([Step([2, 99, 4, 5], 3, 3)], 20, stops=[99])
        self.assertEqual(r["token_ids"], [1, 2, 99])
        self.assertEqual(r["stats"]["stop_reason"], "eos")
        self.assertEqual(r["stats"]["discarded_tail_tokens"], 2)
        step.assert_called_once()

    def test_default_ignores_eos_for_fixed_work(self):
        r, _, _, _ = run_steps([Step([99]), Step([2])], 3)
        self.assertEqual(r["token_ids"], [1, 99, 2])
        self.assertEqual(r["stats"]["stop_reason"], "length")

    def test_invalid_output_lengths(self):
        for length in (0, -1, True, 1.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                run_steps([], length)

    def test_empty_step_fails_instead_of_hanging(self):
        with self.assertRaisesRegex(RuntimeError, "no tokens"):
            run_steps([Step([])], 2)

    def test_invalid_acceptance_counts(self):
        for accepted, proposed in ((-1, 3), (4, 3), (0, -1)):
            with self.subTest(accepted=accepted, proposed=proposed), self.assertRaises(ValueError):
                run_steps([Step([2], accepted, proposed)], 2)


class ComparisonTests(unittest.TestCase):
    def test_first_difference_including_length_mismatch(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1, 2], [1]), 1)
        self.assertEqual(first_difference([], [1]), 0)

    def test_complete_equal_outputs(self):
        rows = [row(mode, repeat=r) for mode in ("eager", "graph", "mtp") for r in range(2)]
        result = compare_outputs(rows, ["eager", "graph", "mtp"], ["p"], 2)
        self.assertTrue(result["all_equal"])
        self.assertEqual(result["reference"], "graph")
        self.assertEqual(len(result["checks"]), 6)

    def test_mismatch_in_a_later_repeat_is_not_hidden(self):
        rows = [row("graph"), row("graph", repeat=1, tokens=[1, 7, 3])]
        result = compare_outputs(rows, ["graph"], ["p"], 2)
        self.assertFalse(result["all_equal"])
        self.assertEqual(result["checks"][1]["first_difference"], 1)

    def test_different_input_fails_even_if_output_matches(self):
        rows = [row("graph"), row("mtp")]
        rows[1]["prompt_token_ids"] = [42]
        result = compare_outputs(rows, ["graph", "mtp"], ["p"], 1)
        self.assertFalse(result["all_equal"])
        self.assertFalse(result["checks"][1]["same_input"])

    def test_equal_benchmark_rows_can_still_differ_from_saved_gate(self):
        rows = [row("graph", tokens=[1, 9, 3]), row("mtp", tokens=[1, 9, 3])]
        self.assertTrue(compare_outputs(rows, ["graph", "mtp"], ["p"], 1)["all_equal"])
        result = compare_outputs(rows, ["graph", "mtp"], ["p"], 1, references={"p": row()})
        self.assertFalse(result["all_equal"])
        self.assertTrue(all(c["first_difference"] == 1 for c in result["checks"]))

    def test_saved_reference_checks_inputs_as_well_as_tokens(self):
        reference = row()
        reference["prompt_token_ids"] = [99]
        result = compare_outputs([row()], ["graph"], ["p"], 1, references={"p": reference})
        self.assertFalse(result["all_equal"])
        self.assertFalse(result["checks"][0]["same_input"])
        self.assertIsNone(result["checks"][0]["first_difference"])

    def test_saved_reference_requires_exact_prompt_coverage(self):
        for references in ({}, {"wrong": row()}, {"p": row(), "extra": row()}):
            with self.subTest(references=references), self.assertRaisesRegex(ValueError, "cover exactly"):
                compare_outputs([row()], ["graph"], ["p"], 1, references=references)

    def test_missing_extra_duplicate_and_empty_results_fail(self):
        for rows in ([], [row()], [row(), row(), row("mtp")],
                     [row(), row("mtp"), row("eager")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                compare_outputs(rows, ["graph", "mtp"], ["p"], 1)

    def test_warmup_is_not_a_measured_sample(self):
        rows = [row(), row(phase="warmup", tokens=[100])]
        self.assertTrue(compare_outputs(rows, ["graph"], ["p"], 1)["all_equal"])
        rows[1]["stats"]["decode_output_tok_s"] = 9999
        result = summarize(rows)[0]
        self.assertEqual(result["measured_rounds"], 1)
        self.assertEqual(result["decode_output_tok_s"]["median"], 100)
        self.assertIsNone(result["acceptance_rate"])

    def test_summary_groups_modes_and_uses_per_round_median(self):
        rows = [row(), row(repeat=1), row(repeat=2), row("mtp")]
        for r, rate in zip(rows, (10, 30, 20, 200)):
            r["stats"] = copy.copy(r["stats"])
            r["stats"]["decode_output_tok_s"] = rate
        summary = summarize(rows)
        self.assertEqual(summary[0]["decode_output_tok_s"], {"median": 20, "min": 10, "max": 30})
        self.assertEqual(summary[1]["mode"], "mtp")


if __name__ == "__main__":
    unittest.main()
