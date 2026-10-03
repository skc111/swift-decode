import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import diagnose_mtp as d
from experiments.provenance import file_hash, object_hash


class BatchAlignmentTests(unittest.TestCase):
    def test_accepted_drafts_and_bonus_map_to_successor_positions(self):
        check = d.inspect_batch([10, 20, 21, 50], 1, [20, 21, 50], [20, 21, 50])
        self.assertTrue(check["collector_matches_target"])
        self.assertIsNone(check["first_difference"])

    def test_target_divergence_is_not_a_collector_error(self):
        check = d.inspect_batch([10, 20, 99, 50], 1, [20, 21, 50], [20, 21, 50])
        self.assertTrue(check["collector_matches_target"])
        self.assertEqual(check["first_difference"], 2)

    def test_reemitting_committed_input_is_detected(self):
        check = d.inspect_batch([10, 20, 21, 50], 1, [10, 20, 21], [20, 21, 50])
        self.assertFalse(check["collector_matches_target"])
        self.assertEqual(check["first_difference"], 1)

    def test_zero_acceptance_still_emits_target_token(self):
        check = d.inspect_batch([10, 50], 1, [50], [50])
        self.assertTrue(check["collector_matches_target"])
        self.assertEqual(check["emitted_tokens"], [50])

    def test_tail_is_clipped_but_collector_is_checked_for_full_batch(self):
        check = d.inspect_batch([10, 20], 1, [20, 21, 50], [20, 22, 50])
        self.assertEqual(check["emitted_tokens"], [20])
        self.assertIsNone(check["first_difference"])
        self.assertFalse(check["collector_matches_target"])


class SavedCaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {"tokens": 4, "respect_eos": False}
        self.summary = {"stage": "gate", "status": "failed", "configuration_sha256": object_hash(self.cfg),
                        "output_comparison": {"reference": "graph", "checks": [
                            {"mode": "mtp", "prompt_id": "p", "round": 2, "equal": False,
                             "first_difference": 2}]}}
        self.graph = {"phase": "measure", "prompt_id": "p", "round": 0,
                      "prompt_token_ids": [100, 101], "token_ids": [10, 20, 21, 50]}
        self.mtp = {**self.graph, "round": 2, "token_ids": [10, 20, 22, 50]}

    def save(self):
        for name, value in (("configuration.json", self.cfg), ("summary.json", self.summary),
                            ("graph.jsonl", self.graph), ("mtp.jsonl", self.mtp)):
            (self.root / name).write_text(json.dumps(value) + "\n")

    def test_selects_the_actual_failing_round_without_modifying_gate(self):
        self.save()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        _, _, rows = d.load_case(self.root)
        self.assertEqual(rows["mtp"]["round"], 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_rejects_changed_configuration(self):
        self.cfg["tokens"] = 5
        self.save()
        with self.assertRaisesRegex(ValueError, "intact configuration"):
            d.load_case(self.root)

    def test_rejects_input_mismatch(self):
        self.mtp["prompt_token_ids"] = [999]
        self.save()
        with self.assertRaisesRegex(ValueError, "prompt IDs differ"):
            d.load_case(self.root)

    def test_rejects_output_that_no_longer_matches_recorded_failure(self):
        self.mtp["token_ids"] = [10, 99, 22, 50]
        self.save()
        with self.assertRaisesRegex(ValueError, "recorded first difference"):
            d.load_case(self.root)

    def test_rejects_missing_failed_row(self):
        self.mtp["round"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "expected one saved mtp row"):
            d.load_case(self.root)

    def test_rejects_truncated_output(self):
        self.mtp["token_ids"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "saved output length"):
            d.load_case(self.root)

    def test_source_changes_are_rejected_but_new_diagnostic_files_are_allowed(self):
        source = self.root / "engine.py"
        source.write_text("original source")
        fingerprints = {source.name: file_hash(source)}
        (self.root / "new_diagnostic.py").write_text("new file")
        with patch.object(d, "ROOT", self.root):
            d.validate_sources(fingerprints)
            source.write_text("changed source")
            with self.assertRaisesRegex(ValueError, "source changed"):
                d.validate_sources(fingerprints)


if __name__ == "__main__":
    unittest.main()
