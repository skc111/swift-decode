import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import provenance as p


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def model(self):
        (self.root / "config.json").write_text("{}")
        (self.root / "tokenrush.json").write_text('{"shards": 1}')
        (self.root / "model-00001.safetensors").write_bytes(b"test-only-not-real-weights")
        return self.root

    def test_missing_model_fails_without_downloading(self):
        with self.assertRaisesRegex(ValueError, "no automatic downloads"):
            p.checkpoint_manifest(self.root / "missing")

    def test_plain_hf_config_is_not_a_packed_checkpoint(self):
        (self.root / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "tokenrush.json"):
            p.checkpoint_manifest(self.root)

    def test_shard_count_mismatch_is_rejected(self):
        self.model()
        (self.root / "tokenrush.json").write_text('{"shards": 2}')
        with self.assertRaisesRegex(ValueError, "shard count"):
            p.checkpoint_manifest(self.root)

    def test_empty_shard_is_rejected(self):
        self.model()
        (self.root / "model-00001.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty weight"):
            p.checkpoint_manifest(self.root)

    def test_manifest_hashes_metadata_not_weights(self):
        a = p.checkpoint_manifest(self.model())
        self.assertEqual(len(a["weight_files"]), 1)
        self.assertNotIn("sha256", a["weight_files"][0])
        self.assertEqual(a["metadata_sha256"]["config.json"], p.file_hash(self.root / "config.json"))
        (self.root / "config.json").write_text('{"changed": true}')
        self.assertNotEqual(p.object_hash(a), p.object_hash(p.checkpoint_manifest(self.root)))

    def test_draft_can_use_a_single_unsplit_safetensors_file(self):
        (self.root / "config.json").write_text("{}")
        (self.root / "model.safetensors").write_bytes(b"fake")
        self.assertEqual(p.checkpoint_manifest(self.root, packed=False)["weight_files"][0]["bytes"], 4)

    def test_source_snapshot_includes_untracked_files(self):
        (self.root / "new.py").write_text("# untracked source\n")
        hashes = {"new.py": p.file_hash(self.root / "new.py")}
        with patch.object(p, "ROOT", self.root):
            p.snapshot_sources(self.root / "snapshot", hashes)
            self.assertEqual((self.root / "snapshot/new.py").read_text(), "# untracked source\n")

    def test_snapshot_rejects_source_changes_and_path_escape(self):
        (self.root / "new.py").write_text("changed")
        with patch.object(p, "ROOT", self.root):
            for hashes in ({"new.py": "old-hash"}, {"../outside.py": "hash"}):
                with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                    p.snapshot_sources(self.root / "snapshot", hashes)

    def test_runtime_ignores_utilization_but_tracks_driver(self):
        a = {"packages": {"torch": "version"}, "gpu_driver_query": {"stdout": "GPU-A, driver1"},
             "nvidia_smi": "time1/util0"}
        b = {**a, "nvidia_smi": "time2/util20"}
        self.assertEqual(p.runtime_fingerprint(a), p.runtime_fingerprint(b))
        b["gpu_driver_query"] = {"stdout": "GPU-A, driver2"}
        self.assertNotEqual(p.runtime_fingerprint(a), p.runtime_fingerprint(b))

    def test_failed_environment_command_is_recorded(self):
        result = p.command([str(self.root / "nonexistent-program")])
        self.assertIsNone(result["returncode"])
        self.assertIn("error", result)

    def test_object_hash_does_not_depend_on_dict_key_order(self):
        self.assertEqual(p.object_hash({"a": 1, "b": 2}), p.object_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
