from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import single_gpu as s
from experiments.gpu import Adapter
from experiments.provenance import object_hash


def record(mode, repeat=0):
    return {"mode": mode, "prompt_id": "p", "round": repeat, "phase": "measure",
            "prompt_token_ids": [11], "token_ids": [1, 2],
            "stats": {"first_token_s": .1, "decode_output_tok_s": 100, "output_tok_s": 20,
                      "tpot_s": .01, "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048,
                      "acceptance_rate": None}}


class CliTests(unittest.TestCase):
    def test_plan_does_not_import_torch_or_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "not-created"
            with patch.dict(sys.modules, {"torch": None}), redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(s.main(["--plan", "--stage", "bench", "--model", "/not-downloaded",
                                         "--output", str(output)]), 0)
            self.assertFalse(output.exists())
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["measured_requests"], 27)
            self.assertFalse(plan["gpu_used"])
            self.assertTrue(plan["requires_passing_gate"])

    def test_invalid_cli_combinations(self):
        for argv in (["--modes", "mtp"], ["--modes", "graph,graph"], ["--modes", "graph,typo"],
                     ["--modes", "graph,dflash"], ["--mtp-depth", "8"], ["--tokens", "0"],
                     ["--warmup", "-1"], ["--stage", "bench"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                s.validate(s.parser().parse_args(["--model", "/local", *argv]))

    def test_child_keeps_parameters_and_same_python(self):
        args = s.parser().parse_args(["--model", "/model", "--modes", "graph,dflash", "--draft-model", "/draft",
                                      "--kv", "fp8", "--backend", "marlin", "--respect-eos"])
        cmd = s.child_command(args, "dflash", Path("/output"))
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[cmd.index("--worker-mode") + 1], "dflash")
        self.assertEqual(cmd[cmd.index("--draft-model") + 1], "/draft")
        self.assertIn("--respect-eos", cmd)
        self.assertEqual(cmd[cmd.index("--kv") + 1], "fp8")

    def test_prompt_ids_and_text_must_be_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prompts.json"
            for prompts in ([], {}, [{"id": "a"}], [{"id": "a", "text": " "}],
                            [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]):
                p.write_text(json.dumps(prompts))
                with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                    s.read_prompts(p)
            p.write_text('[{"id": "p", "text": "hello"}]')
            self.assertEqual(s.read_prompts(p)[0]["id"], "p")

    def test_gate_needs_matching_code_and_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            summary = {"stage": "gate", "status": "passed", "configuration_sha256": "config", "runtime_sha256": "runtime"}
            s.write_json(Path(d) / "summary.json", summary)
            s.check_gate(d, "config", "runtime")
            for config, runtime in (("changed-code", "runtime"), ("config", "changed-driver")):
                with self.subTest(config=config, runtime=runtime), self.assertRaises(ValueError):
                    s.check_gate(d, config, runtime)
            summary["status"] = "failed"
            s.write_json(Path(d) / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "did not pass"):
                s.check_gate(d, "config", "runtime")


class TokenAdapterTests(unittest.TestCase):
    class TensorList(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            return type(self)(value) if isinstance(index, slice) else value

        def tolist(self):
            return list(self)

    def test_spec_step_emits_successors_not_the_committed_input_again(self):
        for mode in ("mtp", "dflash"):
            for accepted in (0, 1, 3):
                with self.subTest(mode=mode, accepted=accepted):
                    a = Adapter.__new__(Adapter)
                    a.mode, a.k = mode, 3
                    a.engine = SimpleNamespace(drafts=self.TensorList([20, 21, 22]), tok=[50],
                                               spec_step=Mock(return_value=accepted),
                                               spec_step_dflash=Mock(return_value=accepted))
                    step = a.step()
                    self.assertEqual(step.tokens, [20, 21, 22][:accepted] + [50])
                    self.assertEqual(step.accepted, accepted)
                    self.assertEqual(step.proposed, 3)

    def test_capacity_includes_verify_scratch(self):
        a = Adapter.__new__(Adapter)
        a.k, a.args = 3, SimpleNamespace(tokens=4, max_len=9)
        a.tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value="text"),
                                      encode=Mock(return_value=[1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "scratch"):
            a.encode("x")
        a.args.max_len = 10
        self.assertEqual(a.encode("x"), [1, 2, 3])


class SuiteTests(unittest.TestCase):
    """Fake workers exercise orchestration only; these are NOT GPU tests."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {"sources": {}, "fake_cpu_test": True}
        self.prompts = [{"id": "p", "text": "hello"}]
        for target, value in (("configuration", self.cfg), ("check_gpu_environment", {"passed": True, "errors": []})):
            p = patch.object(s, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def args(self, output, stage="gate", gate=None):
        argv = ["--model", "/fake", "--output", str(self.root / output), "--repeats", "1", "--stage", stage]
        if gate:
            argv += ["--gate-dir", str(self.root / gate)]
        return s.parser().parse_args(argv)

    def fake_child(self, cmd, log, timeout):
        mode = cmd[cmd.index("--worker-mode") + 1]
        output = Path(cmd[cmd.index("--output") + 1])
        (output / f"{mode}.jsonl").write_text(json.dumps(record(mode)) + "\n")
        return 0

    def run_suite(self, args, child=None):
        with redirect_stdout(io.StringIO()), patch.object(s, "run_child", side_effect=child or self.fake_child):
            return s.suite(args, s.validate(args), self.prompts)

    def test_gate_then_bench_writes_complete_evidence(self):
        self.assertEqual(self.run_suite(self.args("gate")), 0)
        gate = json.loads((self.root / "gate/summary.json").read_text())
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["performance"], [])
        self.assertEqual(gate["configuration_sha256"], object_hash(self.cfg))
        self.assertEqual(self.run_suite(self.args("bench", "bench", "gate")), 0)
        result = json.loads((self.root / "bench/summary.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["performance"]), 3)
        self.assertTrue((self.root / "bench/comparison.csv").exists())
        suite = json.loads((self.root / "bench/suite.json").read_text())
        self.assertTrue(all(j["status"] == "complete" for j in suite["jobs"]))

    def test_mismatch_fails_gate_and_blocks_bench(self):
        def wrong(cmd, log, timeout):
            rc = self.fake_child(cmd, log, timeout)
            if cmd[cmd.index("--worker-mode") + 1] == "mtp":
                path = Path(cmd[cmd.index("--output") + 1]) / "mtp.jsonl"
                r = record("mtp")
                r["token_ids"] = [1, 99]
                path.write_text(json.dumps(r) + "\n")
            return rc
        self.assertEqual(self.run_suite(self.args("gate"), wrong), 1)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.run_suite(self.args("bench", "bench", "gate"))
        self.assertFalse((self.root / "bench").exists())

    def test_child_failure_stops_and_records_failure(self):
        with self.assertRaisesRegex(RuntimeError, "exit 7"):
            self.run_suite(self.args("failed"), lambda *a: 7)
        manifest = json.loads((self.root / "failed/suite.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["jobs"]), 1)
        self.assertEqual(manifest["jobs"][0]["returncode"], 7)

    def test_normal_kernel_bench_mismatch_is_saved_not_hidden(self):
        self.run_suite(self.args("gate"))
        def changed(cmd, log, timeout):
            self.fake_child(cmd, log, timeout)
            if cmd[cmd.index("--worker-mode") + 1] == "mtp":
                path = Path(cmd[cmd.index("--output") + 1]) / "mtp.jsonl"
                r = record("mtp")
                r["token_ids"] = [9, 10]
                path.write_text(json.dumps(r) + "\n")
            return 0
        self.assertEqual(self.run_suite(self.args("bench", "bench", "gate"), changed), 0)
        result = json.loads((self.root / "bench/summary.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["output_comparison"]["all_equal"])
        self.assertEqual(len(result["performance"]), 3)

    def test_existing_output_is_never_overwritten(self):
        (self.root / "existing").mkdir()
        marker = self.root / "existing/keep.txt"
        marker.write_text("keep me")
        with self.assertRaises(FileExistsError):
            self.run_suite(self.args("existing"))
        self.assertEqual(marker.read_text(), "keep me")

    def test_runtime_change_blocks_bench_before_workers(self):
        self.run_suite(self.args("gate"))
        with patch.object(s, "check_gpu_environment", return_value={"passed": True, "errors": [], "gpu": "different"}):
            with self.assertRaisesRegex(ValueError, "environment differs"):
                self.run_suite(self.args("bench", "bench", "gate"), lambda *a: self.fail("worker must not start"))

    def test_missing_measured_records_do_not_count_as_a_pass(self):
        def empty(cmd, log, timeout):
            self.fake_child(cmd, log, timeout)
            if cmd[cmd.index("--worker-mode") + 1] == "mtp":
                (Path(cmd[cmd.index("--output") + 1]) / "mtp.jsonl").write_text("")
            return 0
        with self.assertRaisesRegex(ValueError, "incomplete results"):
            self.run_suite(self.args("missing"), empty)
        self.assertEqual(json.loads((self.root / "missing/suite.json").read_text())["status"], "failed")

    def test_environment_failure_is_saved_without_running_workers(self):
        with patch.object(s, "check_gpu_environment", return_value={"passed": False, "errors": ["missing torch"]}):
            with self.assertRaisesRegex(RuntimeError, "missing torch"):
                self.run_suite(self.args("no-gpu"), lambda *a: self.fail("must not run"))
        self.assertTrue((self.root / "no-gpu/environment.json").exists())

    def test_child_process_exit_is_propagated(self):
        rc = s.run_child([sys.executable, "-c", "raise SystemExit(7)"], self.root / "exit.log", 10)
        self.assertEqual(rc, 7)

    @unittest.skipUnless(sys.platform == "linux", "process-group cancellation is a Linux path")
    def test_child_timeout_is_bounded(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            s.run_child([sys.executable, "-c", "import time; time.sleep(60)"], self.root / "timeout.log", .05)


if __name__ == "__main__":
    unittest.main()
