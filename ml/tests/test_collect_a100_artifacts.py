from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "collect_a100_artifacts.py"
SPEC = importlib.util.spec_from_file_location("collect_a100_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectA100ArtifactsTests(unittest.TestCase):
    def test_profiles_exclude_bulk_state_and_tokenizer(self) -> None:
        rules = collector.profile_rules("finished-v1")
        self.assertNotIn("optimizer.pt", rules["run_root_exact"])
        self.assertNotIn("rng_state.pth", rules["run_root_exact"])
        self.assertNotIn("scheduler.pt", rules["run_root_exact"])
        self.assertIn("tokenizer.json", rules["run_root_json_exclude"])
        self.assertNotIn(collector.CHALLENGER_RUN, rules["runs"])

        challenger = collector.profile_rules("with-challenger-v1")
        self.assertIn(collector.CHALLENGER_RUN, challenger["runs"])
        self.assertEqual([], challenger["excluded_log_fragments"])

    def test_cross_run_posttrain_profile_is_narrow_immutable_and_completed(self) -> None:
        rules = collector.profile_rules("cross-run-posttrain-v1")
        self.assertTrue(rules["immutable"])
        self.assertEqual([], rules["runs"])
        self.assertFalse(rules["include_standard_logs"])
        self.assertFalse(rules["include_gguf"])
        self.assertEqual([collector.POSTTRAIN_LOG], rules["required_exact_files"])
        self.assertEqual(1, len(rules["evidence_trees"]))
        tree = rules["evidence_trees"][0]
        self.assertEqual(collector.POSTTRAIN_RUN, tree["root"])
        self.assertEqual("posttrain-completion.json", tree["completion_file"])
        self.assertEqual(
            "completed_research_evaluation_not_runtime_promotion",
            tree["required_status"],
        )
        self.assertTrue(tree["verify_completion_integrity"])
        for excluded in (
            "optimizer.pt",
            "rng_state.pth",
            "scheduler.pt",
            "tokenizer.json",
        ):
            self.assertIn(excluded, tree["excluded_file_names"])

    def test_local_manifest_and_comparison_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "ml" / "runs" / "a.json"
            second = root / "ml" / "gguf" / "b.gguf"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"alpha\n")
            second.write_bytes(b"beta\n")
            remote = {
                "schema_version": 1,
                "profile": "finished-v1",
                "selection_rules_sha256": "1" * 64,
                "files": [
                    {
                        "path": "ml/gguf/b.gguf",
                        "sha256": collector.sha256_file(second),
                        "size_bytes": second.stat().st_size,
                    },
                    {
                        "path": "ml/runs/a.json",
                        "sha256": collector.sha256_file(first),
                        "size_bytes": first.stat().st_size,
                    },
                ],
            }
            local, missing = collector.local_manifest(root, remote)
            result = collector.comparison(remote, local, missing)
            self.assertTrue(result["matched"])
            self.assertEqual(
                result["remote_manifest_sha256"],
                result["local_manifest_sha256"],
            )
            self.assertEqual(
                collector.canonical_bytes(remote), collector.canonical_bytes(local)
            )

    def test_comparison_reports_changed_and_missing_files(self) -> None:
        remote = {
            "schema_version": 1,
            "profile": "finished-v1",
            "selection_rules_sha256": "2" * 64,
            "files": [
                {"path": "a", "sha256": "a" * 64, "size_bytes": 1},
                {"path": "b", "sha256": "b" * 64, "size_bytes": 2},
            ],
        }
        local = {
            "schema_version": 1,
            "profile": "finished-v1",
            "selection_rules_sha256": "2" * 64,
            "files": [
                {"path": "a", "sha256": "c" * 64, "size_bytes": 3},
            ],
        }
        result = collector.comparison(remote, local, ["b"])
        self.assertFalse(result["matched"])
        self.assertEqual(["a"], result["hash_mismatches"])
        self.assertEqual(["a"], result["size_mismatches"])
        self.assertEqual(["b"], result["missing"])

    def test_validate_relative_path_rejects_traversal(self) -> None:
        for unsafe in ("../secret", "/absolute/path", "a/../../secret", ""):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(RuntimeError):
                    collector.validate_relative_path(unsafe)

    def test_evidence_files_are_byte_stable(self) -> None:
        manifest = {
            "schema_version": 1,
            "profile": "finished-v1",
            "selection_rules_sha256": "3" * 64,
            "files": [{"path": "x", "sha256": "4" * 64, "size_bytes": 0}],
        }
        result = collector.comparison(manifest, manifest, [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector.write_evidence(root, "finished-v1", manifest, manifest, result)
            before = {
                path.name: path.read_bytes() for path in sorted(root.iterdir())
            }
            collector.write_evidence(root, "finished-v1", manifest, manifest, result)
            after = {
                path.name: path.read_bytes() for path in sorted(root.iterdir())
            }
            self.assertEqual(before, after)
            for payload in after.values():
                json.loads(payload)

    def test_immutable_snapshot_cannot_be_repointed(self) -> None:
        original = {
            "schema_version": 1,
            "profile": "cross-run-posttrain-v1",
            "selection_rules_sha256": "5" * 64,
            "files": [{"path": "a", "sha256": "6" * 64, "size_bytes": 1}],
        }
        changed = {
            **original,
            "files": [{"path": "a", "sha256": "7" * 64, "size_bytes": 1}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "cross-run-posttrain-v1.remote-manifest.json"
            path.write_bytes(collector.canonical_bytes(original))
            collector.enforce_immutable_snapshot(
                root, "cross-run-posttrain-v1", original, True
            )
            with self.assertRaisesRegex(RuntimeError, "immutable profile"):
                collector.enforce_immutable_snapshot(
                    root, "cross-run-posttrain-v1", changed, True
                )

            # Mutable profiles retain the existing incremental behavior.
            collector.enforce_immutable_snapshot(
                root, "cross-run-posttrain-v1", changed, False
            )


if __name__ == "__main__":
    unittest.main()
