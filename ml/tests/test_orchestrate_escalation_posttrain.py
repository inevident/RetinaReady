import csv
import hashlib
import json
import math
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file as save_safetensors
import sys
import tempfile
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from calibrate_escalation_adapter import build_report as build_policy_report  # noqa: E402
from evaluate_decision_logits import (  # noqa: E402
    load_decision_training_contract,
    summarize,
)
from evaluate_peft import adapter_metadata  # noqa: E402
from train_qlora import select_rows_for_run, selected_rows_sha256  # noqa: E402
from orchestrate_escalation_posttrain import (  # noqa: E402
    Candidate,
    DEFAULT_DELTA,
    DEFAULT_FALSE_PRIORITY_RISK,
    DEFAULT_FALSE_ROUTINE_RISK,
    PosttrainError,
    RiskProfile,
    SafetyCriteria,
    build_parser,
    deduplicate_candidates,
    discover_candidates,
    evaluation_command,
    reuse_or_evaluate,
    select_candidate,
    sha256_file,
    validate_completed_full_run,
    validate_policy_report,
    build_decision_evidence,
    decision_evidence_path,
    input_content_binding,
    atomic_write_json,
    load_external_candidate,
    validate_partition_separation,
    with_integrity,
    reuse_or_write_selection,
)


MANIFEST_FIELDS = [
    "split",
    "patient_id",
    "image_id",
    "image_path",
    "dr_grade",
    "escalation_label",
    "overall_quality",
    "source_split",
    "grade_source_field",
    "filename_side_matches_grade_field",
]


class EscalationPosttrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = {
            "model_id": "test/model",
            "model_revision": "model-revision",
            "processor_id": "test/processor",
            "processor_revision": "processor-revision",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def adapter_config_payload(self) -> dict:
        return {
            "base_model_name_or_path": self.sources["model_id"],
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
            "bias": "none",
            "modules_to_save": None,
            "rank_pattern": {},
            "use_dora": False,
        }

    @staticmethod
    def write_adapter_weights(path: Path, marker: bytes) -> dict:
        digest = hashlib.sha256(marker).digest()
        values = np.frombuffer(digest[:14], dtype=np.uint8).astype(np.float32)
        values = (values + 1.0) / 256.0
        prefix = (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj"
        )
        save_safetensors(
            {
                f"{prefix}.lora_A.weight": values[:6].reshape(2, 3),
                f"{prefix}.lora_B.weight": values[6:14].reshape(4, 2),
            },
            str(path),
            metadata={"format": "pt"},
        )
        names = [
            f"{prefix}.lora_A.default.weight",
            f"{prefix}.lora_B.default.weight",
        ]
        return {
            "count": 14,
            "tensor_count": 2,
            "names_sha256": hashlib.sha256(
                "\n".join(names).encode("utf-8")
            ).hexdigest(),
        }

    def provenance_payload(self, trainable: dict) -> dict:
        return {
            "status": "completed",
            "failure": None,
            "checkpoint_sources": self.sources,
            "effective_config": {
                "task": "escalation",
                "loss_scope": "decision_token",
                **self.sources,
                "lora_rank": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "lora_target_regex": None,
            },
            "trainable_parameters": trainable,
            "label_token_contract": {
                "ROUTINE": {
                    "first_token_id": 2073,
                    "full_encoding": [2073],
                },
                "PRIORITY": {
                    "first_token_id": 65324,
                    "full_encoding": [65324],
                },
            },
        }

    def make_candidate(
        self,
        identifier: str,
        *,
        weights: bytes,
        adapter_config: dict | None = None,
        global_step: int | None = None,
    ) -> Candidate:
        directory = self.root / identifier
        directory.mkdir()
        config_path = directory / "adapter_config.json"
        config_path.write_text(
            json.dumps(adapter_config or self.adapter_config_payload()), encoding="utf-8"
        )
        weights_path = directory / "adapter_model.safetensors"
        trainable = self.write_adapter_weights(weights_path, weights)
        provenance_path = directory / "run_provenance.json"
        provenance_path.write_text(
            json.dumps(self.provenance_payload(trainable)),
            encoding="utf-8",
        )
        trainer_state_path = None
        epoch = None
        if global_step is not None:
            trainer_state_path = directory / "trainer_state.json"
            epoch = float(global_step // 10)
            trainer_state_path.write_text(
                json.dumps({"global_step": global_step, "epoch": epoch}),
                encoding="utf-8",
            )
        return Candidate(
            identifier=identifier,
            path=directory.resolve(),
            role=(
                "completed_root_best_by_training_eval_loss"
                if identifier == "root"
                else "retained_epoch_checkpoint"
            ),
            global_step=global_step,
            epoch=epoch,
            trainer_state_path=trainer_state_path,
            adapter_config_path=config_path,
            weights_path=weights_path,
            provenance_path=provenance_path,
        )

    def make_manifest(self, split: str, prefix: str) -> tuple[Path, list[dict[str, str]]]:
        rows: list[dict[str, str]] = []
        for index, (label, grade) in enumerate(
            (("ROUTINE", "0"), ("ROUTINE", "1"), ("PRIORITY", "2"), ("PRIORITY", "4"))
        ):
            image_path = self.root / f"{prefix}-{index}.jpg"
            image_path.write_bytes(b"test-image-placeholder")
            rows.append(
                {
                    "split": split,
                    "patient_id": f"{prefix}-patient-{index}",
                    "image_id": f"{prefix}-image-{index}",
                    "image_path": str(image_path),
                    "dr_grade": grade,
                    "escalation_label": label,
                    "overall_quality": "1",
                    "source_split": f"source-{split}",
                    "grade_source_field": "left_eye_DR_Level",
                    "filename_side_matches_grade_field": "true",
                }
            )
        manifest = self.root / f"{prefix}-{split}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return manifest, rows

    def make_completed_full_run(
        self, *, max_train_samples: int | None = None, max_eval_samples: int | None = 4
    ) -> tuple[Path, dict, Path, dict]:
        train_manifest, train_rows = self.make_manifest("train", "full-train")
        val_manifest, val_rows = self.make_manifest("val", "full-val")
        calibration_manifest, _ = self.make_manifest(
            "calibration", "full-calibration"
        )
        eval_manifest, _ = self.make_manifest("eval", "full-eval")
        run = self.root / "full-run"
        run.mkdir()
        (run / "adapter_config.json").write_text(
            json.dumps(self.adapter_config_payload()), encoding="utf-8"
        )
        trainable = self.write_adapter_weights(
            run / "adapter_model.safetensors", b"full-root"
        )
        config = {
            "task": "escalation",
            **self.sources,
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "calibration_manifest": str(calibration_manifest),
            "eval_manifest": str(eval_manifest),
            "output_dir": str(run),
            "epochs": 1,
            "max_steps": -1,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "learning_rate": 0.0001,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 32,
            "lora_rank": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "lora_target_regex": None,
            "loss_scope": "decision_token",
            "stratified_sampling": False,
            "seed": 42,
        }
        config_path = self.root / "full-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        provenance = self.provenance_payload(trainable)
        provenance["effective_config"].update(config)
        provenance["selected_data"] = {
            "train_rows": len(train_rows),
            "train_rows_sha256": selected_rows_sha256(
                select_rows_for_run(
                    train_rows,
                    max_train_samples,
                    seed=42,
                    stratified=False,
                    task="escalation",
                ),
                "escalation",
            ),
            "validation_rows": len(val_rows),
            "validation_rows_sha256": selected_rows_sha256(
                select_rows_for_run(
                    val_rows,
                    max_eval_samples,
                    seed=43,
                    stratified=False,
                    task="escalation",
                ),
                "escalation",
            ),
        }
        provenance["manifests"] = {
            "train": {
                "sha256": sha256_file(train_manifest),
                "rows": len(train_rows),
            },
            "validation": {
                "sha256": sha256_file(val_manifest),
                "rows": len(val_rows),
            },
        }
        provenance["config_source"] = {
            "path": f"/workspace/retina-ready/{config_path.name}",
            "sha256": sha256_file(config_path),
            "values": config,
        }
        provenance_path = run / "run_provenance.json"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return config_path, config, provenance_path, provenance

    def make_decision_report(
        self,
        candidate: Candidate,
        manifest: Path,
        rows: list[dict[str, str]],
        split: str,
        scores: list[float] | None = None,
    ) -> Path:
        scores = scores or [0.10, 0.20, 0.80, 0.90]
        results = []
        for row, score in zip(rows, scores, strict=True):
            margin = math.log(score / (1.0 - score))
            results.append(
                {
                    "image_id": row["image_id"],
                    "patient_id": row["patient_id"],
                    "image_path": row["image_path"],
                    "truth": row["escalation_label"],
                    "prediction": "PRIORITY" if score >= 0.5 else "ROUTINE",
                    "positive_label": "PRIORITY",
                    "negative_label": "ROUTINE",
                    "positive_logit": margin,
                    "negative_logit": 0.0,
                    "positive_minus_negative_logit": margin,
                    "positive_probability": score,
                    "negative_probability": 1.0 - score,
                    "latency_ms": 1.0,
                }
            )
        payload = {
            "run": {
                "mode": "decision-token-logits",
                "task": "escalation",
                **self.sources,
                "adapter": {
                    **adapter_metadata(candidate.path),
                    "weights_sha256": sha256_file(candidate.weights_path),
                },
                "training_contract": load_decision_training_contract(
                    adapter_metadata(candidate.path),
                    self.sources["model_id"],
                    task="escalation",
                ),
                "decision_threshold": 0.5,
                "roc_auc_positive_class": "PRIORITY",
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "expected_split": split,
                "available_rows": len(rows),
                "selected_rows": len(rows),
                "sampling": "sequential",
                "token_contract": {
                    "PRIORITY": {
                        "token_id": 65324,
                        "full_encoding": [65324],
                    },
                    "ROUTINE": {
                        "token_id": 2073,
                        "full_encoding": [2073],
                    },
                },
                "prefix_validation": {"prefix_tokens": 10},
                "completed_at": "2026-08-01T00:00:00+00:00",
            },
            "summary": summarize(results, 1.0, task="escalation"),
            "results": results,
        }
        report = self.root / f"{candidate.identifier}-{split}.json"
        report.write_text(json.dumps(payload), encoding="utf-8")
        return report

    @staticmethod
    def metrics(
        *, auc: float, false_routine: int, priority_recall: float, balanced: float
    ) -> dict:
        return {
            "roc_auc": auc,
            "false_routine_count": false_routine,
            "priority_recall": priority_recall,
            "balanced_accuracy": balanced,
            "routine_recall": max(0.50, min(1.0, 2 * balanced - priority_recall)),
        }

    def record(self, identifier: str, metrics: dict) -> dict:
        return {
            "candidate": {"identifier": identifier},
            "validation": {"metrics": metrics},
        }

    def test_discovers_root_and_epoch_checkpoint(self) -> None:
        run = self.root / "run"
        run.mkdir()
        adapter_config = json.dumps(self.adapter_config_payload())
        (run / "adapter_config.json").write_text(adapter_config, encoding="utf-8")
        trainable = self.write_adapter_weights(
            run / "adapter_model.safetensors", b"root"
        )
        provenance = self.provenance_payload(trainable)
        (run / "run_provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        checkpoint = run / "checkpoint-10"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(
            adapter_config, encoding="utf-8"
        )
        self.write_adapter_weights(
            checkpoint / "adapter_model.safetensors", b"epoch-one"
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 10,
                    "epoch": 1.0,
                    "log_history": [
                        {"step": 10, "epoch": 1.0, "eval_loss": 0.4}
                    ],
                }
            ),
            encoding="utf-8",
        )

        candidates = discover_candidates(
            run.resolve(),
            provenance,
            expected_epochs=3.0,
            expected_steps_per_epoch=10,
        )

        self.assertEqual([item.identifier for item in candidates], ["root", "checkpoint-10"])
        self.assertTrue((checkpoint / "run_provenance.json").is_file())

    def test_fractional_or_non_boundary_checkpoint_fails_closed(self) -> None:
        run = self.root / "run"
        run.mkdir()
        adapter_config = json.dumps(self.adapter_config_payload())
        (run / "adapter_config.json").write_text(adapter_config, encoding="utf-8")
        trainable = self.write_adapter_weights(
            run / "adapter_model.safetensors", b"root"
        )
        provenance = self.provenance_payload(trainable)
        (run / "run_provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        checkpoint = run / "checkpoint-15"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(
            adapter_config, encoding="utf-8"
        )
        self.write_adapter_weights(
            checkpoint / "adapter_model.safetensors", b"injected"
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 15,
                    "epoch": 1.5,
                    "log_history": [
                        {"step": 15, "epoch": 1.5, "eval_loss": 0.1}
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PosttrainError, "not a valid retained epoch"):
            discover_candidates(
                run.resolve(),
                provenance,
                expected_epochs=3.0,
                expected_steps_per_epoch=10,
            )

    def test_deduplicates_exact_root_checkpoint_alias(self) -> None:
        root = self.make_candidate("root", weights=b"same")
        duplicate = self.make_candidate(
            "checkpoint-10", weights=b"same", global_step=10
        )
        distinct = self.make_candidate(
            "checkpoint-20", weights=b"different", global_step=20
        )

        unique, audit = deduplicate_candidates([root, duplicate, distinct])

        self.assertEqual([item.identifier for item in unique], ["root", "checkpoint-20"])
        self.assertEqual(audit["root"]["alias_count"], 2)
        self.assertEqual(audit["root"]["deduplicated_evaluations_saved"], 1)
        self.assertEqual(
            [item["identifier"] for item in audit["root"]["aliases"]],
            ["root", "checkpoint-10"],
        )

    def test_same_weights_with_different_config_fails_closed(self) -> None:
        first = self.make_candidate(
            "root", weights=b"same", adapter_config={"r": 8}
        )
        second = self.make_candidate(
            "checkpoint-10",
            weights=b"same",
            adapter_config={"r": 16},
            global_step=10,
        )
        with self.assertRaisesRegex(PosttrainError, "different adapter_config"):
            deduplicate_candidates([first, second])

    def test_explicit_external_candidate_uses_locked_source_contract(self) -> None:
        candidate = self.make_candidate("smoke-qv", weights=b"external")
        loaded = load_external_candidate(
            candidate.path, index=1, config=self.sources
        )
        self.assertEqual(loaded.role, "explicit_external_completed_adapter")
        self.assertTrue(loaded.identifier.startswith("external-01-"))

        provenance = json.loads(candidate.provenance_path.read_text())
        provenance["effective_config"]["processor_revision"] = "wrong"
        candidate.provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "contract mismatch"):
            load_external_candidate(candidate.path, index=1, config=self.sources)

    def test_rejects_invalid_or_semantically_malformed_safetensors(self) -> None:
        invalid = self.make_candidate("invalid-bytes", weights=b"invalid")
        invalid.weights_path.write_bytes(b"not-a-safetensors-file")
        with self.assertRaisesRegex(PosttrainError, "invalid or unreadable"):
            load_external_candidate(invalid.path, index=1, config=self.sources)

        empty = self.make_candidate("empty", weights=b"empty")
        save_safetensors({}, str(empty.weights_path), metadata={"format": "pt"})
        with self.assertRaisesRegex(PosttrainError, "contains no tensors"):
            load_external_candidate(empty.path, index=2, config=self.sources)

        rank_mismatch = self.make_candidate("rank-mismatch", weights=b"rank")
        adapter_config = json.loads(rank_mismatch.adapter_config_path.read_text())
        adapter_config["r"] = 3
        rank_mismatch.adapter_config_path.write_text(json.dumps(adapter_config))
        provenance = json.loads(rank_mismatch.provenance_path.read_text())
        provenance["effective_config"]["lora_rank"] = 3
        rank_mismatch.provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(PosttrainError, "LoRA rank mismatch"):
            load_external_candidate(
                rank_mismatch.path, index=3, config=self.sources
            )

        nonfinite = self.make_candidate("nonfinite", weights=b"nan")
        prefix = (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj"
        )
        save_safetensors(
            {
                f"{prefix}.lora_A.weight": np.array(
                    [[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
                ),
                f"{prefix}.lora_B.weight": np.zeros((4, 2), dtype=np.float32),
            },
            str(nonfinite.weights_path),
            metadata={"format": "pt"},
        )
        with self.assertRaisesRegex(PosttrainError, "non-finite"):
            load_external_candidate(nonfinite.path, index=4, config=self.sources)

        inventory = self.make_candidate("inventory", weights=b"inventory")
        provenance = json.loads(inventory.provenance_path.read_text())
        provenance["trainable_parameters"]["tensor_count"] = 3
        inventory.provenance_path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(PosttrainError, "inventory/provenance mismatch"):
            load_external_candidate(inventory.path, index=5, config=self.sources)

        semantic_tamper = self.make_candidate("rslora", weights=b"rslora")
        adapter_config = json.loads(semantic_tamper.adapter_config_path.read_text())
        adapter_config["use_rslora"] = True
        semantic_tamper.adapter_config_path.write_text(json.dumps(adapter_config))
        with self.assertRaisesRegex(PosttrainError, "unsupported LoRA semantics"):
            load_external_candidate(
                semantic_tamper.path, index=6, config=self.sources
            )

    def test_external_directory_and_checkpoint_provenance_symlinks_fail(self) -> None:
        external = self.make_candidate("external-real", weights=b"external")
        external_link = self.root / "external-link"
        external_link.symlink_to(external.path, target_is_directory=True)
        with self.assertRaisesRegex(PosttrainError, "non-symlink directory"):
            load_external_candidate(external_link, index=1, config=self.sources)

        root = self.make_candidate("run", weights=b"root")
        checkpoint = root.path / "checkpoint-10"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_bytes(
            root.adapter_config_path.read_bytes()
        )
        self.write_adapter_weights(
            checkpoint / "adapter_model.safetensors", b"checkpoint"
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 10,
                    "epoch": 1.0,
                    "log_history": [
                        {"step": 10, "epoch": 1.0, "eval_loss": 0.25}
                    ],
                }
            ),
            encoding="utf-8",
        )
        provenance = json.loads(root.provenance_path.read_text())
        discover_candidates(
            root.path,
            provenance,
            expected_epochs=1.0,
            expected_steps_per_epoch=10,
        )
        checkpoint_provenance = checkpoint / "run_provenance.json"
        external_provenance = self.root / "outside-checkpoint-provenance.json"
        checkpoint_provenance.replace(external_provenance)
        checkpoint_provenance.symlink_to(external_provenance)
        with self.assertRaisesRegex(PosttrainError, "non-symlink file"):
            discover_candidates(
                root.path,
                provenance,
                expected_epochs=1.0,
                expected_steps_per_epoch=10,
            )

    def test_root_config_source_is_required_and_full_size_cap_is_allowed(self) -> None:
        config_path, config, provenance_path, provenance = (
            self.make_completed_full_run(max_eval_samples=4)
        )
        run_dir, _loaded, train_rows, val_rows = validate_completed_full_run(
            config, config_path
        )
        self.assertEqual(run_dir, Path(config["output_dir"]).resolve())
        self.assertEqual((len(train_rows), len(val_rows)), (4, 4))

        locked_source = json.loads(json.dumps(provenance["config_source"]))
        provenance.pop("config_source")
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "missing config_source"):
            validate_completed_full_run(config, config_path)

        provenance["config_source"] = json.loads(json.dumps(locked_source))
        provenance["config_source"]["values"]["learning_rate"] = 9.0
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "config_source.values changed"):
            validate_completed_full_run(config, config_path)

        provenance["config_source"] = json.loads(json.dumps(locked_source))
        provenance["config_source"]["sha256"] = "0" * 64
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "training config SHA mismatch"):
            validate_completed_full_run(config, config_path)

        provenance["config_source"] = json.loads(json.dumps(locked_source))
        provenance["effective_config"]["learning_rate"] = 9.0
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "effective_config differs"):
            validate_completed_full_run(config, config_path)

    def test_root_rejects_partial_cap_and_directory_symlink(self) -> None:
        config_path, config, _provenance_path, _provenance = (
            self.make_completed_full_run(max_eval_samples=3)
        )
        with self.assertRaisesRegex(PosttrainError, "validation cap must equal"):
            validate_completed_full_run(config, config_path)

        # Directory symlink rejection occurs before provenance is trusted.
        real_run = Path(config["output_dir"])
        run_link = self.root / "full-run-link"
        run_link.symlink_to(real_run, target_is_directory=True)
        config["output_dir"] = str(run_link)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "non-symlink directory"):
            validate_completed_full_run(config, config_path)

    def test_selection_is_auc_first_then_false_routine_then_balanced(self) -> None:
        criteria = SafetyCriteria(
            minimum_roc_auc=0.0,
            minimum_balanced_accuracy=0.0,
            minimum_priority_recall=0.0,
            minimum_routine_recall=0.0,
        )
        safer_at_point_five = self.record(
            "root",
            self.metrics(
                auc=0.90, false_routine=0, priority_recall=1.0, balanced=0.95
            ),
        )
        better_auc = self.record(
            "checkpoint-10",
            self.metrics(
                auc=0.91, false_routine=2, priority_recall=0.8, balanced=0.80
            ),
        )
        self.assertEqual(
            select_candidate([safer_at_point_five, better_auc], criteria)[
                "candidate"
            ]["identifier"],
            "checkpoint-10",
        )

        fewer_misses = self.record(
            "checkpoint-20",
            self.metrics(
                auc=0.91, false_routine=1, priority_recall=0.9, balanced=0.75
            ),
        )
        self.assertEqual(
            select_candidate([better_auc, fewer_misses], criteria)["candidate"]["identifier"],
            "checkpoint-20",
        )

        better_balanced = self.record(
            "checkpoint-30",
            self.metrics(
                auc=0.91, false_routine=1, priority_recall=0.9, balanced=0.85
            ),
        )
        self.assertEqual(
            select_candidate([fewer_misses, better_balanced], criteria)["candidate"]["identifier"],
            "checkpoint-30",
        )

    def test_lower_ranked_checkpoint_never_bypasses_failed_winner_floor(self) -> None:
        top_auc_below_recall_floor = self.record(
            "root",
            self.metrics(
                auc=0.95,
                false_routine=4,
                priority_recall=0.70,
                balanced=0.80,
            ),
        )
        lower_auc_passing = self.record(
            "checkpoint-10",
            self.metrics(
                auc=0.90,
                false_routine=0,
                priority_recall=1.0,
                balanced=0.90,
            ),
        )
        with self.assertRaisesRegex(PosttrainError, "top-ranked.*safety floors"):
            select_candidate(
                [top_auc_below_recall_floor, lower_auc_passing], SafetyCriteria()
            )

    def test_default_risk_profile_exposes_finite_sample_limit(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.false_routine_risk, DEFAULT_FALSE_ROUTINE_RISK)
        self.assertEqual(args.false_priority_risk, DEFAULT_FALSE_PRIORITY_RISK)
        self.assertEqual(args.delta, DEFAULT_DELTA)

        evidence = RiskProfile().provenance()
        bounds = evidence["zero_error_one_sided_clopper_pearson_upper_bounds"]
        self.assertAlmostEqual(bounds["false_ROUTINE_at_n_35"], 0.0820316358566705)
        self.assertAlmostEqual(bounds["false_PRIORITY_at_n_40"], 0.0721575245055146)
        self.assertEqual(
            evidence["five_percent_certifiable_with_zero_errors"],
            {
                "false_ROUTINE_at_n_35": False,
                "false_PRIORITY_at_n_40": False,
            },
        )

    def test_evaluation_command_is_complete_sequential_split_bound(self) -> None:
        candidate = self.make_candidate("root", weights=b"adapter")
        manifest, _rows = self.make_manifest("val", "v")
        command = evaluation_command(
            "python-test",
            candidate,
            manifest,
            "val",
            self.root / "out.json",
            self.sources,
            20,
        )
        self.assertNotIn("--limit", command)
        self.assertEqual(command[command.index("--sampling") + 1], "sequential")
        self.assertEqual(command[command.index("--expected-split") + 1], "val")
        self.assertEqual(command[command.index("--task") + 1], "escalation")

    def test_reuse_requires_a_complete_hash_bound_report(self) -> None:
        candidate = self.make_candidate("root", weights=b"adapter")
        manifest, rows = self.make_manifest("val", "v")
        report = self.make_decision_report(candidate, manifest, rows, "val")
        evidence = build_decision_evidence(
            report_path=report,
            candidate=candidate,
            input_binding=input_content_binding(manifest, "val"),
        )
        atomic_write_json(decision_evidence_path(report), evidence)
        calls = []

        validation, action = reuse_or_evaluate(
            candidate=candidate,
            manifest=manifest,
            expected_split="val",
            output=report,
            config=self.sources,
            python_executable="python-test",
            progress_every=20,
            runner=lambda command, cwd: calls.append((command, cwd)),
        )
        self.assertEqual(action, "reused_after_full_validation")
        self.assertEqual(validation["metrics"]["roc_auc"], 1.0)
        self.assertEqual(calls, [])

        payload = json.loads(report.read_text())
        payload["results"].pop()
        report.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "results for"):
            reuse_or_evaluate(
                candidate=candidate,
                manifest=manifest,
                expected_split="val",
                output=report,
                config=self.sources,
                python_executable="python-test",
                progress_every=20,
                runner=lambda command, cwd: calls.append((command, cwd)),
            )
        self.assertEqual(calls, [])

    def test_report_reuse_binds_adapter_config_images_and_tokens(self) -> None:
        candidate = self.make_candidate("root", weights=b"adapter")
        manifest, rows = self.make_manifest("val", "v")
        report = self.make_decision_report(candidate, manifest, rows, "val")
        atomic_write_json(
            decision_evidence_path(report),
            build_decision_evidence(
                report_path=report,
                candidate=candidate,
                input_binding=input_content_binding(manifest, "val"),
            ),
        )

        changed_config = json.loads(candidate.adapter_config_path.read_text())
        changed_config["r"] = 64
        candidate.adapter_config_path.write_text(
            json.dumps(changed_config), encoding="utf-8"
        )
        with self.assertRaisesRegex(PosttrainError, "evidence binding is stale"):
            reuse_or_evaluate(
                candidate=candidate,
                manifest=manifest,
                expected_split="val",
                output=report,
                config=self.sources,
                python_executable="python-test",
                progress_every=20,
            )

        # Restore the exact adapter evidence, then mutate an input image.
        atomic_write_json(
            decision_evidence_path(report),
            build_decision_evidence(
                report_path=report,
                candidate=candidate,
                input_binding=input_content_binding(manifest, "val"),
            ),
        )
        Path(rows[0]["image_path"]).write_bytes(b"changed-after-scoring")
        with self.assertRaisesRegex(PosttrainError, "evidence binding is stale"):
            reuse_or_evaluate(
                candidate=candidate,
                manifest=manifest,
                expected_split="val",
                output=report,
                config=self.sources,
                python_executable="python-test",
                progress_every=20,
            )

        # A self-consistent probability report still fails if its class token
        # is not the one recorded during training.
        Path(rows[0]["image_path"]).write_bytes(b"test-image-placeholder")
        payload = json.loads(report.read_text())
        payload["run"]["token_contract"]["PRIORITY"]["token_id"] = 11
        report.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "token differs"):
            reuse_or_evaluate(
                candidate=candidate,
                manifest=manifest,
                expected_split="val",
                output=report,
                config=self.sources,
                python_executable="python-test",
                progress_every=20,
            )

    def test_four_way_separation_rejects_content_duplicate(self) -> None:
        train_manifest, train = self.make_manifest("train", "t")
        val_manifest, val = self.make_manifest("val", "v")
        calibration_manifest, calibration = self.make_manifest(
            "calibration", "c"
        )
        eval_manifest, evaluation = self.make_manifest("eval", "e")
        # Distinct IDs and paths still leak identical source bytes.
        Path(val[0]["image_path"]).write_bytes(Path(train[0]["image_path"]).read_bytes())
        with self.assertRaisesRegex(PosttrainError, "image_content_sha256"):
            validate_partition_separation(
                {
                    "train": (train_manifest, train),
                    "val": (val_manifest, val),
                    "calibration": (calibration_manifest, calibration),
                    "eval": (eval_manifest, evaluation),
                }
            )

    def test_selection_report_is_immutable_but_repeatably_reusable(self) -> None:
        path = self.root / "checkpoint-selection.json"
        expected = with_integrity(
            {
                "artifact_type": "selection",
                "generated_at": "first",
                "candidates": [{"report_action": "evaluated_now", "id": "root"}],
                "selected": {"id": "root"},
            }
        )
        observed, action = reuse_or_write_selection(path, expected)
        self.assertEqual(action, "written_now")
        original_bytes = path.read_bytes()

        rerun = with_integrity(
            {
                "artifact_type": "selection",
                "generated_at": "later",
                "candidates": [
                    {"report_action": "reused_after_full_validation", "id": "root"}
                ],
                "selected": {"id": "root"},
            }
        )
        reused, action = reuse_or_write_selection(path, rerun)
        self.assertEqual(action, "reused_after_full_validation")
        self.assertEqual(reused, observed)
        self.assertEqual(path.read_bytes(), original_bytes)

        changed = with_integrity(
            {
                "artifact_type": "selection",
                "generated_at": "later",
                "candidates": [{"report_action": "reused", "id": "other"}],
                "selected": {"id": "other"},
            }
        )
        with self.assertRaisesRegex(PosttrainError, "immutable checkpoint selection"):
            reuse_or_write_selection(path, changed)

    def test_policy_reuse_is_bound_to_requested_risk_profile(self) -> None:
        candidate = self.make_candidate("root", weights=b"adapter")
        calibration_manifest, calibration_rows = self.make_manifest(
            "calibration", "c"
        )
        evaluation_manifest, evaluation_rows = self.make_manifest("eval", "e")
        calibration_report = self.make_decision_report(
            candidate, calibration_manifest, calibration_rows, "calibration"
        )
        evaluation_report = self.make_decision_report(
            candidate, evaluation_manifest, evaluation_rows, "eval"
        )
        profile = RiskProfile()
        policy = build_policy_report(
            calibration_report=calibration_report,
            evaluation_report=evaluation_report,
            calibration_manifest=calibration_manifest,
            evaluation_manifest=evaluation_manifest,
            false_routine_risk=profile.false_routine_risk,
            false_priority_risk=profile.false_priority_risk,
            delta=profile.delta,
        )
        policy_path = self.root / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")

        validated = validate_policy_report(
            policy_path,
            calibration_report,
            evaluation_report,
            calibration_manifest,
            evaluation_manifest,
            candidate,
            profile,
        )
        self.assertEqual(validated["policy"]["per_gate_delta"], DEFAULT_DELTA)

        tampered_core = {
            key: value for key, value in json.loads(json.dumps(policy)).items()
            if key != "integrity"
        }
        tampered_core["policy"]["routine_if_score_strictly_less_than"] = 0.123456
        policy_path.write_text(json.dumps(with_integrity(tampered_core)), encoding="utf-8")
        with self.assertRaisesRegex(PosttrainError, "fresh deterministic rebuild"):
            validate_policy_report(
                policy_path,
                calibration_report,
                evaluation_report,
                calibration_manifest,
                evaluation_manifest,
                candidate,
                profile,
            )

        policy_path.write_text(json.dumps(policy), encoding="utf-8")

        with self.assertRaisesRegex(PosttrainError, "stale policy reuse"):
            validate_policy_report(
                policy_path,
                calibration_report,
                evaluation_report,
                calibration_manifest,
                evaluation_manifest,
                candidate,
                RiskProfile(false_routine_risk=0.20),
            )


if __name__ == "__main__":
    unittest.main()
