import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from evaluate_combined_pipeline import (  # noqa: E402
    DEFAULT_LOCK,
    EvaluationInputError,
    build_report,
    compute_metrics,
    escalation_decision,
    pipeline_state,
    quality_decision,
)


class CombinedPolicyTests(unittest.TestCase):
    def test_both_policies_are_strict_and_abstain_on_threshold_equality(self) -> None:
        quality_policy = {
            "ready_threshold_strictly_greater_than": 0.8,
            "retake_threshold_strictly_less_than": 0.2,
        }
        self.assertEqual(quality_decision(0.8, quality_policy), "LIMITED")
        self.assertEqual(quality_decision(0.2, quality_policy), "LIMITED")
        self.assertEqual(
            quality_decision(math.nextafter(0.8, math.inf), quality_policy), "READY"
        )
        self.assertEqual(
            quality_decision(math.nextafter(0.2, -math.inf), quality_policy), "RETAKE"
        )

        review_policy = {
            "routine_if_score_strictly_less_than": 0.1,
            "priority_if_score_strictly_greater_than": 0.9,
        }
        self.assertEqual(escalation_decision(0.1, review_policy), "UNCERTAIN")
        self.assertEqual(escalation_decision(0.9, review_policy), "UNCERTAIN")
        self.assertEqual(escalation_decision(0.09, review_policy), "ROUTINE")
        self.assertEqual(escalation_decision(0.91, review_policy), "PRIORITY")

    def test_quality_gate_blocks_retake_and_limited_before_escalation(self) -> None:
        for escalation in ("ROUTINE", "PRIORITY", "UNCERTAIN"):
            self.assertEqual(pipeline_state("RETAKE", escalation), "RETAKE")
            self.assertEqual(pipeline_state("LIMITED", escalation), "LIMITED")
        self.assertEqual(pipeline_state("READY", "ROUTINE"), "ROUTINE_REVIEW")
        self.assertEqual(pipeline_state("READY", "PRIORITY"), "PRIORITY_REVIEW")
        self.assertEqual(pipeline_state("READY", "UNCERTAIN"), "UNCERTAIN")

    def test_danger_and_workload_denominators_keep_blocked_and_uncertain_images(self) -> None:
        def record(
            patient: str,
            grade: int,
            quality_truth: str,
            quality_call: str,
            standalone: str,
            final: str,
        ) -> dict[str, object]:
            return {
                "patient_id": patient,
                "technical_quality_truth": quality_truth,
                "dr_grade": grade,
                "truth_review_priority": "PRIORITY" if grade >= 2 else "ROUTINE",
                "quality_decision": quality_call,
                "quality_gate_passed": quality_call == "READY",
                "cached_standalone_review_priority_decision": standalone,
                "final_pipeline_state": final,
            }

        records = [
            record("priority-a", 2, "READY", "READY", "ROUTINE", "ROUTINE_REVIEW"),
            record("priority-a", 3, "RETAKE", "LIMITED", "PRIORITY", "LIMITED"),
            record("priority-b", 4, "READY", "READY", "UNCERTAIN", "UNCERTAIN"),
            record("routine-a", 0, "READY", "READY", "PRIORITY", "PRIORITY_REVIEW"),
            record("routine-a", 1, "READY", "READY", "UNCERTAIN", "UNCERTAIN"),
            record("routine-b", 0, "RETAKE", "RETAKE", "ROUTINE", "RETAKE"),
        ]
        metrics = compute_metrics(records)
        image = metrics["image_level"]["full_pipeline"]
        self.assertEqual(image["false_routine_danger"]["count"], 1)
        self.assertEqual(image["false_routine_danger"]["denominator"], 3)
        self.assertEqual(image["false_priority_workload"]["count"], 1)
        self.assertEqual(image["false_priority_workload"]["denominator"], 3)
        self.assertEqual(sum(image["final_state_counts"].values()), 6)
        self.assertEqual(image["quality_blocked_count"], 2)
        self.assertEqual(image["downstream_uncertain_count"], 2)

        patient = metrics["patient_level"]["combined_adverse_events_over_image_decisions"]
        self.assertEqual(patient["false_routine_danger"]["count"], 1)
        self.assertEqual(patient["false_routine_danger"]["denominator"], 2)
        self.assertEqual(patient["false_priority_workload"]["count"], 1)
        self.assertEqual(patient["false_priority_workload"]["denominator"], 2)


class CombinedArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_pinned_full_report_joins_every_official_validation_image(self) -> None:
        self.assertEqual(self.report["evaluation_set"]["evaluated_images"], 400)
        self.assertEqual(self.report["evaluation_set"]["evaluated_patients"], 100)
        self.assertTrue(self.report["evaluation_set"]["all_images_retained_in_full_run"])
        self.assertEqual(len(self.report["records"]), 400)
        self.assertEqual(
            self.report["metrics"]["image_level"]["dr_grade_truth_counts"],
            {"0": 174, "1": 46, "2": 92, "3": 68, "4": 20},
        )

    def test_full_pipeline_reproduces_frozen_scores_and_quality_first_gate(self) -> None:
        image = self.report["metrics"]["image_level"]
        self.assertEqual(
            image["quality_gate"]["decision_counts"],
            {"READY": 21, "RETAKE": 23, "LIMITED": 356},
        )
        self.assertEqual(
            image["standalone_escalation_before_quality_gate"]["decision_counts"],
            {"ROUTINE": 99, "PRIORITY": 56, "UNCERTAIN": 245},
        )
        self.assertEqual(
            image["full_pipeline"]["final_state_counts"],
            {
                "RETAKE": 23,
                "LIMITED": 356,
                "ROUTINE_REVIEW": 4,
                "PRIORITY_REVIEW": 5,
                "UNCERTAIN": 12,
            },
        )
        self.assertEqual(
            sum(
                row["review_priority_stage_executed_in_simulated_pipeline"]
                for row in self.report["records"]
            ),
            21,
        )
        self.assertTrue(
            all(
                row["review_priority_stage_executed_in_simulated_pipeline"]
                or row["final_pipeline_state"] in {"RETAKE", "LIMITED"}
                for row in self.report["records"]
            )
        )

    def test_requested_safety_workload_coverage_and_gate_effect_metrics(self) -> None:
        full = self.report["metrics"]["image_level"]["full_pipeline"]
        self.assertEqual(
            (full["false_routine_danger"]["count"], full["false_routine_danger"]["denominator"]),
            (0, 180),
        )
        self.assertEqual(
            (full["false_priority_workload"]["count"], full["false_priority_workload"]["denominator"]),
            (0, 220),
        )
        self.assertEqual(full["quality_blocked_count"], 379)
        self.assertEqual(full["abstention_count"], 368)
        self.assertEqual(full["no_decisive_review_release_count"], 391)
        self.assertEqual(full["decisive_review_count"], 9)

        gate = self.report["metrics"][
            "quality_gate_effect_on_dataset_defined_priority"
        ]
        self.assertEqual(gate["blocked_priority_truth_images"], 167)
        self.assertEqual(gate["priority_truth_images_passed_to_escalation"], 13)
        self.assertEqual(
            gate["standalone_priority_review_decisions_blocked_by_quality"], 51
        )
        self.assertEqual(
            gate["priority_truth_patients_with_all_priority_images_quality_blocked"],
            39,
        )

        adverse = self.report["metrics"]["patient_level"][
            "combined_adverse_events_over_image_decisions"
        ]
        self.assertEqual(
            (adverse["false_routine_danger"]["count"], adverse["false_routine_danger"]["denominator"]),
            (0, 50),
        )
        self.assertEqual(
            (adverse["false_priority_workload"]["count"], adverse["false_priority_workload"]["denominator"]),
            (0, 60),
        )

    def test_full_report_is_hash_verified_and_runs_no_new_inference(self) -> None:
        provenance = self.report["provenance"]
        self.assertTrue(provenance["all_locked_inputs_verified"])
        self.assertTrue(
            all(entry["verified"] for entry in provenance["verified_inputs"].values())
        )
        self.assertTrue(provenance["reused_existing_image_level_scores"])
        self.assertFalse(provenance["new_model_inference_executed"])
        self.assertFalse(provenance["cloud_resources_used"])
        self.assertEqual(self.report["integrity_checks"]["locked_input_files_verified"], 11)
        self.assertTrue(
            self.report["integrity_checks"][
                "quality_report_prior_hash_attestation_verified"
            ]
        )
        self.assertTrue(
            self.report["integrity_checks"]["escalation_promotion_bindings_verified"]
        )

    def test_patient_review_queue_aggregation_fails_closed_on_any_blocker(self) -> None:
        aggregate = self.report["metrics"]["patient_level"][
            "fail_closed_review_queue_aggregate"
        ]
        self.assertEqual(
            aggregate["decision_counts"],
            {"ROUTINE_REVIEW": 0, "PRIORITY_REVIEW": 4, "UNCERTAIN": 96},
        )
        self.assertEqual(aggregate["false_routine_danger"]["denominator"], 50)
        self.assertEqual(aggregate["false_priority_workload"]["denominator"], 50)

    def test_changed_locked_hash_fails_closed(self) -> None:
        lock = json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))
        lock["inputs"]["quality_image_level_report"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            changed_lock = Path(temporary) / "changed-lock.json"
            changed_lock.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationInputError, "SHA-256 mismatch"):
                build_report(changed_lock)

    def test_smoke_subset_is_deterministic_but_validates_full_join_first(self) -> None:
        first = build_report(limit=8)
        second = build_report(limit=8)
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_set"]["evaluated_images"], 8)
        self.assertEqual(first["provenance"]["joined_full_cohort_rows"], 400)
        with self.assertRaisesRegex(EvaluationInputError, "complete patient bundles"):
            build_report(limit=5)


if __name__ == "__main__":
    unittest.main()
