from pathlib import Path
import sys
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from evaluate_hybrid_runtime import (  # noqa: E402
    attention_metadata,
    decision_trace_metadata,
    summarize,
)


class HybridEvaluatorTests(unittest.TestCase):
    def test_decision_trace_is_enumerated_and_fail_closed(self) -> None:
        confirmed = {
            "status": "RETAKE",
            "decision_trace": {
                "specialist": "RETAKE candidate",
                "gemma": "Confirmed",
                "policy": "RETAKE",
            },
        }
        self.assertEqual(decision_trace_metadata(confirmed)["gemma"], "Confirmed")
        confirmed["decision_trace"]["policy"] = "READY"
        with self.assertRaises(ValueError):
            decision_trace_metadata(confirmed)

        abstained = {
            "status": "LIMITED",
            "decision_trace": {
                "specialist": "Abstained",
                "gemma": "Skipped",
                "policy": "LIMITED",
            },
        }
        self.assertEqual(decision_trace_metadata(abstained)["gemma"], "Skipped")

    def test_attention_schema_is_local_png_only(self) -> None:
        payload = {
            "quality_attention": {
                "label": "Model quality attention \u2014 not pathology localization.",
                "factor": "clarity",
                "factor_label": "Clarity",
                "method": "factor-grad-cam",
                "image_data_url": "data:image/png;base64,cG5n",
            }
        }
        self.assertEqual(attention_metadata(payload)["factor"], "clarity")
        payload["quality_attention"]["image_data_url"] = "https://example.test/map.png"
        with self.assertRaises(ValueError):
            attention_metadata(payload)

    def test_attention_is_counted_only_on_retake(self) -> None:
        records = [
            {
                "patient_id": "a",
                "truth": "RETAKE",
                "decision": "RETAKE",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 10,
                "api_latency_ms": 9,
                "attention_present": True,
                "attention_method": "factor-grad-cam",
            },
            {
                "patient_id": "b",
                "truth": "READY",
                "decision": "READY",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 10,
                "api_latency_ms": 9,
                "attention_present": False,
                "attention_method": None,
            },
        ]
        attention = summarize(records)["quality_attention"]
        self.assertEqual(attention["present"], 1)
        self.assertEqual(attention["retake_decisions"], 1)
        self.assertEqual(attention["unexpected_non_retake"], 0)
        self.assertEqual(attention["retake_without_attention"], 0)

    def test_abstentions_and_failures_stay_in_denominator(self) -> None:
        records = [
            {
                "patient_id": "a",
                "truth": "READY",
                "decision": "READY",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 100,
                "api_latency_ms": 90,
            },
            {
                "patient_id": "b",
                "truth": "RETAKE",
                "decision": "RETAKE",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 110,
                "api_latency_ms": 100,
            },
            {
                "patient_id": "c",
                "truth": "RETAKE",
                "decision": "LIMITED",
                "flow": "specialist_limited",
                "wall_latency_ms": 20,
                "api_latency_ms": 15,
            },
            {
                "patient_id": "d",
                "truth": "READY",
                "decision": "LIMITED",
                "flow": "http_or_schema_failure",
                "wall_latency_ms": 40000,
                "api_latency_ms": None,
            },
        ]
        metrics = summarize(records)
        self.assertEqual(metrics["images"], 4)
        self.assertEqual(metrics["decision_counts"]["LIMITED"], 2)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["accepted_accuracy"], 1.0)
        self.assertEqual(metrics["flow"]["http_or_schema_failure"], 1)
        self.assertEqual(metrics["patient_event_metrics"]["false_ready_patient_count"], 0)

    def test_false_decisions_count_at_image_and_patient_level(self) -> None:
        records = [
            {
                "patient_id": "a",
                "truth": "RETAKE",
                "decision": "READY",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 1,
                "api_latency_ms": 1,
            },
            {
                "patient_id": "a",
                "truth": "RETAKE",
                "decision": "READY",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 1,
                "api_latency_ms": 1,
            },
            {
                "patient_id": "b",
                "truth": "READY",
                "decision": "RETAKE",
                "flow": "gemma_confirmed",
                "wall_latency_ms": 1,
                "api_latency_ms": 1,
            },
        ]
        metrics = summarize(records)
        self.assertEqual(metrics["false_ready_count"], 2)
        self.assertEqual(metrics["false_retake_count"], 1)
        self.assertEqual(metrics["patient_event_metrics"]["false_ready_patient_count"], 1)
        self.assertEqual(metrics["patient_event_metrics"]["false_retake_patient_count"], 1)


if __name__ == "__main__":
    unittest.main()
