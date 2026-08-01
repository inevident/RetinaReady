from pathlib import Path
import sys
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from evaluate_decision_logits import summarize  # noqa: E402
from train_qlora import (  # noqa: E402
    select_rows_for_run,
    target_for,
    task_contract,
    validate_task_label_tokens,
)


class FakeTokenizer:
    def encode(self, value: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        values = {
            "ROUTINE": [101, 201],
            "PRIORITY": [102, 201],
        }
        return values[value]


class MultitaskQLoRATests(unittest.TestCase):
    def test_escalation_contract_is_clinical_priority_not_capture_quality(self) -> None:
        contract = task_contract("escalation")
        self.assertEqual(contract["label_field"], "escalation_label")
        self.assertEqual(contract["labels"], ("ROUTINE", "PRIORITY"))
        target = target_for(
            {"escalation_label": "PRIORITY"}, task="escalation"
        )
        self.assertEqual(target["decision"], "PRIORITY")
        self.assertIn("clinician", target["next_step"])
        self.assertIn("not a diagnosis", target["disclaimer"])

    def test_escalation_label_tokens_must_diverge_on_first_token(self) -> None:
        result = validate_task_label_tokens(FakeTokenizer(), "escalation")
        self.assertEqual(result["ROUTINE"]["first_token_id"], 101)
        self.assertEqual(result["PRIORITY"]["first_token_id"], 102)

    def test_stratified_escalation_sampling_balances_both_labels(self) -> None:
        rows = [
            {"escalation_label": label, "image_id": f"{label}-{index}"}
            for label in ("ROUTINE", "PRIORITY")
            for index in range(8)
        ]
        selected = select_rows_for_run(
            rows, 10, seed=4, stratified=True, task="escalation"
        )
        self.assertEqual(
            sum(row["escalation_label"] == "ROUTINE" for row in selected),
            5,
        )
        self.assertEqual(
            sum(row["escalation_label"] == "PRIORITY" for row in selected),
            5,
        )

    def test_escalation_summary_counts_dangerous_false_routine(self) -> None:
        results = [
            {
                "truth": "PRIORITY",
                "prediction": "ROUTINE",
                "positive_probability": 0.2,
                "latency_ms": 10,
            },
            {
                "truth": "ROUTINE",
                "prediction": "ROUTINE",
                "positive_probability": 0.1,
                "latency_ms": 10,
            },
        ]
        metrics = summarize(results, 0.1, task="escalation")["metrics"]
        self.assertEqual(metrics["positive_label"], "PRIORITY")
        self.assertEqual(metrics["false_negative_count"], 1)
        self.assertEqual(metrics["false_negative_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
