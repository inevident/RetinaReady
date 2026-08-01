import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from audit_retfound_green_license import build_report  # noqa: E402


LICENSE_TEXT = b"""Justin's Custom Non-Commercial Research Licence (CNCRL)
You may use this only for non-commercial research.
Industry-Involved Project includes one where a Commercial Entity provides funding, resources or personnel.
Such users must obtain prior written permission from the Licensor.
"""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class RETFoundGreenLicenseAuditTests(unittest.TestCase):
    def make_config(self, root: Path) -> dict:
        manifests = {}
        for name in ("train", "val", "calibration", "eval"):
            path = root / f"{name}.csv"
            payload = f"split,{name}\n".encode()
            path.write_bytes(payload)
            manifests[name] = {"path": path.name, "sha256": digest(payload)}
        control = root / "control.json"
        control_payload = b'{"status":"experimental"}\n'
        control.write_bytes(control_payload)
        return {
            "experiment": "test",
            "upstream": {
                "project_name": "RETFound-Green",
                "project_repository": "https://example.test/repo",
                "repository_revision": "abc123",
                "paper_url": "https://example.test/paper",
                "license_name": "CNCRL",
                "license_raw_url": "https://example.test/LICENSE",
                "license_sha256": digest(LICENSE_TEXT),
                "required_license_signatures": [
                    "only for non-commercial research",
                    "Industry-Involved Project",
                    "provides funding, resources or personnel",
                    "prior written permission",
                ],
                "weights_release": "v0.1",
                "weights_url": "https://example.test/weights.pth",
                "architecture": "test backbone",
                "normalization": "test normalization",
            },
            "project_context": {
                "event_name": "funded event",
                "event_url": "https://example.test/event",
                "commercial_entity_involvement": True,
                "evidence": [],
                "prior_written_permission": {
                    "obtained": False,
                    "evidence_path": None,
                    "evidence_sha256": None,
                },
            },
            "frozen_inputs": {
                "manifests": manifests,
                "control_report": {
                    "path": control.name,
                    "sha256": digest(control_payload),
                },
            },
            "local_weight_path": "weights.pth",
        }

    def test_industry_involvement_without_permission_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_report(
                self.make_config(root),
                license_payload=LICENSE_TEXT,
                project_root=root,
            )
        self.assertEqual(report["status"], "BLOCKED_LICENSE_INCOMPATIBLE")
        self.assertFalse(report["permitted_to_download_or_train"])
        self.assertTrue(report["frozen_input_audit"]["passed"])
        self.assertFalse(report["execution"]["weight_download_attempted_by_this_audit"])
        self.assertFalse(report["execution"]["training_started_by_this_audit"])

    def test_verified_written_permission_can_unblock_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            evidence = root / "permission.txt"
            evidence.write_text("signed permission", encoding="utf-8")
            config["project_context"]["prior_written_permission"] = {
                "obtained": True,
                "evidence_path": evidence.name,
                "evidence_sha256": digest(evidence.read_bytes()),
            }
            report = build_report(
                config,
                license_payload=LICENSE_TEXT,
                project_root=root,
            )
        self.assertEqual(report["status"], "READY_FOR_EXPERIMENT_SETUP")
        self.assertTrue(report["permitted_to_download_or_train"])

    def test_tampered_license_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_report(
                self.make_config(root),
                license_payload=LICENSE_TEXT + b"tampered",
                project_root=root,
            )
        self.assertEqual(report["status"], "AUDIT_FAILED_FAIL_CLOSED")
        self.assertFalse(report["permitted_to_download_or_train"])

    def test_changed_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            (root / "eval.csv").write_text("changed", encoding="utf-8")
            report = build_report(
                config,
                license_payload=LICENSE_TEXT,
                project_root=root,
            )
        self.assertEqual(report["status"], "AUDIT_FAILED_FAIL_CLOSED")
        self.assertFalse(report["frozen_input_audit"]["passed"])


if __name__ == "__main__":
    unittest.main()
