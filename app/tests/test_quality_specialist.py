import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from quality_specialist import QUALITY_ATTENTION_LABEL, QualitySpecialist


class SpecialistBundleTests(unittest.TestCase):
    def test_manifest_verifies_all_three_artifacts_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "backbone": root / "backbone.pth",
                "decision_head": root / "decision.pt",
                "factor_head": root / "factor.pt",
            }
            for name, path in paths.items():
                path.write_bytes(name.encode("utf-8"))
            manifest = {
                "schema_version": 1,
                **{
                    name: {
                        "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in paths.items()
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            self.assertTrue(
                QualitySpecialist._verify_bundle_manifest(
                    backbone_path=paths["backbone"],
                    decision_head_path=paths["decision_head"],
                    factor_head_path=paths["factor_head"],
                )
            )

            paths["decision_head"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                QualitySpecialist._verify_bundle_manifest(
                    backbone_path=paths["backbone"],
                    decision_head_path=paths["decision_head"],
                    factor_head_path=paths["factor_head"],
                )

    def test_quality_attention_is_factor_specific_embedded_png(self) -> None:
        specialist = QualitySpecialist.__new__(QualitySpecialist)
        specialist._torch = torch
        specialist._device = torch.device("cpu")

        class TinyBackbone:
            def __init__(self) -> None:
                convolution = torch.nn.Conv2d(3, 4, 1, bias=False)
                with torch.no_grad():
                    convolution.weight.fill_(0.25)
                self.features = convolution

        factor_head = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            factor_head.weight.zero_()
            # Clarity is output index 2. Negative weights make the inverse
            # quality target produce positive spatial evidence.
            factor_head.weight[2].fill_(-0.5)
        specialist._backbone = TinyBackbone()
        specialist._factor_heads = [factor_head]
        specialist._factor_mean = torch.zeros(4)
        specialist._factor_std = torch.ones(4)

        ramp = torch.linspace(-0.8, 0.8, 64).reshape(8, 8)
        tensor = torch.stack([ramp, ramp, ramp])
        payload = specialist._quality_attention(tensor, "clarity")

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["label"], QUALITY_ATTENTION_LABEL)
        self.assertEqual(payload["factor"], "clarity")
        self.assertEqual(payload["factor_label"], "Clarity")
        self.assertEqual(payload["method"], "factor-grad-cam")
        prefix, encoded = payload["image_data_url"].split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(image.format, "PNG")

    def test_attention_failure_is_non_fatal(self) -> None:
        specialist = QualitySpecialist.__new__(QualitySpecialist)
        with patch.object(
            QualitySpecialist,
            "_quality_attention",
            side_effect=RuntimeError("optional CAM unavailable"),
        ):
            self.assertIsNone(
                specialist._safe_quality_attention(object(), "artifact")
            )


if __name__ == "__main__":
    unittest.main()
