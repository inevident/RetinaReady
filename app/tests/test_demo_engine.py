import unittest

from demo_engine import SUPPORTED_STATUSES, analyze_demo


class DemoEngineTests(unittest.TestCase):
    def test_each_explicit_scenario_is_supported(self) -> None:
        for status in SUPPORTED_STATUSES:
            with self.subTest(status=status):
                result = analyze_demo(
                    b"sample-image",
                    filename="capture.jpg",
                    content_type="image/jpeg",
                    scenario=status,
                )
                self.assertEqual(result["status"], status)
                self.assertIn("No diagnosis", result["disclaimer"])

    def test_non_image_is_unsupported(self) -> None:
        result = analyze_demo(
            b"not-an-image",
            filename="notes.txt",
            content_type="text/plain",
        )
        self.assertEqual(result["status"], "UNSUPPORTED")

    def test_arbitrary_image_fails_closed_without_model(self) -> None:
        arguments = {
            "filename": "capture.jpg",
            "content_type": "image/jpeg",
        }
        first = analyze_demo(b"same-image", **arguments)
        second = analyze_demo(b"same-image", **arguments)
        self.assertEqual(first["status"], second["status"])
        self.assertEqual(first["status"], "LIMITED")
        self.assertIsNone(first["confidence"])
        self.assertIn("not assessed", first["summary"])


if __name__ == "__main__":
    unittest.main()
