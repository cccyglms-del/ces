import unittest
from pathlib import Path

from PIL import Image

from kmtool.analysis.ingestion import detect_candidate_plot_bbox, detect_candidate_risk_bbox


class IngestionDetectionTestCase(unittest.TestCase):
    def test_detect_plot_bbox_excludes_title_and_risk_table(self):
        image = Image.open(Path("sample_data") / "km_minimal_sample.png").convert("RGB")
        left, top, right, bottom = detect_candidate_plot_bbox(image)

        self.assertGreater(left, 80)
        self.assertGreater(top, 80)
        self.assertLess(bottom, int(image.height * 0.80))
        self.assertGreater(right - left, int(image.width * 0.60))
        self.assertGreater(bottom - top, int(image.height * 0.45))

    def test_detect_risk_bbox_below_plot_region(self):
        image = Image.open(Path("sample_data") / "km_minimal_sample.png").convert("RGB")
        plot_bbox = detect_candidate_plot_bbox(image)
        risk_bbox = detect_candidate_risk_bbox(image, plot_bbox)

        self.assertIsNotNone(risk_bbox)
        self.assertGreater(risk_bbox[1], plot_bbox[3] - 4)
        self.assertGreater(risk_bbox[3] - risk_bbox[1], 30)


if __name__ == "__main__":
    unittest.main()
