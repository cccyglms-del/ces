import unittest

from kmtool.analysis.reconstruction import reconstruct_arm_ipd
from kmtool.models import CurveSeries, RiskTableRow


class ReconstructionTestCase(unittest.TestCase):
    def test_reconstruct_with_risk_table_preserves_sample_size(self):
        curve = CurveSeries(
            curve_id="curve_1",
            arm_label="Treatment A",
            detected_color=(200, 30, 30),
            pixel_points=[(0, 0), (1, 1)],
            data_points=[(0.0, 1.0), (6.0, 0.82), (12.0, 0.61), (18.0, 0.43)],
            confidence=0.72,
        )
        risk_rows = [
            RiskTableRow(time=0.0, arm_counts={"Treatment A": 100}),
            RiskTableRow(time=6.0, arm_counts={"Treatment A": 82}),
            RiskTableRow(time=12.0, arm_counts={"Treatment A": 61}),
            RiskTableRow(time=18.0, arm_counts={"Treatment A": 43}),
        ]

        arm = reconstruct_arm_ipd(
            study_id="study-1",
            comparison_id="cmp-1",
            curve=curve,
            risk_rows=risk_rows,
            fallback_total_n=100,
        )

        self.assertEqual(arm.reconstruction_method, "guyot_interval_approx")
        self.assertEqual(len(arm.time), 100)
        self.assertGreater(sum(arm.event), 0)
        self.assertGreater(arm.confidence, curve.confidence)

    def test_reconstruct_without_risk_table_warns_and_uses_fallback(self):
        curve = CurveSeries(
            curve_id="curve_2",
            arm_label="Treatment B",
            detected_color=(30, 30, 200),
            pixel_points=[(0, 0), (1, 1)],
            data_points=[(0.0, 1.0), (4.0, 0.95), (8.0, 0.80), (12.0, 0.74)],
            confidence=0.60,
        )

        arm = reconstruct_arm_ipd(
            study_id="study-2",
            comparison_id="cmp-2",
            curve=curve,
            risk_rows=[],
            fallback_total_n=80,
        )

        self.assertEqual(arm.reconstruction_method, "interval_heuristic")
        self.assertEqual(len(arm.time), 80)
        self.assertTrue(any("heuristic fallback" in warning for warning in arm.warnings))


if __name__ == "__main__":
    unittest.main()
