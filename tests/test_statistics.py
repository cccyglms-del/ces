import unittest

from kmtool.analysis.statistics import compute_pairwise_result
from kmtool.models import ReconstructedArmData


class StatisticsTestCase(unittest.TestCase):
    def test_pairwise_statistics_detect_worse_second_arm(self):
        arm_a = ReconstructedArmData(
            study_id="study",
            comparison_id="cmp",
            arm_label="Treatment A",
            time=[4] * 10 + [8] * 10 + [12] * 20,
            event=[1] * 20 + [0] * 20,
            source_curve_id="curve_a",
            reconstruction_method="test",
            confidence=0.8,
        )
        arm_b = ReconstructedArmData(
            study_id="study",
            comparison_id="cmp",
            arm_label="Treatment B",
            time=[2] * 14 + [5] * 12 + [12] * 14,
            event=[1] * 26 + [0] * 14,
            source_curve_id="curve_b",
            reconstruction_method="test",
            confidence=0.8,
        )

        result = compute_pairwise_result(arm_a, arm_b, comparison_id="cmp")

        self.assertLess(result.log_rank_p, 0.05)
        self.assertGreater(result.hr, 1.0)
        self.assertGreater(result.hr_ci_high, result.hr_ci_low)


if __name__ == "__main__":
    unittest.main()
