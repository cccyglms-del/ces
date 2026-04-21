import unittest

from kmtool.analysis.indirect import compute_bucher_indirect, reported_hr_to_effect
from kmtool.models import IndirectComparisonRequest


class IndirectComparisonTestCase(unittest.TestCase):
    def test_bucher_indirect_combines_log_hazard_ratios(self):
        ab_effect = reported_hr_to_effect(
            study_id="ab-1",
            comparison_label="A vs B",
            treatment_left="A",
            treatment_right="B",
            hr=0.80,
            ci_low=0.70,
            ci_high=0.92,
            source_method="reported_text",
            endpoint_text="overall survival",
            population_text="metastatic disease",
        )
        bc_effect = reported_hr_to_effect(
            study_id="bc-1",
            comparison_label="B vs C",
            treatment_left="B",
            treatment_right="C",
            hr=0.90,
            ci_low=0.80,
            ci_high=1.02,
            source_method="reported_text",
            endpoint_text="overall survival",
            population_text="metastatic disease",
        )
        request = IndirectComparisonRequest(
            treatment_a="A",
            treatment_b="B",
            treatment_c="C",
            endpoint="overall survival",
            selected_ab_studies=["ab-1"],
            selected_bc_studies=["bc-1"],
            population_hint="metastatic disease",
        )

        result = compute_bucher_indirect(request, [ab_effect], [bc_effect])

        self.assertAlmostEqual(result.ac_hr, 0.72, delta=0.04)
        self.assertLess(result.ci95[0], result.ac_hr)
        self.assertGreater(result.ci95[1], result.ac_hr)

    def test_bucher_blocks_inconsistent_population_without_override(self):
        ab_effect = reported_hr_to_effect(
            study_id="ab-2",
            comparison_label="A vs B",
            treatment_left="A",
            treatment_right="B",
            hr=0.80,
            ci_low=0.70,
            ci_high=0.92,
            source_method="reported_text",
            endpoint_text="overall survival",
            population_text="first line",
        )
        bc_effect = reported_hr_to_effect(
            study_id="bc-2",
            comparison_label="B vs C",
            treatment_left="B",
            treatment_right="C",
            hr=0.90,
            ci_low=0.80,
            ci_high=1.02,
            source_method="reported_text",
            endpoint_text="overall survival",
            population_text="refractory disease",
        )
        request = IndirectComparisonRequest(
            treatment_a="A",
            treatment_b="B",
            treatment_c="C",
            endpoint="overall survival",
            selected_ab_studies=["ab-2"],
            selected_bc_studies=["bc-2"],
            population_hint="first line",
            allow_inconsistent=False,
        )

        with self.assertRaises(ValueError):
            compute_bucher_indirect(request, [ab_effect], [bc_effect])


if __name__ == "__main__":
    unittest.main()
