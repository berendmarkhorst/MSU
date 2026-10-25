"""Synthetic checks; no study files are read."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
import brain


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.travel = pd.DataFrame([[0, 600, 3600], [900, 0, 300], [4200, 480, 0]],
                                   index=list("ABC"), columns=list("ABC"))
        self.strokes = pd.Series([100., 50., 40.], index=list("ABC"))
        self.params = replace(brain.DEFAULT, ischemic_fraction=1.)

    def evaluate(self, **kwargs):
        return brain.evaluate_policy("A", 100, self.travel, self.strokes, ["B", "C"],
                                     params=self.params, **kwargs)

    def test_curve_knots(self):
        np.testing.assert_allclose(brain.outcome_percent([20, 100, 140, 270]), [31, 22.5, 19.5, 15.5])

    def test_single_location_against_hand_calculation(self):
        travel = pd.DataFrame([[0.]], index=["A"], columns=["A"])
        result = brain.evaluate_policy("A", 0, travel, pd.Series([100.], index=["A"]), ["A"])
        # Conventional OTT=104, MSU OTT=65, service=45 minutes; 86 outcome patients.
        load = 100 * 45 / 525_600
        conventional = (22.5 - 4 * (22.5 - 19.5) / 40) / 100
        msu = (31 - 45 * (31 - 22.5) / 80) / 100
        self.assertAlmostEqual(result["busy_probability"], load / (1 + load))
        self.assertAlmostEqual(result["conventional_outcomes"], 86 * conventional)
        self.assertAlmostEqual(result["additional_outcomes"], 86 * (msu - conventional) / (1 + load))

    def test_curve_is_horizontal_after_270(self):
        np.testing.assert_allclose(brain.outcome_percent([270, 271, 400, 1000]), 15.5)

    def test_late_patients_remain_in_both_outcomes(self):
        travel = pd.DataFrame([[0, 18000], [18000, 0]], index=list("AB"), columns=list("AB"))
        x = brain.evaluate_policy("A", 100, travel, pd.Series([10., 20.], index=list("AB")),
                                  ["B"], params=self.params)
        self.assertEqual(x["modelled_patients"], 30)
        # At A, conventional OTT=404, and its contribution is 10*15.5%, not zero.
        self.assertAlmostEqual(x["conventional_outcomes"], 10*.155+20*brain.outcome_percent(104)/100)
        self.assertGreater(x["additional_outcomes"], 0)

    def test_no_gain_when_both_pathways_are_on_flat_tail(self):
        p = replace(self.params, onset_to_alarm_min=300)
        x = brain.evaluate_policy("A", 100, self.travel, self.strokes, ["B"], params=p)
        self.assertAlmostEqual(x["additional_outcomes"], 0)
        self.assertGreater(x["offered_calls"], 0)

    def test_no_treatment_uptake_factor(self):
        x = brain.evaluate_policy("A", 100, self.travel, self.strokes, ["B"])
        self.assertAlmostEqual(x["modelled_patients"], 190*.86)

    def test_population_fraction_changes_outcomes_not_workload(self):
        x = self.evaluate()
        y = brain.evaluate_policy("A", 100, self.travel, self.strokes, ["B", "C"],
                                  params=replace(self.params, ischemic_fraction=.25))
        self.assertAlmostEqual(y["expected_outcomes"], .25*x["expected_outcomes"])
        self.assertEqual(y["busy_probability"], x["busy_probability"])

    def test_extra_calls_change_only_workload(self):
        x = self.evaluate()
        for multiplier in [1.5, 2., 3.]:
            y = self.evaluate(call_multiplier=multiplier)
            for key in ["modelled_patients", "registered_strokes", "conventional_outcomes", "mean_service_min"]:
                self.assertAlmostEqual(x[key], y[key])
            load = multiplier*x["offered_load"]
            self.assertAlmostEqual(y["busy_probability"], load/(1+load))
            self.assertAlmostEqual(y["additional_outcomes"], x["additional_outcomes"]*(1+x["offered_load"])/(1+load))

    def test_zero_calls(self):
        x = self.evaluate(call_multiplier=0)
        self.assertEqual(x["busy_probability"], 0)
        self.assertEqual(x["served_calls"], 0)

    def test_slower_msu_is_not_dispatched(self):
        x = self.evaluate()
        self.assertEqual(x["dispatchable_strokes"], 150)

    def test_response_uses_station_to_patient_direction(self):
        p = brain.postcode_benefits("A", 100, self.travel, self.strokes, ["B"], ["A"], self.params)
        np.testing.assert_allclose(p.conventional_ott_min, [106, 106, 164])

    def test_hospital_return_leg_is_directional(self):
        p = brain.postcode_benefits("A", 100, self.travel, self.strokes, ["B", "C"], params=self.params)
        self.assertEqual(p.set_index("postcode").loc["C", "service_min"], 175)

    def test_tied_destination_controls_return_leg(self):
        travel = pd.DataFrame([[0, 120, 120], [300, 0, 60], [600, 60, 0]], index=list("ABC"), columns=list("ABC"))
        p = brain.postcode_benefits("A", 10, travel, self.strokes, ["C", "B"], params=self.params)
        first = p.iloc[0]
        self.assertEqual(first.hospital, "C")
        self.assertEqual(first.service_min, 57)  # 0 + 30 + 2 + 15 + 10.

    def test_reordered_labels(self):
        x = self.evaluate()
        y = brain.evaluate_policy("A", 100, self.travel[list("CAB")], self.strokes[list("BCA")],
                                  ["B", "C"], params=self.params)
        self.assertEqual(x, y)

    def test_optimizer_matches_evaluation_for_all_call_scenarios(self):
        for multiplier in [1., 1.5, 2., 3.]:
            kwargs = dict(hospitals=["B", "C"], ambulance_bases=["A"], params=self.params, call_multiplier=multiplier)
            grid = brain.optimize_policy(self.travel, self.strokes, [0, 18, 50, 100], **kwargs)
            for row in grid.to_dict("records"):
                result = brain.evaluate_policy(row["base"], row["radius_min"], self.travel, self.strokes, **kwargs)
                self.assertEqual(row, result)

    def test_postcode_gains_sum_to_policy_gain(self):
        p = brain.postcode_benefits("A", 100, self.travel, self.strokes, ["B", "C"], params=self.params)
        self.assertAlmostEqual(p.additional_outcomes.sum(), self.evaluate()["additional_outcomes"])
        np.testing.assert_allclose(p.additional_outcomes, p.modelled_patients*p.benefit_per_100_ischemic_strokes/100)

    def test_missing_locations_rejected(self):
        for hospitals in [[], ["missing"]]:
            with self.assertRaises(ValueError):
                brain.evaluate_policy("A", 10, self.travel, self.strokes, hospitals)
        with self.assertRaises(ValueError): self.evaluate(ambulance_bases=[])

    def test_invalid_numeric_inputs_rejected(self):
        for multiplier in [-1, float("nan")]:
            with self.assertRaises(ValueError): self.evaluate(call_multiplier=multiplier)
        with self.assertRaises(ValueError): brain.Parameters(ischemic_fraction=1.1)
        with self.assertRaises(ValueError): brain.Parameters(minutes_per_year=0)

    def test_deterministic_policy_ties(self):
        zero = pd.DataFrame(0., index=list("BA"), columns=list("BA"))
        grid = brain.optimize_policy(zero, pd.Series(0., index=zero.index), [30, 18], ["A"])
        self.assertEqual((grid.iloc[0].base, grid.iloc[0].radius_min), ("A", 18))



if __name__ == "__main__":
    unittest.main()
