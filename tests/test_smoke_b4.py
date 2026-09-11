import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.smoke_test_b4 import CONFIG_PATH, load_config, run_smoke_test


class TestSmokeEfficientNetB4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(CONFIG_PATH)
        cls.report = run_smoke_test(config)

    def test_forward_pass_shape(self):
        self.assertEqual(
            self.report["output_shape"],
            [self.report["input_shape"][0], self.report["num_classes"]],
        )

    def test_softmax_rows_sum_to_one(self):
        for row_sum in self.report["probability_row_sums"]:
            self.assertAlmostEqual(row_sum, 1.0, places=3)

    def test_loss_is_finite(self):
        self.assertTrue(self.report["checks"]["finite_loss"])

    def test_gradients_are_finite(self):
        self.assertTrue(self.report["checks"]["finite_gradients"])

    def test_optimizer_step_applied(self):
        self.assertTrue(self.report["checks"]["optimizer_step"])

    def test_overall_status_pass(self):
        self.assertEqual(self.report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
