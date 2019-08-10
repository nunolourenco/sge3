import unittest
import warnings
import math
import sge.utilities.protected_math as protected_math


class TestUtilities(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=DeprecationWarning)

    def test_sqrt_positive(self):
        self.assertEqual(protected_math._sqrt_(4), 2, "Error: ValueError")

    def test_sqrt_negative(self):
        self.assertEqual(protected_math._sqrt_(-4), 2, "Error: ValueError")

    def test_log_pos(self):
        self.assertEqual(protected_math._log_(10), math.log(10), "Error: ValueError")

    def test_log_zero(self):
        self.assertEqual(protected_math._log_(0), 0, "Error: ValueError")

    def test_log_negative(self):
        self.assertEqual(protected_math._log_(-1), 0, "Error: ValueError")

    def test_division(self):
        self.assertEqual(protected_math._div_(1, 2), 0.5, "Error: ValueError")

    def test_division_neg(self):
        self.assertEqual(protected_math._div_(-1, 2), -0.5, "Error: ValueError")

    def test_division_zero(self):
        self.assertEqual(protected_math._div_(-1, 0), 1, "Error: ValueError")


if __name__ == '__main__':
    unittest.main()
