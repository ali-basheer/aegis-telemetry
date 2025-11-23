import unittest
from src.physics.thermodynamics import ZeldovichEngine

class TestThermodynamics(unittest.TestCase):
    def test_low_temp_cutoff(self):
        # Should return minimal NOx at low temps
        limit = ZeldovichEngine.calculate_limit(20, 10)
        self.assertEqual(limit, 50.0)

    def test_high_load_scaling(self):
        # Should return higher values for high load
        limit = ZeldovichEngine.calculate_limit(90, 100)
        self.assertGreater(limit, 50.0)

if __name__ == '__main__':
    unittest.main()
