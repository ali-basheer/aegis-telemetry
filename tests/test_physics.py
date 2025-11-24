"""
UNIT TEST: PHYSICS ENGINE VERIFICATION
PROFILE: FORD 6.7L POWERSTROKE

DESCRIPTION:
    Validates the math kernels before deployment.
    Ensures that the Zeldovich Solver, EKF, and Stoichiometry modules
    return realistic values for a V8 Turbo Diesel.
"""

import unittest
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics.combustion import ForensicCombustionValidator
from src.physics.stoichiometry import ClosedLoopAuditor
from src.physics.sensor_fusion import SensorIntegrityMonitor

class TestFordPhysics(unittest.TestCase):
    
    def setUp(self):
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        self.combustion = ForensicCombustionValidator()
        self.stoich = ClosedLoopAuditor()
        self.ekf = SensorIntegrityMonitor()

    def test_idle_condition(self):
        """
        Case 1: Warm Idle (600 RPM, Low Load).
        Expectation: Very low NOx, Lean mixture.
        """
        frame = {
            'rpm': 600,
            'load': 15.0,
            'temp': 85.0, # Coolant C
            'maf': 18.0,  # g/s (V8 breathes heavy even at idle)
            'fuel_rate': 1.2, # L/h
            'actual_nox': 50.0
        }
        
        # 1. Check Combustion (Zeldovich)
        # At idle, combustion temp is low -> Minimal thermal NOx
        res_comb = self.combustion.validate_snapshot(frame)
        self.assertLess(res_comb['physics_nox_ppm'], 200.0, "Idle NOx should be low")
        
        # 2. Check Stoichiometry
        # Diesel Idle is VERY lean (Lambda > 4.0)
        res_stoich = self.stoich.audit_cycle(frame)
        self.assertGreater(res_stoich['lambda_calc'], 2.0, "Diesel idle must be lean")

    def test_highway_cruise(self):
        """
        Case 2: Highway (1800 RPM, 40% Load).
        Expectation: Moderate NOx, Peak Volumetric Efficiency.
        """
        frame = {
            'rpm': 1800,
            'load': 40.0,
            'temp': 90.0,
            'maf': 120.0, 
            'fuel_rate': 12.0, 
            'actual_nox': 350.0,
            'baro': 100.0
        }
        
        # Check EKF (Sensor Fusion)
        # The predicted MAF should match the sensor MAF roughly
        res_ekf = self.ekf.audit_frame(frame, 0.1)
        self.assertLess(res_ekf['nis_score'], 3.0, "EKF should match generic highway flow")

    def test_rolling_coal_detection(self):
        """
        Case 3: The "Tuner" Scenario (Full Load, Rich Mixture).
        Expectation: Lambda < 1.1 should trigger Alert.
        """
        frame = {
            'rpm': 2500,
            'load': 100.0,
            'temp': 95.0,
            'maf': 250.0, 
            'fuel_rate': 95.0, # Massive fuel dump (Tuner)
            'actual_nox': 800.0
        }
        
        res = self.stoich.audit_cycle(frame)
        self.assertIn("NON_COMPLIANT", res['stoich_status'], "Must detect Rolling Coal")
        self.assertLess(res['lambda_calc'], 1.1, "Rich condition confirmed")

if __name__ == '__main__':
    print("--- VERIFYING FORD 6.7L PHYSICS KERNELS ---")
    unittest.main()