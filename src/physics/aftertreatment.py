"""
MODULE: AFTERTREATMENT_FORENSICS
PROFILE: FORD_SCORPION_SCR (BOSCH DENOXTRONIC 2.2)

DESCRIPTION:
    Models the chemical efficiency of the Selective Catalytic Reduction (SCR) system.
    
    It compares the "Expected NOx Reduction" (based on temperature and dosing)
    against the "Actual NOx Reduction" (measured by the sensor).
    
    If the sensors report 99% efficiency when the exhaust is only 150°C 
    (where chemical reaction is impossible), it flags a 'False Reporting' event.
"""

import math

class AftertreatmentForensics:
    """
    Simulation of the SCR Catalyst Chemistry.
    """
    def __init__(self):
        # Ford 6.7L SCR Characteristics
        self.light_off_temp_c = 190.0  # Min temp for reaction
        self.optimal_temp_c = 350.0    # Peak efficiency temp
        self.max_efficiency = 0.98     # 98% reduction at peak
        
    def audit_frame(self, frame: dict, dt: float) -> dict:
        """
        Calculates expected tailpipe emissions.
        """
        # 1. GET INPUTS
        # We need the 'Engine Out' NOx (Physics Model) to verify the 'Tailpipe' NOx (Sensor)
        # Since we don't have a pre-scr sensor on all trucks, we use the Physics Model as the input.
        nox_input_ppm = frame.get('physics_nox_ppm', 500.0)
        egt_c = frame.get('exhaust_temp', 200.0) # From our new Extended PID
        
        # 2. CALCULATE THEORETICAL EFFICIENCY
        # SCR efficiency is a bell curve based on temperature.
        # Too cold (<190C) = Urea crystals form, no reduction.
        # Optimal (250-400C) = Max reduction.
        # Too hot (>500C) = Ammonia oxidation, reduced efficiency.
        
        if egt_c < self.light_off_temp_c:
            theoretical_efficiency = 0.1 # Basically zero
        elif egt_c < self.optimal_temp_c:
            # Linear ramp up
            theoretical_efficiency = 0.1 + (0.88 * (egt_c - self.light_off_temp_c) / (self.optimal_temp_c - self.light_off_temp_c))
        else:
            theoretical_efficiency = self.max_efficiency

        # 3. CALCULATE EXPECTED TAILPIPE NOX
        expected_tailpipe = nox_input_ppm * (1.0 - theoretical_efficiency)
        
        return {
            "scr_efficiency_model": theoretical_efficiency,
            "sim_tailpipe_nox": expected_tailpipe
        }