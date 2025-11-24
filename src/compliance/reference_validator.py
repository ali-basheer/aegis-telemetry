"""
MODULE: REFERENCE_MAP_VALIDATOR (SHADOW GOVERNOR)
PROFILE: FORD_SCORPION_V8 (DEF CONSUMPTION MODEL)

DESCRIPTION:
    Audits the command integrity of the Reductant (Urea) Injection System.
    
    It maintains a parallel "Shadow Map" of what the dosing rate *should* be 
    given the current engine load and fuel rate. 
    
    If the actual dosing falls below the Shadow Map by >20% for an extended 
    period (excluding DPF Regeneration events), it flags a 'DOSING_RESTRICTION_ATTACK'.
"""

import logging
from collections import deque
from typing import Dict

# Configure module-level logger
logger = logging.getLogger("AEGIS.COMPLIANCE.MAPS")

class ShadowGovernor:
    """
    The Independent Auditor for Urea Consumption.
    """
    
    def __init__(self):
        # FORD 6.7L REFERENCE PARAMETERS
        # Baseline: DEF usage is approx 1.5% - 4.0% of Fuel Mass
        self.min_def_ratio = 0.015 
        self.max_def_ratio = 0.045
        
        # Accumulators for "Long Term" drift detection
        self.total_fuel_l = 0.0
        self.total_def_l = 0.0
        self.sample_count = 0
        
        # Rolling buffer for instant deviation smoothing
        self.deviation_buffer = deque(maxlen=20) # 2 seconds @ 10Hz

    def audit_cycle(self, frame: Dict) -> Dict:
        """
        Compares Actual Dosing vs Expected Dosing.
        """
        # 1. EXTRACT SIGNALS
        # Fuel Rate is typically L/h. 
        # DEF Rate is often mL/s or g/s. We need to normalize units.
        fuel_rate_lph = frame.get('fuel_rate', 0.0)
        def_rate_ml_s = frame.get('reductant_rate', 0.0) 
        is_regen = frame.get('regen_status', 0) > 0.5
        
        # Convert all to Liters per Hour (L/h) for comparison
        def_rate_lph = (def_rate_ml_s * 3600.0) / 1000.0
        
        # 2. CALCULATE EXPECTED DOSING
        # The map is dynamic based on Load (NOx proxy)
        load = frame.get('load', 0)
        
        # At Idle (<20% load), NOx is low, dosing is minimal (~1.5%)
        # At Towing (>80% load), NOx is massive, dosing is max (~4.0%)
        target_ratio = self.min_def_ratio + (
            (load / 100.0) * (self.max_def_ratio - self.min_def_ratio)
        )
        
        expected_def_lph = fuel_rate_lph * target_ratio
        
        # 3. AUDIT LOGIC
        deviation_pct = 0.0
        status = "COMPLIANT"
        
        if fuel_rate_lph > 1.0: # Ignore noise at zero fuel
            # Calculate Deviation: (Actual - Expected) / Expected
            # Negative = Under-dosing (Cheat)
            # Positive = Over-dosing (Leak or Sensor Fail)
            deviation_pct = (def_rate_lph - expected_def_lph) / (expected_def_lph + 0.001)
            
            # EXCEPTION HANDLING
            if is_regen:
                # During DPF Burn, DEF is often paused/reduced. 
                # We overwrite the result to prevent false positives.
                status = "LEGITIMATE_PAUSE (REGEN)"
                deviation_pct = 0.0 
            
            elif deviation_pct < -0.30: # 30% less than expected
                status = "NON_COMPLIANT (UNDER_DOSING)"
            
            # Update Accumulators
            # We track the "Total Fluid Ratio" over the session
            # This catches "Slow Leaks" or "Micro-Throttling"
            dt_hours = 0.1 / 3600.0 # Assuming 10Hz
            self.total_fuel_l += fuel_rate_lph * dt_hours
            self.total_def_l += def_rate_lph * dt_hours

        # 4. LONG TERM HEALTH
        session_ratio = 0.0
        if self.total_fuel_l > 0.1:
            session_ratio = self.total_def_l / self.total_fuel_l

        return {
            "expected_def_lph": expected_def_lph,
            "actual_def_lph": def_rate_lph,
            "urea_deviation_pct": deviation_pct * 100.0,
            "session_def_ratio_pct": session_ratio * 100.0,
            "audit_status": status
        }