"""
MODULE: CLOSED_LOOP_AUDITOR (STOICHIOMETRY)
PROFILE: FORD_SCORPION_V8 (ULSD DIESEL)

DESCRIPTION:
    Audits the Air-Fuel Ratio (AFR) consistency.
    
    Unlike gasoline engines which target Lambda 1.0, Diesel engines operate 
    lean (Lambda > 1.0). This module calculates the 'Physical Lambda' 
    derived from Mass Air Flow (MAF) and Fuel Injection Rate.

    It detects two major forms of fraud:
    1. "Rolling Coal" (Lambda < 1.1): Injecting excess fuel for power/smoke.
    2. "O2 Masquerade": When the physical math proves the engine is running 
       rich, but the O2 sensor reports it is running lean (to fool the ECU).
"""

import logging

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS.STOICH")

class ClosedLoopAuditor:
    """
    Validates that the Air/Fuel mixture matches the reported sensor data.
    """
    
    def __init__(self):
        # PHYSICS CONSTANTS
        # Stoichiometric ratio for Ultra-Low Sulfur Diesel (ULSD)
        self.AFR_STOICH_DIESEL = 14.5 
        
        # THRESHOLDS
        # Lambda < 1.0 means "Rich" (Not possible in healthy modern diesel)
        # Lambda < 1.1 is the "Smoke Limit" (Soot formation spikes exponentially)
        self.SMOKE_LIMIT_LAMBDA = 1.1
        
        # Allowed variance between Calculated vs Sensor Lambda
        self.MAX_SENSOR_DEVIATION = 0.15 # 15% tolerance for transient lag

    def audit_cycle(self, frame: dict) -> dict:
        """
        Performs the Lambda consistency check.
        """
        # 1. GET INPUTS
        # MAF is in g/s
        maf_g_s = frame.get('maf', 0.0)
        
        # Fuel Rate from Ford PID 0x5E is in L/h
        fuel_lph = frame.get('fuel_rate', 0.0)
        
        # 2. CALCULATE PHYSICAL LAMBDA (The "Truth")
        # Step A: Convert Fuel to Mass Flow (g/s)
        # Diesel Density approx 835 g/L
        fuel_g_h = fuel_lph * 835.0
        fuel_g_s = fuel_g_h / 3600.0
        
        calc_lambda = 99.0 # Default to "Infinite Air" (Fuel Cut)
        
        if fuel_g_s > 0.01:
            # AFR = Mass_Air / Mass_Fuel
            current_afr = maf_g_s / fuel_g_s
            # Lambda = Current_AFR / Stoich_AFR
            calc_lambda = current_afr / self.AFR_STOICH_DIESEL
            
        # 3. GET REPORTED LAMBDA (The "Story")
        # PID 0x24 (SAE) or 0x34 (Bank 1 Sensor 1)
        # EQ_RATIO (Equivalence Ratio) is often 1/Lambda or Lambda depending on PID
        # Standard SAE J1979 PID 0x24 returns Equivalence Ratio (lambda)
        reported_lambda = frame.get('o2_lambda', calc_lambda) # Trust physics if sensor missing

        # 4. AUDIT LOGIC
        status = "COMPLIANT"
        deviation = 0.0
        
        # Check A: The "Rolling Coal" Check
        if calc_lambda < self.SMOKE_LIMIT_LAMBDA and fuel_lph > 2.0:
            status = "NON_COMPLIANT (EXCESS_SMOKE/RICH)"
            
        # Check B: The "Sensor Lying" Check
        # If we calculate Lambda 1.2 (Rich for Diesel) but Sensor says 4.0 (Very Clean)
        if fuel_lph > 2.0:
            deviation = abs(calc_lambda - reported_lambda)
            
            # If deviation is huge AND we are not in a transient (rapid throttle change)
            # (We use a simple load check as a proxy for transient stability)
            if deviation > self.MAX_SENSOR_DEVIATION:
                if calc_lambda < reported_lambda:
                    status = "SENSOR_MISMATCH (MASKING_RICH_CONDITION)"
                else:
                    status = "SENSOR_MISMATCH (DRIFT)"

        return {
            "lambda_calc": calc_lambda,
            "lambda_sensor": reported_lambda,
            "maf_delta_pct": deviation * 100.0,
            "stoich_status": status
        }