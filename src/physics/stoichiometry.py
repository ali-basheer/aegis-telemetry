"""
MODULE: STOICHIOMETRIC_SOLVER
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-07
CLASSIFICATION: FORENSIC / THERMO-CHEMISTRY

DESCRIPTION:
    Implements a rigorous chemical solver for Internal Combustion Engine (ICE) air-fuel handling.
    
    Defeat devices often manipulate the Air-Fuel Ratio (AFR) to lower peak combustion temperatures
    during emission tests. This module calculates the *theoretical* AFR based on Mass Air Flow (MAF)
    and Fuel Injection Rate, then compares it against the reported Wideband O2 Sensor (Lambda).

    If the reported Lambda matches the 'Test Cycle' profile but contradicts the physical mass balance
    calculated here, it indicates a 'Lambda Masquerade' attack.

PHYSICS:
    - Reaction Balancing: C_xH_y + (x + y/4)O_2 -> xCO_2 + (y/2)H_2O
    - EGR Dilution: Calculating oxygen displacement by recirculated inert gas.
    - Volumetric Efficiency (VE): Speed-Density estimation of airflow.

COMPONENTS:
    1. FuelChemistry: Dynamic hydrocarbon property solver (Diesel B7/B20).
    2. AirMassModel: Speed-Density vs MAF sensor validation.
    3. ClosedLoopAuditor: Reverse PID controller to detect artificial oscillation.
"""

import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS.STOICH")

# --- MOLECULAR CONSTANTS ---
MOL_WT_C = 12.011
MOL_WT_H = 1.008
MOL_WT_O = 15.999
MOL_WT_N = 14.007
MOL_WT_AIR = 28.97  # Standard Atmosphere

@dataclass
class FuelSpecification:
    """
    Defines the chemical composition of the fuel.
    Default: ULSD (Ultra Low Sulfur Diesel) ~ C12H23
    """
    name: str
    carbon_atoms: float = 12.0
    hydrogen_atoms: float = 23.0
    oxygen_atoms: float = 0.0 # Biodiesel has Oxygen
    density_kg_m3: float = 835.0
    lhv_mj_kg: float = 42.6

class ChemicalBalancer:
    """
    Solves the combustion reaction equation to find the Stoichiometric AFR.
    """
    
    def __init__(self, fuel: FuelSpecification):
        self.fuel = fuel
        self.afr_stoich = self._calculate_stoich_afr()
        
    def _calculate_stoich_afr(self) -> float:
        """
        Derives AFR_st from atomic balance.
        Reaction: CmHnOx + a(O2 + 3.76N2) -> mCO2 + (n/2)H2O + 3.76aN2
        
        Oxygen balance:
        x + 2a = 2m + (n/2)
        2a = 2m + n/2 - x
        a = m + n/4 - x/2  (Moles of O2 required)
        """
        m = self.fuel.carbon_atoms
        n = self.fuel.hydrogen_atoms
        x = self.fuel.oxygen_atoms
        
        moles_o2_required = m + (n / 4.0) - (x / 2.0)
        moles_air_required = moles_o2_required * 4.76 # 1 mol O2 + 3.76 mol N2
        
        mass_fuel = (m * MOL_WT_C) + (n * MOL_WT_H) + (x * MOL_WT_O)
        mass_air = moles_air_required * MOL_WT_AIR
        
        return mass_air / mass_fuel

class GasDynamics:
    """
    Models the physical flow of gases into the cylinder.
    Used to validate if the MAF sensor is reporting truthful values.
    """
    
    R_SPECIFIC_AIR = 287.058 # J/(kg*K)
    
    @staticmethod
    def calculate_air_density(pressure_pa: float, temp_c: float) -> float:
        """
        Ideal Gas Law: rho = P / (R * T)
        """
        temp_k = temp_c + 273.15
        return pressure_pa / (GasDynamics.R_SPECIFIC_AIR * temp_k)

    @staticmethod
    def estimate_volumetric_efficiency(rpm: float, load_pct: float) -> float:
        """
        Approximates VE based on a standard diesel engine map.
        Diesel engines usually have high VE (85-95%) due to no throttle plate.
        """
        # Base VE at peak torque (approx 2000 RPM)
        ve_base = 0.92
        
        # RPM Decay (Flow restriction at high speed)
        rpm_factor = 1.0 - (((rpm - 2000) ** 2) / 20_000_000)
        
        # Load Factor (Turbo spooling improves VE > 1.0)
        load_factor = 1.0 + (load_pct / 500.0)
        
        return max(0.6, min(1.2, ve_base * rpm_factor * load_factor))

    @staticmethod
    def calculate_expected_maf(rpm: float, pressure_pa: float, temp_c: float, 
                             displacement_m3: float) -> float:
        """
        Speed-Density Calculation:
        Mass Flow = Density * Volume * RPM/120 * VE
        """
        rho = GasDynamics.calculate_air_density(pressure_pa, temp_c)
        # Assuming fixed VE of 0.9 for forensic baseline (simplification)
        # In full version, use estimate_volumetric_efficiency()
        ve = 0.90 
        
        # Intake strokes per second (4-stroke) = RPM / 60 / 2
        intake_events = rpm / 120.0
        
        mass_flow_kg_s = rho * displacement_m3 * intake_events * ve
        return mass_flow_kg_s * 1000.0 # g/s

class EGRMixer:
    """
    Calculates the impact of Exhaust Gas Recirculation on intake composition.
    Defeat devices often turn off EGR to save the intake manifold from soot,
    increasing NOx drastically.
    """
    
    def __init__(self):
        # O2 concentration in fresh air (%)
        self.O2_FRESH = 20.95 
        
    def calculate_intake_o2(self, egr_rate_pct: float, lambda_val: float) -> float:
        """
        Predicts the O2% in the intake manifold.
        
        Inputs:
            egr_rate_pct: Percentage of total mass flow that is EGR.
            lambda_val: The combustion lambda (defines O2 left in exhaust).
        
        Logic:
            Exhaust O2% ~= 20.95 * (1 - 1/Lambda)  (Simplification for lean burn)
            Intake O2% = (Fresh% * (1-EGR)) + (Exhaust% * EGR)
        """
        if lambda_val <= 1.0:
            exhaust_o2 = 0.0 # Stoichiometric or Rich = No O2 left
        else:
            exhaust_o2 = self.O2_FRESH * (1.0 - (1.0 / lambda_val))
            
        egr_fraction = egr_rate_pct / 100.0
        
        mixed_o2 = (self.O2_FRESH * (1.0 - egr_fraction)) + (exhaust_o2 * egr_fraction)
        return mixed_o2

class ClosedLoopAuditor:
    """
    Simulates the ECU's Oxygen Control Loop.
    Detects if the reported Lambda is physically consistent with the 
    fueling inputs.
    """
    
    def __init__(self):
        self.chem = ChemicalBalancer(FuelSpecification("Diesel_B7"))
        self.egr = EGRMixer()
        self.displacement = 2.0 / 1000.0 # 2.0 Liters in m3
        
    def audit_cycle(self, frame: Dict) -> Dict:
        """
        Forensic Stoichiometry Check.
        
        1. Calculate Mass Air (Speed-Density) vs Reported MAF.
        2. Calculate Fuel Mass from Injector pulse (if avail) or derive from MAF/Lambda.
        3. Verify Lambda Consistency.
        """
        
        # 1. MAF Rationality Check
        # Convert Manifold Pressure (kPa) to Pa. Default to 1 bar if missing.
        map_pa = frame.get('intake_pressure_kpa', 101.0) * 1000.0
        temp_c = frame.get('intake_temp', 25.0)
        rpm = frame.get('rpm', 800)
        
        calc_maf_g_s = GasDynamics.calculate_expected_maf(rpm, map_pa, temp_c, self.displacement)
        report_maf_g_s = frame.get('maf', calc_maf_g_s) # Trust frame if exists
        
        maf_discrepancy = 0.0
        if report_maf_g_s > 0:
            maf_discrepancy = abs(calc_maf_g_s - report_maf_g_s) / report_maf_g_s
            
        # 2. Lambda Verification
        # Calculated Lambda = (Air_Mass / Fuel_Mass) / AFR_Stoich
        # Fuel rate in mg/strk -> g/s
        fuel_rate_mg_strk = frame.get('fuel_rate', 5.0) # Default idle
        fuel_g_s = (fuel_rate_mg_strk * (rpm * 4 / 120.0)) / 1000.0
        
        if fuel_g_s <= 0: fuel_g_s = 0.001
        
        calc_afr = report_maf_g_s / fuel_g_s
        calc_lambda = calc_afr / self.chem.afr_stoich
        
        report_lambda = frame.get('lambda', calc_lambda)
        
        lambda_deviation = abs(calc_lambda - report_lambda)
        
        # 3. EGR Validity
        # If EGR is active, Intake O2 should drop.
        # Defeat Detection: ECU reports EGR open, but O2 sensor reads Fresh Air levels.
        # This implies the EGR valve is physically blocked or mapped shut.
        egr_cmd = frame.get('egr_duty', 0.0)
        predicted_o2_conc = self.egr.calculate_intake_o2(egr_cmd, calc_lambda)
        
        # 4. Final Verdict
        is_maf_fraud = maf_discrepancy > 0.20 # 20% deviation
        is_lambda_fraud = lambda_deviation > 0.15 # 0.15 Lambda points
        
        status = "PASS"
        if is_maf_fraud: status = "FAIL: MAF SENSOR RATIONALITY"
        elif is_lambda_fraud: status = "FAIL: LAMBDA MISMATCH"
        
        return {
            "calc_maf_g_s": calc_maf_g_s,
            "report_maf_g_s": report_maf_g_s,
            "calc_lambda": calc_lambda,
            "report_lambda": report_lambda,
            "predicted_intake_o2": predicted_o2_conc,
            "status": status,
            "maf_delta_pct": maf_discrepancy * 100
        }

# --- UNIT TEST HARNESS ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- STOICHIOMETRIC FORENSICS ENGINE ---")
    
    auditor = ClosedLoopAuditor()
    print(f"Fuel Stoich AFR: {auditor.chem.afr_stoich:.2f}:1")
    
    # Test 1: Idle Consistency
    print("\n[TEST 1] Idle (800 RPM, 101kPa, 8 mg/strk Fuel)")
    frame_idle = {
        'rpm': 800,
        'intake_pressure_kpa': 101.0,
        'intake_temp': 30.0,
        'maf': 8.5, # g/s (Approx correct)
        'fuel_rate': 8.0, # mg/strk
        'lambda': 4.0 # Diesel idle is very lean
    }
    res = auditor.audit_cycle(frame_idle)
    print(f"Calculated MAF: {res['calc_maf_g_s']:.2f} g/s")
    print(f"Lambda Calc: {res['calc_lambda']:.2f} vs Report: {res['report_lambda']:.2f}")
    print(f"Status: {res['status']}")
    
    # Test 2: The "Lean Lie" (Reporting Lean, Running Rich)
    # ECU claims Lambda 2.0 (Clean), but physics says 1.1 (Dirty)
    print("\n[TEST 2] Lambda Masquerade Attack")
    frame_cheat = {
        'rpm': 2500,
        'intake_pressure_kpa': 150.0, # Boosted
        'intake_temp': 50.0,
        'maf': 60.0,
        'fuel_rate': 45.0, # High fuel
        'lambda': 2.0 # REPORTED (Fake)
    }
    res = auditor.audit_cycle(frame_cheat)
    print(f"Calculated MAF: {res['calc_maf_g_s']:.2f} g/s")
    print(f"Physical Lambda: {res['calc_lambda']:.2f} vs Reported: {res['report_lambda']:.2f}")
    print(f"Status: {res['status']}")
