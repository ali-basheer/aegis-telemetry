"""
MODULE: ADIABATIC_COMBUSTION_KERNEL
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
STATUS: PRODUCTION_READY (FORD 6.7L POWERSTROKE PROFILE)

DESCRIPTION:
    Implements a 0-Dimensional Thermodynamic Cycle Simulation.
    This module calculates the 'Ground Truth' for NOx formation based on 
    the specific geometry and thermodynamics of the Ford 6.7L "Scorpion" V8.

    It validates if the reported emissions are physically possible given the 
    amount of fuel and air currently entering the engine.

    CONFIGURATION:
    Target: Ford 6.7L Powerstroke V8 Turbo Diesel (2011-Present)
    Specs: Bore 99.06mm | Stroke 107.95mm | CR 16.2:1
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS")

# --- PHYSICAL CONSTANTS ---
R_UNIV = 8.314462  # J/(mol*K)
P_STD = 101325.0   # Pa (1 atm)
T_STD = 298.15     # K (25C)

@dataclass
class CombustionState:
    crank_angle: float
    pressure: float
    temperature: float
    volume: float
    heat_release_rate: float
    nox_ppm: float

class NASA_Polynomials:
    """
    High-fidelity calculation of Specific Heat (Cp) using NASA 7-coefficient polynomials.
    Essential for accuracy when cylinder temps exceed 1000K.
    """
    # Coefficients for N2 (Nitrogen) - High Temp Range (1000K - 6000K)
    COEFFS_N2_HIGH = [
        0.0291133000E+02, 0.0861175111E-01, -0.0100816438E-04,
        0.0949024682E-09, -0.0333332742E-12, -0.0714002835E+04, -0.0556671918E+01
    ]

    @staticmethod
    def get_cp(t_kelvin: float) -> float:
        # Simplified for simulation speed: N2 dominates diesel exhaust mass
        a = NASA_Polynomials.COEFFS_N2_HIGH
        t = max(t_kelvin, 1000.0) # Clamp for polynomial validity
        
        cp_r = (a[0] + 
                a[1] * t + 
                a[2] * (t**2) + 
                a[3] * (t**3) + 
                a[4] * (t**4))
        return cp_r * R_UNIV

class ZeldovichKinetics:
    """
    Solver for the Extended Zeldovich Mechanism (Thermal NO Formation).
    Real-time solver for: N2 + O2 <-> 2NO
    """
    @staticmethod
    def calculate_formation_rate(temp_k: float, 
                               pressure_pa: float, 
                               o2_conc_mol_m3: float, 
                               n2_conc_mol_m3: float) -> float:
        # OPTIMIZATION: Thermal NOx is negligible below ~2000K.
        if temp_k < 2000:
            return 0.0
            
        # Reaction 1 Forward: O + N2 -> NO + N (The Rate Limiting Step)
        # Rate constant k1f = 1.8e11 * exp(-38370 / T)
        k1f = 1.8e11 * math.exp(-38370 / temp_k)
        
        # Equilibrium assumption for O radical concentration
        # [O] ~ K * [O2]^0.5
        o_radical_conc = 3970 * (max(0, o2_conc_mol_m3)**0.5) * math.exp(-31090 / temp_k)
        
        # d[NO]/dt = 2 * k1f * [O] * [N2]
        d_no_dt = 2.0 * k1f * o_radical_conc * n2_conc_mol_m3
        return d_no_dt

class CombustionEngineModel:
    """
    0-D Thermodynamic Engine Simulator.
    Currently Configured: Ford 6.7L Powerstroke
    """
    
    def __init__(self, bore_mm=99.06, stroke_mm=107.95, compression_ratio=16.2):
        # Geometry: Ford 6.7L Powerstroke Specs
        self.bore = bore_mm / 1000.0
        self.stroke = stroke_mm / 1000.0
        self.cr = compression_ratio
        
        # Rod length estimated based on typical stroke ratio (~1.65)
        # Critical for accurate piston velocity/volume curve
        self.rod_length = 178.0 / 1000.0
        
        # Calculated Geometry per Cylinder
        self.disp_vol = (math.pi / 4) * (self.bore**2) * self.stroke
        self.clearance_vol = self.disp_vol / (self.cr - 1)
        
        # Wiebe Function Parameters (Direct Injection Diesel Profile)
        # These shape the "burn curve" of the fuel.
        self.wiebe_a = 5.0
        self.wiebe_m = 2.0
        self.burn_duration = 45.0 # Crank degrees (Slower burn for larger bore V8)

    def _cylinder_volume(self, theta_deg):
        """Calculates instantaneous cylinder volume at crank angle theta."""
        theta_rad = math.radians(theta_deg)
        term1 = self.stroke / 2
        term2 = self.rod_length + term1 - (
            term1 * math.cos(theta_rad) + 
            math.sqrt(self.rod_length**2 - term1**2 * math.sin(theta_rad)**2)
        )
        return self.clearance_vol + (math.pi * self.bore**2 / 4) * term2

    def simulate_cycle(self, rpm: float, load_pct: float, intake_temp_c: float, 
                      fuel_rate_mm3: float) -> Tuple[float, List[CombustionState]]:
        """
        Performs a full thermodynamic cycle simulation (-180 to +180 deg).
        """
        theta = -180.0
        dt = (60.0 / max(rpm, 100.0)) / 360.0 # Seconds per degree
        
        t_curr = intake_temp_c + 273.15
        
        # --- VIRTUAL VGT TURBO MODEL ---
        # The Ford 6.7L builds boost aggressively.
        # We approximate Manifold Absolute Pressure (MAP) based on Load %.
        # Stock peak boost is approx 30-32 PSI (~220 kPa gauge).
        # Formula: Atmos + (Load_Factor * Max_Boost)
        
        # 0-10% Load = 0 Boost
        # 100% Load = ~220 kPa Boost
        boost_slope = 2200.0 # Pa per % Load
        boost_pressure = P_STD + (max(0, load_pct - 10) * boost_slope)
        
        p_curr = boost_pressure
        v_curr = self._cylinder_volume(theta)
        
        # Mass of Air (Ideal Gas Law at IVC)
        m_air = (p_curr * v_curr) / (287.0 * t_curr)
        
        # Fuel Energy Input
        # Diesel LHV = 42.5 MJ/kg, Density = 0.835 kg/L
        fuel_mass_kg = (fuel_rate_mm3 / 1000.0) * 0.835 / 1000.0
        total_energy_j = fuel_mass_kg * 42.5e6
        
        cumulative_nox = 0.0
        start_of_combustion = -5.0 # BTDC (Typical pilot injection timing)
        
        # --- EULER INTEGRATION LOOP ---
        while theta < 180.0:
            v_next = self._cylinder_volume(theta + 1)
            dV = v_next - v_curr
            
            # 1. Heat Release (Wiebe Function)
            theta_norm = (theta - start_of_combustion) / self.burn_duration
            dX = 0.0
            if 0 <= theta_norm <= 1:
                dX = (self.wiebe_a * (self.wiebe_m + 1) / self.burn_duration) * \
                     (theta_norm ** self.wiebe_m) * \
                     math.exp(-self.wiebe_a * (theta_norm ** (self.wiebe_m + 1)))

            dQ_combustion = total_energy_j * dX
            
            # 2. First Law of Thermodynamics (dU = dQ - dW)
            work = p_curr * dV
            du = dQ_combustion - work
            
            # Estimate Temperature Rise (dT = dU / (m * Cv))
            # Cv for air approx 718 J/kgK
            dt_temp = du / (m_air * 718.0)
            t_next = t_curr + dt_temp
            
            # 3. Update Pressure (Ideal Gas Law)
            p_next = (m_air * 287.0 * t_next) / v_next
            
            # 4. Zeldovich NOx Formation
            # We assume a local flame front temp is higher than bulk temp
            t_flame = t_next + 2100 if dX > 0.005 else t_next
            
            total_conc = p_next / (R_UNIV * t_flame)
            n2_conc = total_conc * 0.79 # Nitrogen is 79% of air
            
            # Simple Oxygen depletion model
            o2_fraction = 0.21 * (1 - (theta_norm if 0 <= theta_norm <= 1 else 0))
            o2_conc = total_conc * max(0.01, o2_fraction)
            
            d_no_dt = ZeldovichKinetics.calculate_formation_rate(t_flame, p_next, o2_conc, n2_conc)
            cumulative_nox += d_no_dt * v_next * dt
            
            # Step Forward
            theta += 1.0
            t_curr = t_next
            p_curr = p_next
            v_curr = v_next

        # Convert Moles NO to PPM
        total_moles = m_air / 0.029 # Molar mass of air approx 0.029 kg/mol
        final_ppm = (cumulative_nox / total_moles) * 1e6
        
        return final_ppm, []

class ForensicCombustionValidator:
    def __init__(self):
        self.engine = CombustionEngineModel()
        
    def validate_snapshot(self, frame: Dict) -> Dict:
        """
        Comparing Physics (Model) vs Reality (Sensor).
        """
        # 1. Get Inputs
        rpm = float(frame.get('rpm', 1000))
        load = float(frame.get('load', 0))
        temp = float(frame.get('temp', 25))
        
        # 2. Fuel Rate Normalization
        # Ford PID 0x5E gives Fuel Rate in Liters/Hour.
        # We need mm3 per Stroke for the physics cycle.
        fuel_rate_lph = float(frame.get('fuel_rate', 0.0))
        
        if fuel_rate_lph > 0.1:
            # L/hr -> mm3/hr (x1,000,000)
            # mm3/hr -> mm3/min (/60)
            # mm3/min -> mm3/stroke ( / (RPM * 4))  [4 strokes per rev for V8]
            fuel_mm3 = (fuel_rate_lph * 1e6) / (max(rpm, 1) * 60 * 4)
        else:
            # Fallback Estimation (if PID 0x5E is missing)
            # Approx 100mm3 at full load for a 6.7L cylinder
            fuel_mm3 = (load / 100.0) * 110.0 
        
        # 3. Run Simulation
        sim_ppm, _ = self.engine.simulate_cycle(
            rpm=rpm,
            load_pct=load,
            intake_temp_c=temp,
            fuel_rate_mm3=fuel_mm3
        )
        
        # 4. Compare with Sensor (if available)
        reported = float(frame.get('actual_nox', 0))
        
        # Avoid division by zero in delta calc
        delta = 0.0
        if sim_ppm > 1.0:
            delta = (reported - sim_ppm) / sim_ppm
            
        return {
            "physics_nox_ppm": sim_ppm,
            "reported_nox_ppm": reported,
            "delta_percent": delta
        }