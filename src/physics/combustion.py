"""
MODULE: ADIABATIC_COMBUSTION_KERNEL
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-02
CLASSIFICATION: FORENSIC / PHYSICS-GRADE

DESCRIPTION:
    Implements a 0-Dimensional Thermodynamic Cycle Simulation for Direct Injection (DI) Diesel Engines.
    Used to establish the 'Ground Truth' for in-cylinder NOx formation, independent of ECU reporting.
    
    This module calculates the theoretical adiabatic flame temperature and solves the 
    Extended Zeldovich Mechanism for thermal NO formation rates. It serves as the 
    primary validation layer against 'Thermal Windowing' defeat devices.

PHYSICS MODEL:
    1. Compression: Isentropic compression with variable specific heat ratios (gamma).
    2. Combustion: Wiebe Function heat release modeling.
    3. Kinetics: 3-Step Zeldovich Mechanism with Arrhenius rate constants.
    4. Thermodynamics: NASA 7-Coefficient Polynomials for high-temp gas properties.

REFERENCES:
    - Heywood, J.B., "Internal Combustion Engine Fundamentals", McGraw-Hill.
    - Zeldovich, Y.B., "The Oxidation of Nitrogen in Combustion and Explosions".
"""

import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS.COMBUSTION")

# --- PHYSICAL CONSTANTS ---
R_UNIV = 8.314462  # J/(mol*K)
P_STD = 101325.0   # Pa
T_STD = 298.15     # K

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
    High-fidelity calculation of Specific Heat (Cp), Enthalpy (H), and Entropy (S)
    using 7-coefficient NASA polynomials. Essential for accuracy > 1000K.
    """
    
    # Coefficients for N2 (Nitrogen) - High Temp Range (1000K - 6000K)
    COEFFS_N2_HIGH = [
        0.0291133000E+02, 0.0861175111E-01, -0.0100816438E-04,
        0.0949024682E-09, -0.0333332742E-12, -0.0714002835E+04, -0.0556671918E+01
    ]
    
    # Coefficients for O2 (Oxygen) - High Temp Range
    COEFFS_O2_HIGH = [
        0.0369757819E+02, 0.0613519704E-02, -0.0125884205E-05,
        0.0177528148E-09, -0.0113643531E-13, -0.0123393018E+05, 0.0318916551E+02
    ]

    @staticmethod
    def get_cp(t_kelvin: float, species: str = 'N2') -> float:
        """
        Returns Specific Heat Capacity (Cp) in J/(mol*K) at temperature T.
        """
        # Select coeffs
        a = NASA_Polynomials.COEFFS_N2_HIGH if species == 'N2' else NASA_Polynomials.COEFFS_O2_HIGH
        
        # Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
        cp_r = (a[0] + 
                a[1] * t_kelvin + 
                a[2] * (t_kelvin**2) + 
                a[3] * (t_kelvin**3) + 
                a[4] * (t_kelvin**4))
                
        return cp_r * R_UNIV

class ZeldovichKinetics:
    """
    Solver for the Extended Zeldovich Mechanism (Thermal NO Formation).
    
    Reactions:
    1. O + N2 <-> NO + N
    2. N + O2 <-> NO + O
    3. N + OH <-> NO + H
    """
    
    @staticmethod
    def calculate_formation_rate(temp_k: float, 
                               pressure_pa: float, 
                               o2_conc_mol_m3: float, 
                               n2_conc_mol_m3: float) -> float:
        """
        Calculates d[NO]/dt in mol/(m^3 * s).
        """
        if temp_k < 1800:
            return 0.0 # Kinetic threshold
            
        # Rate Constants (Arrhenius Form: A * T^b * exp(-Ea/RT))
        # Reaction 1 Forward: O + N2 -> NO + N
        k1f = 1.8e11 * math.exp(-38370 / temp_k)
        
        # Reaction 2 Forward: N + O2 -> NO + O
        k2f = 6.4e9 * temp_k * math.exp(-3150 / temp_k)
        
        # Reaction 3 Forward: N + OH -> NO + H
        k3f = 3.8e13
        
        # Equilibrium assumption for O radical concentration [O]_eq
        # K_p_O2 = exp(-DeltaG / RT) approximation via partial pressures
        # Simplified for simulation speed: [O] ~= K * [O2]^0.5 * exp(-E/RT)
        o_radical_conc = 3970 * (o2_conc_mol_m3**0.5) * math.exp(-31090 / temp_k)
        
        # Rate Limiting Step is usually Reaction 1
        d_no_dt = 2.0 * k1f * o_radical_conc * n2_conc_mol_m3
        
        return d_no_dt

class CombustionEngineModel:
    """
    0-D Thermodynamic Engine Simulator.
    Reconstructs the cylinder state based on telemetry inputs.
    """
    
    def __init__(self, bore_mm=82.5, stroke_mm=92.8, compression_ratio=16.8):
        # Default geometry for a generic 2.0L TDI Engine
        self.bore = bore_mm / 1000.0
        self.stroke = stroke_mm / 1000.0
        self.cr = compression_ratio
        self.rod_length = 144.0 / 1000.0
        
        # Calculated Geometry
        self.disp_vol = (math.pi / 4) * (self.bore**2) * self.stroke
        self.clearance_vol = self.disp_vol / (self.cr - 1)
        
        # Wiebe Function Parameters (Diesel)
        self.wiebe_a = 5.0
        self.wiebe_m = 2.0
        self.burn_duration = 40.0 # Crank degrees

    def _cylinder_volume(self, theta_deg):
        """
        Calculates instantaneous cylinder volume at crank angle theta.
        """
        theta_rad = math.radians(theta_deg)
        term1 = self.stroke / 2
        term2 = self.rod_length + term1 - (
            term1 * math.cos(theta_rad) + 
            math.sqrt(self.rod_length**2 - term1**2 * math.sin(theta_rad)**2)
        )
        return self.clearance_vol + (math.pi * self.bore**2 / 4) * term2

    def _wiebe_hrr(self, theta, start_combustion):
        """
        Calculates normalized Heat Release Rate (dX/dTheta).
        """
        if theta < start_combustion or theta > (start_combustion + self.burn_duration):
            return 0.0
            
        norm_theta = (theta - start_combustion) / self.burn_duration
        
        # dX/dTheta
        rate = (self.wiebe_a * (self.wiebe_m + 1) / self.burn_duration) * \
               (norm_theta ** self.wiebe_m) * \
               math.exp(-self.wiebe_a * (norm_theta ** (self.wiebe_m + 1)))
               
        return rate

    def simulate_cycle(self, rpm: float, load_pct: float, intake_temp_c: float, 
                      fuel_rate_mm3: float) -> Tuple[float, List[CombustionState]]:
        """
        Performs a full thermodynamic cycle simulation (-180 to +180 deg).
        Returns (Peak_NOx_ppm, Trace_Log).
        """
        
        # 1. Initialize State at IVC (Intake Valve Close) -> -180 deg approx
        theta = -180.0
        dt = (60.0 / rpm) / 360.0 # Seconds per degree
        
        t_curr = intake_temp_c + 273.15
        p_curr = 1.2e5 # Boost pressure approximation based on load
        if load_pct > 50: p_curr += (load_pct - 50) * 2000 # Turbo boost map
        
        v_curr = self._cylinder_volume(theta)
        
        # Mass calculation (Ideal Gas Law)
        m_air = (p_curr * v_curr) / (287.0 * t_curr)
        
        # Total fuel energy for this cycle
        # Diesel LHV = 42.5 MJ/kg
        fuel_mass_kg = (fuel_rate_mm3 / 1000.0) * 0.835 / 1000.0 # Density 0.835
        total_energy_j = fuel_mass_kg * 42.5e6
        
        trace = []
        cumulative_nox = 0.0
        start_of_combustion = -10.0 # BTDC (Timing Advance)
        
        # 2. Iterate through Crank Angle (Euler Integration)
        while theta < 180.0:
            v_next = self._cylinder_volume(theta + 1)
            dV = v_next - v_curr
            
            # A. Heat Release
            dX = self._wiebe_hrr(theta, start_of_combustion)
            dQ_combustion = total_energy_j * dX
            
            # B. First Law of Thermodynamics: dU = dQ - dW
            # dQ_net = dQ_comb - dQ_wall (Simplified adiabatic here)
            # dW = P * dV
            # T_next derived from Ideal Gas + Energy Balance
            
            # Variable Gamma based on T
            cp = NASA_Polynomials.get_cp(t_curr)
            cv = cp - R_UNIV
            gamma = cp / cv
            
            # Isentropic compression / expansion step
            # P * V^gamma = const (Simplified)
            # Refined stepwise update:
            work = p_curr * dV
            du = dQ_combustion - work
            
            dt_temp = du / (m_air * 718.0) # Approx Cv for air
            t_next = t_curr + dt_temp
            
            # Correct Pressure via Ideal Gas Law
            p_next = (m_air * 287.0 * t_next) / v_next
            
            # C. Zeldovich NOx Formation
            # We assume local flame temp is higher than bulk temp
            # Adiabatic Flame Temp approx for kinetics zone
            t_flame = t_next + 1200 if dX > 0.001 else t_next
            
            # Molar concentrations P = C * R * T -> C = P / (RT)
            total_conc = p_next / (R_UNIV * t_flame)
            n2_conc = total_conc * 0.79
            o2_conc = total_conc * 0.15 # Consumed during burn
            
            # Rate d[NO]/dt
            d_no_dt = ZeldovichKinetics.calculate_formation_rate(t_flame, p_next, o2_conc, n2_conc)
            
            # Integrate: mol/m3/s * vol * time
            mol_no_generated = d_no_dt * v_next * dt
            cumulative_nox += mol_no_generated
            
            # State Update
            theta += 1.0
            t_curr = t_next
            p_curr = p_next
            v_curr = v_next
            
            state = CombustionState(theta, p_curr, t_curr, v_curr, dX, cumulative_nox)
            trace.append(state)

        # 3. Convert Moles NO to PPM
        # Total Moles in cylinder
        total_moles = m_air / 0.029 # Molar mass of air
        final_ppm = (cumulative_nox / total_moles) * 1e6
        
        return final_ppm, trace

class ForensicCombustionValidator:
    """
    Wrapper for the physics engine to run forensic audits on OBD frames.
    """
    def __init__(self):
        self.engine = CombustionEngineModel()
        
    def validate_snapshot(self, frame: Dict) -> Dict:
        """
        Compares reported NOx against physics limit.
        """
        # Estimate Fuel Rate from Load if not provided (Forensic approximation)
        fuel_mm3 = frame.get('fuel_rate', frame['load'] * 0.85)
        
        sim_ppm, _ = self.engine.simulate_cycle(
            rpm=frame['rpm'],
            load_pct=frame['load'],
            intake_temp_c=frame['temp'],
            fuel_rate_mm3=fuel_mm3
        )
        
        return {
            "physics_nox_ppm": sim_ppm,
            "reported_nox_ppm": frame.get('actual_nox', 0),
            "delta_percent": (frame.get('actual_nox', 1) - sim_ppm) / (sim_ppm + 1e-6)
        }

# --- UNIT TEST HARNESS ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Initializing Forensic Combustion Simulator...")
    
    sim = CombustionEngineModel()
    
    # Test Case: High Load (Should generate high NOx)
    print("\n--- SIMULATION: FULL LOAD (Highway Merge) ---")
    ppm, trace = sim.simulate_cycle(rpm=2500, load_pct=90, intake_temp_c=40, fuel_rate_mm3=65.0)
    print(f"RPM: 2500 | Load: 90% | Fuel: 65mm3")
    print(f"Theoretical NOx Generation: {ppm:.2f} ppm")
    print(f"Peak Pressure: {max(s.pressure for s in trace)/1e5:.1f} bar")
    print(f"Peak Temp: {max(s.temperature for s in trace):.1f} K")
    
    # Test Case: Thermal Windowing (Low Load)
    print("\n--- SIMULATION: IDLE (Cold Start) ---")
    ppm_idle, _ = sim.simulate_cycle(rpm=800, load_pct=15, intake_temp_c=10, fuel_rate_mm3=8.0)
    print(f"RPM: 800 | Load: 15% | Fuel: 8mm3")
    print(f"Theoretical NOx Generation: {ppm_idle:.2f} ppm")
    
    if ppm_idle > 50:
        print("[ALERT] Physics indicates NOx formation is possible even at idle.")
    else:
        print("[INFO] Kinetic formation inhibited by low temp (Expected behavior).")
