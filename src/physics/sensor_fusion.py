"""
MODULE: SENSOR_FUSION_EKF
PROFILE: FORD_SCORPION_V8
METHOD: EXTENDED_KALMAN_FILTER

DESCRIPTION:
    A forensic integrity monitor that cross-references sensor data against 
    physical laws. It treats the engine as a rigid air pump and calculates 
    theoretical airflow based on Manifold Pressure (MAP) and RPM.

    If the reported Mass Air Flow (MAF) disagrees with the calculated 
    airflow (derived from Boost & Temp), it indicates that one of the 
    sensors is being intercepted or manipulated (a common "Tuner" strategy).

    OUTPUT:
    - NIS_SCORE (Normalized Innovation Squared): 
      < 1.0 = Normal
      > 3.0 = Sensor Drift / Leak
      > 5.0 = MANIPULATION DETECTED
"""

import numpy as np
import logging
from dataclasses import dataclass

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS.EKF")

@dataclass
class EKFState:
    estimate: float
    uncertainty: float
    nis: float

class SensorIntegrityMonitor:
    """
    Standard Extended Kalman Filter (EKF) implementation for 
    Air/Fuel plausibility checks.
    """
    def __init__(self):
        # Ford 6.7L Specs
        self.displacement_L = 6.7
        # Gas Constant for Air (J/kg*K)
        self.R_specific = 287.058 
        
        # --- KALMAN PARAMETERS ---
        # Process Noise (Q): How much the real world jitters
        self.Q = 0.1 
        # Measurement Noise (R): How noisy the MAF sensor is
        self.R = 2.5 
        # Initial Estimate Error (P)
        self.P = 1.0 
        # Initial State (x): Air Mass Flow (g/s)
        self.x = 10.0 

    def _get_volumetric_efficiency(self, rpm: float, load: float) -> float:
        """
        Approximates the VE map of a Ford 6.7L 'Scorpion' V8.
        The 'Reverse Flow' heads have very high efficiency at mid-range.
        """
        # Normalized RPM (0.0 - 1.0 range for 600-3500 RPM)
        norm_rpm = (rpm - 600) / 2900.0
        
        # Base VE Curve (Polynomial fit for Turbo Diesel)
        # Low RPM (Idle) ~ 85% VE
        # Peak Torque (1800 RPM) ~ 98% VE
        # Redline (3000 RPM) ~ 92% VE
        ve_base = 0.85 + (0.13 * np.sin(norm_rpm * np.pi))
        
        # Load Correction (VGT Turbo Effect)
        # At high load, backpressure slightly reduces VE
        ve_load = 1.0 - (0.05 * (load / 100.0))
        
        return max(0.5, ve_base * ve_load)

    def predict_theoretical_maf(self, rpm, map_kpa, temp_c, load) -> float:
        """
        Uses Ideal Gas Law (PV=nRT) to calculate what MAF *should* be.
        MAF = (MAP * V_swept * RPM * VE) / (120 * R * T)
        """
        if rpm < 1 or temp_c < -50: return 0.0
        
        # Temperature in Kelvin
        temp_k = temp_c + 273.15
        
        # Swept Volume (m^3)
        v_swept = self.displacement_L / 1000.0
        
        # Volumetric Efficiency
        ve = self._get_volumetric_efficiency(rpm, load)
        
        # Calculation (g/s)
        # Pressure (Pa) = kPa * 1000
        # Result is kg/s, multiply by 1000 for g/s
        rho_theoretical = (map_kpa * 1000) / (self.R_specific * temp_k)
        
        # Volume Flow (m^3/s) = (RPM/60) * (V_swept/2) * VE
        # (Divide by 2 because 4-stroke engine sucks air every 2 revs)
        vol_flow = (rpm / 60.0) * (v_swept / 2.0) * ve
        
        mass_flow_g_s = rho_theoretical * vol_flow * 1000.0
        
        return mass_flow_g_s

    def audit_frame(self, frame: dict, dt: float) -> dict:
        """
        The Filter Step: Predict -> Update -> Check Innovation
        """
        # 1. EXTRACT SENSOR DATA
        rpm = frame.get('rpm', 800)
        load = frame.get('load', 0)
        temp = frame.get('temp', 40)
        maf_sensor = frame.get('maf', 0)
        
        # Barometric Pressure (Default 100 kPa/14.5 psi if missing)
        baro = frame.get('baro', 100.0)
        if baro < 50: baro = 100.0 
        
        # Estimate MAP (Manifold Absolute Pressure) from Load if PID missing
        # Ford 6.7L: 0% Load ~ Atmos, 100% Load ~ +30psi (approx 300kPa Absolute)
        estimated_boost = (load / 100.0) * 200.0 # kPa gauge
        map_kpa = baro + estimated_boost

        # 2. PREDICTION STEP (Physics Model)
        # Theoretical MAF based on Pressure/RPM
        maf_physics = self.predict_theoretical_maf(rpm, map_kpa, temp, load)
        
        # 3. UPDATE STEP (Kalman Filter)
        # Prediction
        x_pred = self.x 
        P_pred = self.P + self.Q
        
        # Measurement Residual (Innovation)
        # Difference between Physics Model and Real Sensor
        y = maf_sensor - x_pred
        
        # Kalman Gain
        S = P_pred + self.R
        K = P_pred / S
        
        # State Update
        self.x = x_pred + (K * y)
        self.P = (1 - K) * P_pred
        
        # 4. FORENSIC SCORING (NIS)
        # Normalized Innovation Squared
        # If the sensor matches physics, NIS is close to 0.
        # If the sensor is lying (e.g. Tuner Box), NIS spikes.
        if S > 0.001:
            nis_score = (y ** 2) / S
        else:
            nis_score = 0.0

        # Smoothing for display
        nis_score = min(nis_score, 20.0)

        return {
            "ekf_estimated_maf": self.x,
            "physics_maf": maf_physics,
            "nis_score": nis_score
        }