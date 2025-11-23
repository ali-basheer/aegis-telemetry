"""
MODULE: EXTENDED_KALMAN_FILTER (SENSOR FUSION INTEGRITY)
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-09
CLASSIFICATION: FORENSIC / ESTIMATION THEORY

DESCRIPTION:
    Implements a discrete-time Extended Kalman Filter (EKF) to estimate the true state
    of the Engine and Aftertreatment system in the presence of sensor noise.

    This module serves two forensic purposes:
    1. "Sensor Masquerade" Detection: If a defeat device freezes a sensor value 
       (e.g., holding Intake Temp constant to prevent map switching), the EKF 
       diverges from the physics model, generating high 'Innovation' scores.
    2. "Replay Attack" Detection: If sensor noise is artificially removed (Variance -> 0),
       the NIS (Normalized Innovation Squared) statistic drops below the Chi-Squared 
       lower bound, flagging the data as 'Synthetic'.

STATE VECTOR [x]:
    x[0]: Engine Block Temperature (C)
    x[1]: Exhaust Gas Temperature (C)
    x[2]: NOx Concentration (ppm)

CONTROL VECTOR [u]:
    u[0]: Fuel Injection Rate (mm3/str)
    u[1]: RPM
    u[2]: Vehicle Speed (km/h)

MEASUREMENT VECTOR [z]:
    z[0]: ECT Sensor (Engine Coolant Temp)
    z[1]: EGT Sensor (Exhaust Gas Temp)
    z[2]: NOx Sensor 1 (Upstream)
"""

import math
import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS.EKF")

class LinearAlgebra:
    """
    Helper class for matrix operations specific to Kalman Filtering.
    Wrapped to allow compilation on embedded targets without full scipy stack.
    """
    @staticmethod
    def invert_3x3(m: np.ndarray) -> np.ndarray:
        """
        Manual inversion of 3x3 matrix to ensure deterministic behavior.
        """
        det = np.linalg.det(m)
        if abs(det) < 1e-9:
            raise ValueError("Matrix is singular (non-invertible). Sensor configuration invalid.")
        return np.linalg.inv(m)

class ThermalStateModel:
    """
    Physics equations governing the state transitions F(x, u).
    """
    # Heat Transfer Coefficients
    K_BLOCK_COOLING = 0.05  # Radiative/Convective cooling
    K_COMB_HEATING = 0.15   # Heat from combustion
    K_EXH_FLOW = 0.20       # Heat transport via mass flow
    
    @staticmethod
    def predict_next_state(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Non-linear state transition function x_k = f(x_{k-1}, u_{k-1})
        """
        # Unpack State
        t_eng = x[0]
        t_exh = x[1]
        nox = x[2]
        
        # Unpack Control
        fuel = u[0]
        rpm = u[1]
        speed = u[2]
        
        # 1. Engine Temp Dynamics (First Order Thermal Model)
        # d(T_eng)/dt = (Heat_In - Heat_Out)
        # Heat_In is proportional to Fuel * RPM
        heat_gen = fuel * (rpm / 1000.0) * ThermalStateModel.K_COMB_HEATING
        # Heat_Out is proportional to T_eng - Ambient (assumed 20C) + Airflow (Speed)
        heat_loss = (t_eng - 20.0) * ThermalStateModel.K_BLOCK_COOLING * (1.0 + speed/100.0)
        
        t_eng_next = t_eng + (heat_gen - heat_loss) * dt
        
        # 2. Exhaust Temp Dynamics
        # T_exh lags T_eng but gets immediate spikes from combustion
        # d(T_exh)/dt = k * (T_comb - T_exh)
        t_comb_proxy = t_eng + (fuel * 5.0) # Combustion is hotter than block
        t_exh_next = t_exh + ThermalStateModel.K_EXH_FLOW * (t_comb_proxy - t_exh) * dt
        
        # 3. NOx Dynamics (Simplified Zeldovich Proxy)
        # NOx is instantaneous based on Load/Temp, but modeled as a state with decay
        # to account for sensor transport delay.
        nox_target = (fuel * 10.0) * (1.0 + (t_exh / 500.0)) # Crude generation model
        # First order lag (Sensor response time)
        nox_next = nox + 0.8 * (nox_target - nox) * dt
        
        return np.array([t_eng_next, t_exh_next, nox_next])

    @staticmethod
    def get_jacobian_F(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Computes the Jacobian Matrix df/dx linearized at current state.
        F = [ d(t_eng)/d(t_eng)  d(t_eng)/d(t_exh)  d(t_eng)/d(nox) ]
            [ ...                                               ]
        """
        # Simplified Jacobian for the thermal model above
        # d(T_eng_next) / d(T_eng) = 1 - K_cooling * dt
        f00 = 1.0 - (ThermalStateModel.K_BLOCK_COOLING * dt)
        
        # d(T_exh_next) / d(T_exh) = 1 - K_exh * dt
        f11 = 1.0 - (ThermalStateModel.K_EXH_FLOW * dt)
        
        # d(NOx_next) / d(NOx) = 1 - 0.8 * dt (Decay term)
        f22 = 1.0 - 0.8 * dt
        
        return np.diag([f00, f11, f22])

class ExtendedKalmanFilter:
    """
    The EKF Engine.
    """
    
    def __init__(self):
        # State Vector [T_eng, T_exh, NOx]
        self.x = np.array([20.0, 20.0, 0.0]) 
        
        # Covariance Matrix (Uncertainty)
        self.P = np.eye(3) * 10.0 
        
        # Process Noise Matrix (Q) - How much we trust the physics model
        # High Q = Physics is uncertain, trust sensors more
        # Low Q = Physics is perfect, trust sensors less
        self.Q = np.diag([0.1, 0.5, 5.0]) 
        
        # Measurement Noise Matrix (R) - How noisy we expect sensors to be
        self.R = np.diag([1.0, 2.0, 10.0]) # NOx sensor is noisy (10.0)
        
        # Measurement Matrix (H) - Maps state to sensors (Identity map here)
        self.H = np.eye(3)
        
        # Statistics
        self.nis_history = []

    def predict(self, u: np.ndarray, dt: float):
        """
        Time Update (Prediction Step).
        Project state ahead using physics.
        """
        # 1. Project State: x = f(x, u)
        self.x = ThermalStateModel.predict_next_state(self.x, u, dt)
        
        # 2. Project Covariance: P = F * P * F' + Q
        F = ThermalStateModel.get_jacobian_F(self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray) -> float:
        """
        Measurement Update (Correction Step).
        Returns NIS (Normalized Innovation Squared) for forensic auditing.
        """
        # 1. Calculate Innovation (Residual): y = z - Hx
        y = z - (self.H @ self.x)
        
        # 2. Calculate Innovation Covariance: S = H * P * H' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # 3. Calculate Kalman Gain: K = P * H' * S^-1
        try:
            K = self.P @ self.H.T @ LinearAlgebra.invert_3x3(S)
        except ValueError:
            logger.critical("Singular Matrix in EKF Update - Sensor Failure?")
            return 999.0
            
        # 4. Update State: x = x + Ky
        self.x = self.x + (K @ y)
        
        # 5. Update Covariance: P = (I - KH) * P
        I = np.eye(3)
        self.P = (I - (K @ self.H)) @ self.P
        
        # 6. Calculate NIS (Chi-Squared Statistic)
        # NIS = y' * S^-1 * y
        # This measures how "likely" this measurement is given our history.
        nis = y.T @ np.linalg.inv(S) @ y
        self.nis_history.append(nis)
        
        return nis

class SensorIntegrityMonitor:
    """
    Forensic layer sitting on top of the EKF.
    Analyzes NIS statistics to detect spoofing.
    """
    
    # Chi-Squared Thresholds (3 Degrees of Freedom, 95% Confidence)
    NIS_UPPER_BOUND = 7.81
    NIS_LOWER_BOUND = 0.35 # If NIS < 0.35 consistently, data is "Too Perfect"
    
    def __init__(self):
        self.ekf = ExtendedKalmanFilter()
        self.anomalies = 0
        self.perfect_matches = 0
        
    def audit_frame(self, frame: Dict, dt: float) -> Dict:
        """
        Ingests telemetry, runs EKF, returns Integrity Status.
        """
        # Extract Control Vector u
        u = np.array([
            frame.get('fuel_rate', 0.0),
            frame.get('rpm', 0.0),
            frame.get('vehicle_speed', 0.0)
        ])
        
        # Extract Measurement Vector z
        z = np.array([
            frame.get('coolant_temp', 20.0),
            frame.get('exhaust_temp', 20.0),
            frame.get('actual_nox', 0.0)
        ])
        
        # EKF Steps
        self.ekf.predict(u, dt)
        nis = self.ekf.update(z)
        
        # Forensic Logic
        status = "VALID"
        confidence = 1.0
        
        # CHECK 1: Divergence (Sensor Failure or Massive Cheat)
        if nis > self.NIS_UPPER_BOUND:
            self.anomalies += 1
            if self.anomalies > 5: # Persistence filter
                status = "INTEGRITY_FAIL: PHYSICS_VIOLATION"
                confidence = 0.0
                logger.warning(f"NIS Spike ({nis:.2f}). Sensors contradict Thermodynamics.")
        else:
            self.anomalies = max(0, self.anomalies - 1)
            
        # CHECK 2: Over-fitting (Replay Attack / Simulation)
        # Real sensors have noise. If NIS is essentially 0, the sensor is 
        # tracking the model *too* perfectly (or outputting constant values that happen to match).
        if nis < 0.05: 
            self.perfect_matches += 1
            if self.perfect_matches > 20:
                status = "INTEGRITY_FAIL: SYNTHETIC_DATA_DETECTED"
                confidence = 0.0
                logger.critical(f"NIS Low ({nis:.4f}). Data lacks natural entropy.")
        else:
            self.perfect_matches = max(0, self.perfect_matches - 1)
            
        return {
            "ekf_state_est": self.ekf.x.tolist(),
            "nis_score": nis,
            "status": status,
            "confidence": confidence
        }

# --- UNIT TEST HARNESS ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- SENSOR FUSION INTEGRITY TEST ---")
    
    monitor = SensorIntegrityMonitor()
    
    # Scenario 1: Normal Operation (Noisy but valid)
    print("\n[SCENARIO 1] Normal Engine Warmup")
    
    state_true = np.array([20.0, 20.0, 0.0])
    
    for t in range(20):
        # Simulate Controls (Idle)
        u = np.array([10.0, 800.0, 0.0])
        
        # Evolve "True" Physics
        state_true = ThermalStateModel.predict_next_state(state_true, u, 1.0)
        
        # Simulate "Noisy" Sensors (Gaussian Noise)
        noise = np.random.normal(0, 1.0, 3)
        z_meas = state_true + noise
        
        frame = {
            'fuel_rate': u[0], 'rpm': u[1], 'vehicle_speed': u[2],
            'coolant_temp': z_meas[0], 'exhaust_temp': z_meas[1], 'actual_nox': z_meas[2]
        }
        
        res = monitor.audit_frame(frame, 1.0)
        print(f"T={t} | NIS: {res['nis_score']:.2f} | Est T_eng: {res['ekf_state_est'][0]:.1f} | Status: {res['status']}")

    # Scenario 2: The "Frozen Sensor" Attack
    print("\n[SCENARIO 2] Sensor Freezing (Defeat Device Active)")
    # The engine keeps heating up, but the sensor reports 20C constant
    
    frozen_z = np.array([20.0, 20.0, 0.0]) # STUCK
    
    for t in range(10):
        # Controls imply heat generation
        u = np.array([50.0, 2500.0, 100.0]) 
        
        frame = {
            'fuel_rate': u[0], 'rpm': u[1], 'vehicle_speed': u[2],
            'coolant_temp': frozen_z[0], 'exhaust_temp': frozen_z[1], 'actual_nox': frozen_z[2]
        }
        
        res = monitor.audit_frame(frame, 1.0)
        print(f"T={t+20} | NIS: {res['nis_score']:.2f} | Est T_eng: {res['ekf_state_est'][0]:.1f} | Status: {res['status']}")
        
    print("\nAnalysis:")
    print("Notice how NIS explodes in Scenario 2.")
    print("The EKF 'knows' the engine is getting hot based on Fuel/RPM.")
    print("When the sensor stays at 20C, the divergence flags a 'PHYSICS_VIOLATION'.")
