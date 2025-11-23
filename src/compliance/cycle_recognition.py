"""
MODULE: DRIVING_CYCLE_RECOGNITION (DCR)
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-08
CLASSIFICATION: FORENSIC / ALGORITHMIC
"""

import math
import logging
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure module-level logger
logger = logging.getLogger("AEGIS.COMPLIANCE.DCR")

@dataclass
class TestCycleRef:
    name: str
    duration_sec: int
    velocity_profile: np.ndarray # 1Hz array of target speeds (km/h)

class RegulatoryProfiles:
    @staticmethod
    def get_ftp75() -> TestCycleRef:
        t = np.linspace(0, 500, 500)
        v = 5.0 * np.sin(t/10) + 10.0 * np.sin(t/30) + (t/20)
        v = np.clip(v, 0, 56)
        v[20:40] = 0
        v[150:180] = 0
        return TestCycleRef("EPA_FTP_75_COLD", 500, v)

    @staticmethod
    def get_nedc() -> TestCycleRef:
        profile = []
        for _ in range(4):
            profile.extend([0]*11)
            profile.extend(np.linspace(0, 15, 4))
            profile.extend([15]*8)
            profile.extend(np.linspace(15, 0, 5))
            profile.extend([0]*21)
            profile.extend([0]*50) 
        return TestCycleRef("EURO_NEDC", 780, np.array(profile[:500]))

class DynamicTimeWarper:
    @staticmethod
    def compute_dtw_distance(series_a: np.ndarray, series_b: np.ndarray, 
                             window: int = 50) -> float:
        n, m = len(series_a), len(series_b)
        if n == 0 or m == 0: return float('inf')
        
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            start_j = max(1, i - window)
            end_j = min(m + 1, i + window)
            for j in range(start_j, end_j):
                cost = abs(series_a[i - 1] - series_b[j - 1])
                last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                dtw_matrix[i, j] = cost + last_min
                
        return dtw_matrix[n, m] / (n + m)

class DynoDetector:
    def __init__(self):
        self.steering_buffer = deque(maxlen=600)
        self.gps_speed_buffer = deque(maxlen=60)
        self.wheel_speed_buffer = deque(maxlen=60)
        self.accel_x_buffer = deque(maxlen=60)
        
    def analyze_kinematics(self, frame: Dict) -> Dict:
        self.steering_buffer.append(frame.get('steering_angle', 0.0))
        self.gps_speed_buffer.append(frame.get('gps_speed', 0.0))
        self.wheel_speed_buffer.append(frame.get('vehicle_speed', 0.0))
        self.accel_x_buffer.append(frame.get('accel_x', 0.0))
        
        if len(self.wheel_speed_buffer) < 10:
            return {"status": "CALIBRATING", "dyno_probability": 0.0, "steering_var": 0.0}
            
        steering_var = np.var(list(self.steering_buffer))
        steering_static = steering_var < 0.01
        
        avg_wheel_v = np.mean(list(self.wheel_speed_buffer))
        avg_gps_v = np.mean(list(self.gps_speed_buffer))
        
        is_stationary_geo = avg_gps_v < 1.0
        is_moving_wheels = avg_wheel_v > 10.0
        
        avg_accel = np.mean(np.abs(list(self.accel_x_buffer)))
        expected_accel = np.mean(np.abs(np.gradient(list(self.wheel_speed_buffer))))
        
        inertial_mismatch = (expected_accel > 1.0) and (avg_accel < 0.2)
        
        probability_dyno = 0.0
        details = []
        
        if steering_static:
            probability_dyno += 0.3
            details.append("STEERING_LOCKED")
        if is_moving_wheels and is_stationary_geo:
            probability_dyno += 0.6
            details.append("GPS_WHEEL_MISMATCH")
        if inertial_mismatch:
            probability_dyno += 0.4
            details.append("INERTIAL_DECOUPLING")
            
        return {
            "dyno_probability": min(1.0, probability_dyno),
            "steering_var": steering_var,
            "flags": details
        }

class CycleRecognitionEngine:
    def __init__(self):
        self.dtw_solver = DynamicTimeWarper()
        self.dyno_audit = DynoDetector()
        self.profiles = [RegulatoryProfiles.get_ftp75(), RegulatoryProfiles.get_nedc()]
        self.live_trace = deque(maxlen=500)
        self.match_threshold = 2.5
        
    def process_frame(self, frame: Dict) -> Dict:
        speed = frame.get('vehicle_speed', 0.0)
        self.live_trace.append(speed)
        
        kinematics = self.dyno_audit.analyze_kinematics(frame)
        matches = {}
        detected_cycle = "NONE"
        
        if len(self.live_trace) > 60:
            live_arr = np.array(self.live_trace)
            for profile in self.profiles:
                ref_slice = profile.velocity_profile[:len(live_arr)]
                dist = self.dtw_solver.compute_dtw_distance(live_arr, ref_slice)
                matches[profile.name] = dist
                if dist < self.match_threshold:
                    detected_cycle = profile.name

        return {
            "cycle_detected": detected_cycle,
            "match_scores": matches,
            "dyno_status": kinematics
        }
