"""
MODULE: CYCLE_RECOGNITION_FORENSICS
PROFILE: EPA / WLTP STANDARD DETECTION

DESCRIPTION:
    Real-time pattern matching engine that compares current driving behavior 
    against known government emissions test cycles (FTP-75, HWFET, WLTP).

    If a vehicle drives exactly like the EPA test script, it is highly likely 
    on a dynamometer. This is the primary trigger signal used by "Defeat Devices" 
    to activate their hidden compliance modes.

    ALGORITHM:
    Uses Pearson Correlation on a sliding window (60s) to detect 
    cycle segments regardless of absolute magnitude offsets.
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List, Tuple

# Configure module-level logger
logger = logging.getLogger("AEGIS.COMPLIANCE.DTW")

class CycleRecognitionEngine:
    """
    Forensic Matcher for Regulatory Driving Cycles.
    """
    
    # --- REFERENCE CYCLES (Simplified 1Hz Segments) ---
    # FTP-75 "Cold Start" Phase (First 60s approx pattern)
    # Stop-and-go city driving characteristic of the LA4 cycle.
    FTP_75_SIGNATURE = np.array([
        0, 0, 0, 0, 0, 10, 20, 25, 28, 30, 30, 32, 33, 34, 34, 32, 30, 25, 10, 0, 
        0, 0, 0, 15, 25, 35, 40, 45, 40, 30, 15, 0, 0, 0, 20, 25, 30, 25, 15, 0
    ], dtype=float)

    # HWFET "Highway" Phase (Steady state entry)
    HWFET_SIGNATURE = np.array([
        0, 10, 25, 40, 45, 48, 50, 52, 53, 55, 57, 58, 59, 60, 60, 60, 60, 59, 58, 60
    ], dtype=float)

    def __init__(self):
        self.window_size = 60 # Look at last 60 seconds
        self.speed_buffer = deque(maxlen=self.window_size)
        self.steering_buffer = deque(maxlen=self.window_size)
        
        # Thresholds
        self.match_threshold = 0.85 # 85% Correlation required to flag
        self.dyno_variance_limit = 0.5 # Degrees steering variance

    def process_frame(self, frame: dict) -> dict:
        """
        Ingests a telemetry frame and returns cycle match metrics.
        """
        # 1. UPDATE BUFFERS
        speed = frame.get('speed', 0.0) # km/h
        # Some OBD adapters don't support steering angle; default to None to skip check
        steer = frame.get('steering_angle', 0.0)
        
        self.speed_buffer.append(speed)
        self.steering_buffer.append(steer)
        
        # Default Output
        result = {
            "cycle_detected": "NONE",
            "match_scores": {"FTP_75": 0.0, "HWFET": 0.0},
            "dyno_status": {"is_dyno": False, "steering_var": 0.0, "dyno_probability": 0.0}
        }
        
        # Need full buffer to compare
        if len(self.speed_buffer) < self.window_size:
            return result

        # 2. RUN CORRELATION ANALYSIS (Pattern Matching)
        # Normalize live data to 0-1 range to match shape, not just amplitude
        live_trace = np.array(self.speed_buffer)
        
        # We perform correlations against hardcoded signatures
        # (Resampling signatures to match window size if necessary)
        
        # FTP-75 Check
        # We stretch the signature to fit the buffer for comparison
        ftp_score = self._correlation_score(live_trace, self.FTP_75_SIGNATURE)
        
        # HWFET Check
        hwfet_score = self._correlation_score(live_trace, self.HWFET_SIGNATURE)
        
        result["match_scores"]["FTP_75"] = ftp_score
        result["match_scores"]["HWFET"] = hwfet_score
        
        if ftp_score > self.match_threshold:
            result["cycle_detected"] = "FTP-75 (CITY)"
        elif hwfet_score > self.match_threshold:
            result["cycle_detected"] = "HWFET (HIGHWAY)"
            
        # 3. DYNO DETECTION (Kinematic Lock)
        # If speed > 20 km/h but steering hasn't moved -> Dyno
        avg_speed = np.mean(live_trace)
        steer_var = np.var(self.steering_buffer)
        result["dyno_status"]["steering_var"] = steer_var
        
        if avg_speed > 25.0:
            if steer_var < self.dyno_variance_limit:
                result["dyno_status"]["is_dyno"] = True
                result["dyno_status"]["dyno_probability"] = 0.99
                # If we are on a Dyno, we are almost certainly in a test cycle
                if result["cycle_detected"] == "NONE":
                    result["cycle_detected"] = "GENERIC_DYNO_RUN"
            else:
                # High steering variance = Real Road
                result["dyno_status"]["dyno_probability"] = 0.05
        
        return result

    def _correlation_score(self, live: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates Pearson Correlation Coefficient.
        Handles array resizing simple linear interpolation.
        """
        try:
            # Resize target to match live buffer length
            if len(target) != len(live):
                x_target = np.linspace(0, 1, len(target))
                x_live = np.linspace(0, 1, len(live))
                target_resampled = np.interp(x_live, x_target, target)
            else:
                target_resampled = target
                
            # Zero variance check (idle)
            if np.std(live) < 1e-3 or np.std(target_resampled) < 1e-3:
                return 0.0
                
            # Pearson Correlation
            corr = np.corrcoef(live, target_resampled)[0, 1]
            return max(0.0, float(corr))
            
        except Exception:
            return 0.0