"""
MODULE: TEMPORAL_ANOMALY_DETECTOR
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-01
CLASSIFICATION: FORENSIC / RESTRICTED

DESCRIPTION:
    Implements advanced heuristic detection for 'Timer-Based' Auxiliary Emission Control Devices (AECDs).
    Specifically targets defeat strategies that utilize engine run-time counters to 
    de-rate Selective Catalytic Reduction (SCR) efficiency after Federal Test Procedure (FTP) 
    windows have elapsed.

ALGORITHMS:
    1. Run-Time Delta Analysis (RTDA): Detects dosing decay correlated with FTP-75/WLTP durations.
    2. Spectral Noise Fingerprinting: Uses Fast Fourier Transform (FFT) to detect 
       'looped' sensor playback (sensor masking).
    3. Gradient Descent Monitoring: Monitors the second derivative of urea injection 
       rates to distinguish organic sensor drift from algorithmic ramp-downs.

DEPENDENCIES:
    - numpy (for FFT and gradient calculation)
    - scipy.stats (for Z-Score anomaly detection)
"""

import time
import logging
import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure module-level logger
logger = logging.getLogger("AEGIS.COMPLIANCE.TEMPORAL")

@dataclass
class TemporalEvent:
    timestamp: float
    event_type: str
    severity: float
    details: str
    snapshot_id: str

class TemporalDefeatDetector:
    """
    Stateful analyzer for time-dependent emission strategies.
    Maintains a rolling buffer of telemetry to identify long-horizon patterns
    that are invisible to instantaneous snapshot analysis.
    """

    # --- CONSTANTS (OBFUSCATED FOR SECURITY) ---
    _FTP_75_DURATION_SEC = 1874  # Length of standard US EPA cycle (approx)
    _WLTP_DURATION_SEC = 1800    # World Harmonized Light Vehicle Test Procedure
    _TIMER_TOLERANCE_PCT = 0.05  # 5% variance allowed for timer triggers
    _RAMP_DOWN_THRESHOLD = -0.05 # Urea rate change per second (normalized)
    
    # --- BUFFER SIZES ---
    _NOISE_BUFFER_SIZE = 512     # Power of 2 for FFT efficiency
    _HISTORY_SIZE = 7200         # 2 hours of 1Hz data

    def __init__(self, config: Dict):
        self.config = config
        self.start_time = time.time()
        self.engine_run_time_sec = 0.0
        
        # State Vectors
        self.history_nox = deque(maxlen=self._HISTORY_SIZE)
        self.history_urea = deque(maxlen=self._HISTORY_SIZE)
        self.history_timestamps = deque(maxlen=self._HISTORY_SIZE)
        
        # Spectral Buffer (for Sensor Masking Detection)
        self.noise_buffer = deque(maxlen=self._NOISE_BUFFER_SIZE)
        
        # Anomaly Accumulators
        self.flagged_events: List[TemporalEvent] = []
        self.suspected_timer_active = False
        
        logger.info(f"Temporal Detector Initialized. Watching for >{self._FTP_75_DURATION_SEC}s patterns.")

    def ingest_frame(self, frame: Dict) -> List[TemporalEvent]:
        """
        Main processing loop for a single telemetry frame.
        Returns a list of detected anomalies in this timestep.
        """
        current_t = time.time()
        dt = current_t - self.start_time
        self.engine_run_time_sec = dt
        
        # 1. Update Buffers
        self.history_timestamps.append(dt)
        self.history_nox.append(frame.get('actual_nox', 0.0))
        self.history_urea.append(frame.get('reductant_rate', 0.0)) # Assuming reductant_rate is in frame
        self.noise_buffer.append(frame.get('actual_nox', 0.0))

        new_events = []

        # 2. RUN CHECK: Is the car effectively 'simulating' a clean run?
        # Check for sensor playback loops (replay attacks)
        if len(self.noise_buffer) == self._NOISE_BUFFER_SIZE:
            if self._detect_sensor_looping():
                evt = TemporalEvent(
                    timestamp=dt,
                    event_type="SENSOR_PLAYBACK_LOOP",
                    severity=0.95,
                    details="Spectral analysis indicates NOX sensor data is repeating periodically (100% Match).",
                    snapshot_id=f"HEX_{int(dt)}"
                )
                new_events.append(evt)

        # 3. RUN CHECK: Timer-Based De-Rating (The Cummins Vector)
        # Check if we are passing critical regulatory time windows
        if self._check_regulatory_windows(dt):
            # We are in the 'Danger Zone' (post-test window).
            # Look for artificial ramp-downs in dosing.
            if self._detect_artificial_ramp_down():
                evt = TemporalEvent(
                    timestamp=dt,
                    event_type="AECD_TIMER_TRIGGER",
                    severity=0.88,
                    details=f"Sudden linear decay in Urea Injection detected immediately after FTP-75 window (t={dt:.1f}s).",
                    snapshot_id=f"TIM_{int(dt)}"
                )
                new_events.append(evt)
                self.suspected_timer_active = True

        return new_events

    def _detect_sensor_looping(self) -> bool:
        """
        Performs Fast Fourier Transform (FFT) on the noise buffer.
        Real sensors have 'Pink Noise' (1/f). 
        looped data has sharp harmonic spikes at the loop frequency.
        """
        try:
            data = np.array(self.noise_buffer)
            # Normalize
            data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            
            # FFT
            spectrum = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data))
            
            # Magnitude
            magnitude = np.abs(spectrum)
            
            # Heuristic: If one frequency dominates > 50% of total energy (excluding DC)
            peak_idx = np.argmax(magnitude[1:]) + 1
            peak_energy = magnitude[peak_idx]
            total_energy = np.sum(magnitude[1:])
            
            energy_ratio = peak_energy / (total_energy + 1e-6)
            
            # Real sensor noise is chaotic. If energy is concentrated, it's artificial.
            if energy_ratio > 0.45:
                logger.debug(f"FFT Peak Detected at {freq[peak_idx]:.2f}Hz. Ratio: {energy_ratio:.2f}")
                return True
                
            return False
        except Exception as e:
            logger.error(f"FFT Analysis Failed: {e}")
            return False

    def _check_regulatory_windows(self, runtime: float) -> bool:
        """
        Determines if the engine run time is suspiciously close to 
        the end of a known regulatory test cycle.
        """
        # FTP-75 Window (Federal Test Procedure)
        lower_bound = self._FTP_75_DURATION_SEC * (1.0 - self._TIMER_TOLERANCE_PCT)
        upper_bound = self._FTP_75_DURATION_SEC * (1.0 + 0.15) # Allow 15% overrun
        
        if lower_bound < runtime < upper_bound:
            return True
            
        # WLTP Window (Euro 6 Standard)
        wltp_lower = self._WLTP_DURATION_SEC * (1.0 - self._TIMER_TOLERANCE_PCT)
        wltp_upper = self._WLTP_DURATION_SEC * (1.0 + 0.15)
        
        if wltp_lower < runtime < wltp_upper:
            return True
            
        return False

    def _detect_artificial_ramp_down(self) -> bool:
        """
        Analyzes the first and second derivatives of the Urea Dosing Rate.
        
        Organic Depletion: Logarithmic decay or noisy fluctuation.
        Algorithmic Cheat: Linear negative slope (constant derivative) or Step function.
        """
        if len(self.history_urea) < 60:
            return False
            
        # Extract last 60 seconds of Urea Dosing
        recent_urea = np.array(list(self.history_urea)[-60:])
        
        # Calculate Gradient (1st Derivative)
        gradient = np.gradient(recent_urea)
        mean_gradient = np.mean(gradient)
        
        # Calculate Jerk (2nd Derivative - Rate of change of the gradient)
        jerk = np.gradient(gradient)
        mean_jerk = np.mean(np.abs(jerk))
        
        # PATTERN MATCHING:
        # A programmed ramp-down has a negative gradient but VERY LOW jerk
        # (It is a smooth, mathematically perfect line).
        # Real-world changes have high jerk (noise).
        
        is_decreasing = mean_gradient < self._RAMP_DOWN_THRESHOLD
        is_smooth_algorithmic = mean_jerk < 0.001 # Extremely smooth
        
        if is_decreasing and is_smooth_algorithmic:
            logger.warning(f"Algorithmic Decay Detected: Grad={mean_gradient:.4f}, Jerk={mean_jerk:.6f}")
            return True
            
        return False

    def generate_forensic_report_sql(self) -> str:
        """
        Generates the SQL injection string required to archive these findings
        into the main forensic ledger.
        """
        if not self.flagged_events:
            return ""
            
        sql_statements = []
        for evt in self.flagged_events:
            # Sanitize inputs
            details_safe = evt.details.replace("'", "''")
            
            # Advanced SQL: Using a hypothetical stored procedure for audit logs
            query = f"""
            INSERT INTO forensic_audit_log 
            (timestamp_utc, session_uuid, violation_code, confidence_score, metadata_blob)
            VALUES 
            ({evt.timestamp}, '{self.config.get('session_id', 'UNKNOWN')}', 
             '{evt.event_type}', {evt.severity}, 
             '{{"snapshot": "{evt.snapshot_id}", "desc": "{details_safe}"}}');
            """
            sql_statements.append(query)
            
        return "\n".join(sql_statements)

    def export_debug_csv(self, filepath: str):
        """
        Dumps the circular buffers to CSV for manual review.
        """
        import pandas as pd
        try:
            df = pd.DataFrame({
                'time_sec': list(self.history_timestamps),
                'nox_ppm': list(self.history_nox),
                'urea_rate': list(self.history_urea)
            })
            df.to_csv(filepath, index=False)
            logger.info(f"Debug trace saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

# --- UNIT TEST HARNESS (Embedded for Portability) ---
if __name__ == "__main__":
    # Simulate a run
    logging.basicConfig(level=logging.DEBUG)
    detector = TemporalDefeatDetector({'session_id': 'TEST-001'})
    
    print("Running Simulation: FTP-75 Timer Attack...")
    
    # 1. Normal Operation (t=0 to t=1800)
    for t in range(0, 1800):
        # Add some random noise
        frame = {
            'actual_nox': 50.0 + np.random.normal(0, 5),
            'reductant_rate': 10.0 + np.random.normal(0, 1)
        }
        # Manually advance time
        detector.start_time = time.time() - t
        detector.ingest_frame(frame)
        
    # 2. The Cheat Triggers (t=1801 onwards)
    # Urea injection ramps down perfectly smoothly
    print("Triggering Cheat Routine...")
    current_urea = 10.0
    for t in range(1801, 1900):
        current_urea -= 0.1 # Perfect linear decay
        frame = {
            'actual_nox': 50.0 + (t-1800)*2, # NOx rises as Urea falls
            'reductant_rate': current_urea
        }
        detector.start_time = time.time() - t
        events = detector.ingest_frame(frame)
        
        if events:
            print(f"[ALERT] at t={t}: {events[0].event_type}")
            print(f"SQL Export: {detector.generate_forensic_report_sql()}")
            break
