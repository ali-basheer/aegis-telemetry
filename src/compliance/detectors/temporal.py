"""
MODULE: TEMPORAL_ANOMALY_DETECTOR
PROFILE: FORD_SCORPION_V8 (BOSCH DENOXTRONIC LOGIC)

DESCRIPTION:
    Detects "Thermal Windowing" and "Cycle Beating" strategies.
    
    Manufacturers often claim they must disable emissions controls at low temperatures 
    to "protect the engine." This module validates those claims against the 
    known operating limits of the Bosch Denoxtronic 2.2 system.

    If the SCR system shuts down while the engine is within the 'Safe Operating Area' 
    (Exhaust > 200C, Ambient > -5C), it flags a 'THERMAL_WINDOW_VIOLATION'.
"""

import time
import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional
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
    """
    
    # --- BOSCH DENOXTRONIC 2.2 HARDWARE LIMITS ---
    # Sources: Bosch Technical Specs & EPA Dieselgate Reports
    # The system is physically capable of dosing within these limits.
    # Shutdowns inside this window are suspicious.
    MIN_SCR_TEMP_C = 190.0  # Urea crystallization limit
    MIN_AMBIENT_C = -7.0    # Heater operational limit
    
    # --- REGULATORY CYCLE TIMERS ---
    FTP_75_DURATION_SEC = 1874
    WARM_UP_WINDOW_SEC = 1200 # 20 minutes allowed for warm-up

    def __init__(self, config: Dict):
        self.config = config
        self.start_time = time.time()
        self.engine_run_time = 0.0
        self.is_hot_start = False
        
        # State Buffers
        self.temp_history = deque(maxlen=300) # 5 minutes
        self.dosing_history = deque(maxlen=300)
        
        # Hysteresis Tracking
        self.window_violation_start = None
        
        logger.info("[TEMPORAL] Detector Online. Calibrated for Denoxtronic 2.2 thermal limits.")

    def ingest_frame(self, frame: Dict) -> List[TemporalEvent]:
        """
        Main processing loop.
        """
        current_t = time.time()
        dt = current_t - self.start_time
        self.engine_run_time = dt
        
        # Extract Signals
        egt = frame.get('exhaust_temp', 0)
        ambient = frame.get('temp', 20) - 15 # Approximate ambient from coolant delta if sensor missing
        dosing_active = frame.get('reductant_rate', 0) > 0 or frame.get('regen_status', 0) > 0
        
        events = []

        # 1. CHECK: THERMAL WINDOW ABUSE
        # Rule: If EGT > 200C AND Ambient > -7C, you MUST be dosing.
        # Exception: Active Regeneration (where dosing pauses or changes mode).
        
        in_safe_zone = (egt > self.MIN_SCR_TEMP_C) and (ambient > self.MIN_AMBIENT_C)
        
        if in_safe_zone and not dosing_active:
            # We are in the "Safe Zone" but the system is OFF.
            if self.window_violation_start is None:
                self.window_violation_start = current_t
            
            duration = current_t - self.window_violation_start
            if duration > 60.0: # Grace period of 60s
                evt = TemporalEvent(
                    timestamp=current_t,
                    event_type="THERMAL_WINDOW_VIOLATION",
                    severity=0.9,
                    details=f"SCR Disabled in Safe Zone (EGT={egt:.0f}C). Duration: {duration:.1f}s",
                    snapshot_id=f"WIN_{int(dt)}"
                )
                if duration % 60 == 0: # Log every minute
                    events.append(evt)
        else:
            self.window_violation_start = None

        # 2. CHECK: HOT RESTART DEFEAT
        # Some cheats work by detecting a "Hot Start" (engine already warm)
        # and switching to a dirty map, knowing tests usually start cold.
        if dt < 30.0 and not self.is_hot_start:
            if frame.get('temp', 0) > 70:
                self.is_hot_start = True
                logger.info("[TEMPORAL] Hot Start Detected. Monitoring for 'Dirty Idle' strategy.")
        
        if self.is_hot_start and dt < 600:
            # If we are hot, we should be dosing IMMEDIATELY.
            # If 5 minutes pass with no dosing after a hot start -> CHEAT.
            if not dosing_active and dt > 300:
                evt = TemporalEvent(
                    timestamp=current_t,
                    event_type="HOT_START_BYPASS",
                    severity=0.95,
                    details="Engine warm but Emissions System dormant > 5 mins.",
                    snapshot_id=f"HOT_{int(dt)}"
                )
                events.append(evt)

        return events