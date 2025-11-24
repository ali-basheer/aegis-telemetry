"""
MODULE: HEADLESS_ORCHESTRATOR (SYSTEM SERVICE)
PROFILE: FORD SCORPION V8 (DEPLOYED)

DESCRIPTION:
    The 'Silent Mode' runner designed for embedded deployment (Raspberry Pi/Telematics Box).
    
    Unlike main.py (which has a UI), this script runs as a background daemon.
    It prioritizes:
    1. UPTIME: Auto-reconnection logic if the OBD cable is bumped.
    2. POWER: 'Sleep Mode' when the engine is off (RPM=0) to save truck battery.
    3. INTEGRITY: Flushes the WAL database to disk before shutting down.

    USAGE:
    Run via systemd or cron: 'python src/core/orchestrator.py'
"""

import time
import logging
import logging.config
import yaml
import os
import signal
import sys
from datetime import datetime

# Import the AEGIS Stack
from src.hardware.obd_interface import OBDInterface
from src.physics.combustion import ForensicCombustionValidator
from src.physics.aftertreatment import AftertreatmentForensics
from src.physics.stoichiometry import ClosedLoopAuditor
from src.physics.sensor_fusion import SensorIntegrityMonitor
from src.compliance.detectors.temporal import TemporalDefeatDetector
from src.compliance.reference_validator import ShadowGovernor
from src.compliance.cycle_recognition import CycleRecognitionEngine
from src.ai.meta_classifier import AIEngine
from src.core.database import ForensicDatabase

# --- HEADLESS LOGGER CONFIG ---
# We can't print to screen, so we must ensure file logging is robust
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "aegis_service.log")),
        logging.StreamHandler(sys.stdout) # For systemctl status viewing
    ]
)
logger = logging.getLogger("AEGIS.DAEMON")

class HeadlessOrchestrator:
    """
    The long-running service manager.
    """
    
    def __init__(self):
        self.running = True
        self.sleep_mode = False
        self.last_engine_activity = time.time()
        
        # Load Configuration
        self.config = self._load_config()
        
        # Initialize Subsystems
        logger.info("--- STARTING AEGIS FORENSIC DAEMON ---")
        self.hal = OBDInterface(self.config.get('hardware', {}))
        self.db = ForensicDatabase(self.config['database']['path'])
        
        # Initialize Physics Kernels
        self.phys_comb = ForensicCombustionValidator()
        self.phys_scr = AftertreatmentForensics()
        self.phys_stoich = ClosedLoopAuditor()
        self.phys_ekf = SensorIntegrityMonitor()
        
        # Initialize Detectors
        self.comp_temporal = TemporalDefeatDetector({})
        self.comp_maps = ShadowGovernor()
        self.comp_cycles = CycleRecognitionEngine()
        self.ai = AIEngine(self.config['intelligence']['model_path'])
        
        # Start Session
        self.session_id = datetime.now().strftime("SVC_%Y%m%d_%H%M%S")
        self.db.start_session(self.session_id, "HEADLESS_UNIT", "AUTO_DETECT")
        
        # Signal Handlers (for graceful shutdown via systemctl stop)
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _load_config(self):
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_path, "config", "settings.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _shutdown(self, signum, frame):
        logger.warning("SHUTDOWN SIGNAL RECEIVED. FLUSHING BUFFERS...")
        self.running = False

    def _check_power_management(self, rpm):
        """
        Smart Sleep: If engine is off > 5 mins, slow down polling to 1Hz.
        """
        if rpm > 200:
            self.last_engine_activity = time.time()
            if self.sleep_mode:
                logger.info("WAKE UP EVENT: Engine Started.")
                self.sleep_mode = False
        else:
            if not self.sleep_mode and (time.time() - self.last_engine_activity > 300):
                logger.info("SLEEP EVENT: Engine Inactive > 5m. Entering Power Save.")
                self.sleep_mode = True

    def run(self):
        """
        The Infinite Loop.
        """
        # Initial Connection
        while self.running and not self.hal.connect():
            logger.error("HAL Connection Failed. Retrying in 10s...")
            time.sleep(10)

        while self.running:
            try:
                loop_start = time.time()
                
                # 1. ACQUIRE DATA
                frame = self.hal.get_data()
                if not frame:
                    # Watchdog: Connection lost?
                    logger.warning("HAL returned empty frame. Reconnecting...")
                    self.hal.close()
                    time.sleep(2)
                    self.hal.connect()
                    continue

                # 2. POWER MANAGEMENT
                self._check_power_management(frame.get('rpm', 0))
                if self.sleep_mode:
                    time.sleep(1.0) # Slow poll
                    continue

                # 3. RUN PHYSICS PIPELINE
                # A. Combustion
                comb_res = self.phys_comb.validate_snapshot(frame)
                frame.update(comb_res)
                
                # B. Aftertreatment (SCR)
                scr_res = self.phys_scr.audit_frame(frame, 0.1)
                frame.update(scr_res)
                
                # C. Stoichiometry
                stoich_res = self.phys_stoich.audit_cycle(frame)
                frame.update(stoich_res)
                
                # D. Sensor Fusion
                ekf_res = self.phys_ekf.audit_frame(frame, 0.1)
                frame.update(ekf_res)

                # 4. RUN COMPLIANCE PIPELINE
                # E. Reference Maps
                map_res = self.comp_maps.audit_cycle(frame)
                frame.update(map_res)
                
                # F. Cycle Recognition
                cycle_res = self.comp_cycles.process_frame(frame)
                frame['dtw_dist'] = cycle_res['match_scores']['FTP_75'] # Log the score
                
                # G. Temporal Checks
                temp_events = self.comp_temporal.ingest_frame(frame)
                for evt in temp_events:
                    logger.warning(f"COMPLIANCE ALERT: {evt.event_type} - {evt.details}")
                    self.db.log_event(evt.event_type, "HIGH", evt.details)

                # 5. AI ADJUDICATION
                # Create a rolling window (simplified for headless)
                ai_verdict = self.ai.analyze_frame(frame)
                if ai_verdict['is_cheating']:
                    msg = f"AI DETECTED FRAUD ({ai_verdict['confidence']:.1f}%): {ai_verdict['evidence']}"
                    logger.critical(msg)
                    self.db.log_event("AI_FLAG", "CRITICAL", msg)

                # 6. PERSISTENCE
                self.db.log_packet(self.session_id, frame)

                # 7. RATE LIMITING
                # Target 10Hz for High Res Logging
                elapsed = time.time() - loop_start
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)

            except Exception as e:
                logger.error(f"CRASH IN LOOP: {e}", exc_info=True)
                time.sleep(1)

        # Cleanup
        self.hal.close()
        self.db.close()
        logger.info("Daemon Stopped Gracefully.")

if __name__ == "__main__":
    daemon = HeadlessOrchestrator()
    daemon.run()