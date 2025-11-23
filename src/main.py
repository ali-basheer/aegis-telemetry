"""
PROJECT: A.E.G.I.S. (Automated Emissions Governance & Intelligence System)
FILE: MAIN_ORCHESTRATOR
AUTHOR: ALI BASHEER
STATUS: DEPLOYED

DESCRIPTION:
    The central nervous system of the forensic suite.
    Initializes all 10 subsystems, establishes the cryptographic session,
    and executes the Real-Time Telemetry Pipeline.
"""

import sys
import os

# --- CRITICAL PATH INJECTION ---
# This block ensures Python can find the 'src' module regardless of where the script is run from.
# It calculates the path to the 'aegis_telemetry' root folder and adds it to sys.path.

current_file = os.path.abspath(__file__)            # .../aegis_telemetry/src/main.py
src_dir = os.path.dirname(current_file)             # .../aegis_telemetry/src
project_root = os.path.dirname(src_dir)             # .../aegis_telemetry

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NOW we can import local modules safely
import time
import logging
import yaml
import threading
import queue
import uuid
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

# --- IMPORT MODULES ---
# 1. Hardware
from src.hardware.obd_interface import OBDInterface
from src.hardware.can_bus.decoder import ProprietarySniffer

# 2. Physics
from src.physics.combustion import ForensicCombustionValidator
from src.physics.aftertreatment import AftertreatmentForensics
from src.physics.stoichiometry import ClosedLoopAuditor
from src.physics.sensor_fusion import SensorIntegrityMonitor

# 3. Compliance
from src.compliance.detectors.temporal import TemporalDefeatDetector
from src.compliance.reference_validator import ShadowGovernor
from src.compliance.cycle_recognition import CycleRecognitionEngine

# 4. Intelligence
from src.ai.meta_classifier import AIEngine

# 5. Core
from src.core.database import ForensicDatabase

# --- LOGGING CONFIG ---
# Ensure log directory exists
log_dir = os.path.join(project_root, "data", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "system_debug.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AEGIS.MAIN")

class TelemetryBus:
    """
    Thread-safe data interchange bus.
    Aggregates async CAN frames and sync OBD polls into a unified snapshot.
    """
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=100)
        self.current_snapshot: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def update_signal(self, key: str, value: Any):
        with self._lock:
            self.current_snapshot[key] = value

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.current_snapshot.copy()

class AEGIS_Kernel:
    """
    The Master Controller.
    """
    
    def __init__(self):
        self.logger = logger
        # Load config relative to project root
        config_path = os.path.join(project_root, "config", "settings.yaml")
        self.config = self._load_config(config_path)
        
        self.bus = TelemetryBus()
        self.running = False
        self.session_id = str(uuid.uuid4())
        
        # --- SUBSYSTEM INITIALIZATION ---
        self.logger.info("--- INITIALIZING FORENSIC SUBSYSTEMS ---")
        
        # 1. Persistence Layer (Must be first for Chain of Custody)
        # Fix relative path for DB
        db_rel_path = self.config['database']['path']
        db_path = os.path.join(project_root, db_rel_path)
        
        self.db = ForensicDatabase(db_path)
        self.db.start_session(self.session_id, "VIN_UNKNOWN", "CAL_ID_INIT")
        self.logger.info(f"Session Registry Created: {self.session_id}")

        # 2. Hardware Layer
        self.hal = OBDInterface(self.config['hardware'])
        self.sniffer = ProprietarySniffer()
        
        # 3. Physics Kernels
        self.phys_combustion = ForensicCombustionValidator()
        self.phys_scr = AftertreatmentForensics()
        self.phys_stoich = ClosedLoopAuditor()
        self.phys_ekf = SensorIntegrityMonitor()
        
        # 4. Compliance Engines
        self.comp_temporal = TemporalDefeatDetector({'session_id': self.session_id})
        self.comp_maps = ShadowGovernor()
        self.comp_cycles = CycleRecognitionEngine()
        
        # 5. AI Layer
        model_rel_path = self.config['ai']['model_path']
        model_path = os.path.join(project_root, model_rel_path)
        self.ai_judge = AIEngine(model_path)
        
        self.logger.info("ALL SYSTEMS ONLINE. READY FOR AUDIT.")

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.critical(f"Config file not found at {path}")
            raise

    def _acquisition_loop(self):
        """
        Runs on a separate thread.
        Polls hardware at high frequency (10Hz-50Hz) to ensure no aliasing.
        """
        self.logger.info("Acquisition Thread Started.")
        if not self.hal.connect():
            self.logger.critical("HAL Connection Failed. Running in Virtual Mode.")
            
        while self.running:
            # 1. Poll OBD/Emulator
            raw_data = self.hal.get_data()
            if raw_data:
                for k, v in raw_data.items():
                    self.bus.update_signal(k, v)
            
            time.sleep(0.05) # 20Hz

    def run_audit(self):
        """
        The Main Event Loop (1Hz - 5Hz Analysis Rate).
        """
        self.running = True
        
        # Start Data Acquisition Thread
        acq_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        acq_thread.start()
        
        cycle_count = 0
        window_buffer = []
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 1. FETCH SNAPSHOT
                frame = self.bus.get_snapshot()
                if not frame:
                    time.sleep(0.1)
                    continue
                
                # Add timestamp
                frame['t'] = time.time()
                dt = 1.0 # Simplified delta t
                
                # --- LAYER 1: PHYSICS VALIDATION ---
                # A. Combustion (Thermodynamics)
                comb_res = self.phys_combustion.validate_snapshot(frame)
                frame['nox_raw_est'] = comb_res['physics_nox_ppm']
                frame['combustion_delta'] = comb_res['delta_percent']
                
                # B. Aftertreatment (Chemistry)
                scr_res = self.phys_scr.audit_frame(frame, dt)
                frame['nox_expected'] = scr_res['sim_tailpipe_nox']
                frame['scr_efficiency_model'] = scr_res['sim_efficiency']
                
                # C. Stoichiometry (Gas Dynamics)
                stoich_res = self.phys_stoich.audit_cycle(frame)
                frame['stoich_error'] = stoich_res['maf_delta_pct'] / 100.0
                
                # D. Sensor Fusion (EKF)
                ekf_res = self.phys_ekf.audit_frame(frame, dt)
                frame['nis_score'] = ekf_res['nis_score']
                
                # --- LAYER 2: COMPLIANCE CHECKS ---
                # E. Reference Maps (Calibration)
                map_res = self.comp_maps.audit_cycle(frame)
                frame['urea_dev'] = map_res['urea_deviation_pct'] / 100.0
                
                # F. Cycle Recognition (DTW)
                dtw_res = self.comp_cycles.process_frame(frame)
                # Get lowest distance score
                min_dist = min(dtw_res['match_scores'].values()) if dtw_res['match_scores'] else 100.0
                frame['dtw_dist'] = min_dist
                frame['steer'] = dtw_res['dyno_status']['steering_var']
                
                # G. Temporal (Timer Exploits)
                temp_events = self.comp_temporal.ingest_frame(frame)
                
                # --- LAYER 3: INTELLIGENCE ---
                # Accumulate window for AI
                window_buffer.append(frame)
                if len(window_buffer) > 10: window_buffer.pop(0)
                
                # Run AI Verdict
                ai_verdict = self.ai_judge.analyze_session(window_buffer)
                
                # --- LAYER 4: PERSISTENCE ---
                # Log packet to Immutable Ledger
                self.db.log_packet(self.session_id, frame)
                
                # --- DASHBOARD ---
                self._render_cli_dashboard(frame, ai_verdict, dtw_res, ekf_res)
                
                # Frequency Control
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, 1.0 - elapsed) # 1Hz Audit Rate
                time.sleep(sleep_time)
                cycle_count += 1
                
        except KeyboardInterrupt:
            self.logger.info("User requested shutdown.")
        finally:
            self.running = False
            self.logger.info(f"Session {self.session_id} Finalized. Evidence Chain Sealed.")

    def _render_cli_dashboard(self, frame, ai, dtw, ekf):
        """
        Draws the Hacker/Engineer UI.
        """
        # Clear screen (ANSI)
        print("\033[2J\033[H", end="")
        
        c_green = "\033[92m"
        c_red = "\033[91m"
        c_yellow = "\033[93m"
        c_reset = "\033[0m"
        
        verdict_color = c_green
        if ai['verdict'] != "COMPLIANT": verdict_color = c_red
        
        print(f"{c_yellow}=== A.E.G.I.S. FORENSIC MONITOR (LIVE) ==={c_reset}")
        print(f"SESSION: {self.session_id[:8]}... | TIME: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        print(f"ENGINE STATE   | RPM: {frame.get('rpm',0):4.0f} | LOAD: {frame.get('load',0):3.0f}% | SPEED: {frame.get('vehicle_speed',0):3.0f} km/h")
        print(f"THERMODYNAMICS | T_ENG: {frame.get('coolant_temp',0):3.0f}C | T_EXH: {frame.get('exhaust_temp',0):3.0f}C | EKF_NIS: {ekf['nis_score']:.2f}")
        print("-" * 60)
        print(f"EMISSIONS      | NOx ACT: {frame.get('actual_nox',0):4.0f} ppm | NOx EXP: {frame.get('nox_expected',0):4.0f} ppm")
        print(f"UREA DOSING    | ACT: {frame.get('reductant_rate',0):4.1f} mg/s | MAP DEV: {frame.get('urea_dev',0)*100:+.1f}%")
        print("-" * 60)
        print(f"CYCLE DETECT   | MATCH: {dtw['cycle_detected']} (Dist: {frame['dtw_dist']:.2f})")
        print(f"DYNO STATUS    | PROB: {dtw['dyno_status']['dyno_probability']*100:.0f}% | STEER VAR: {frame['steer']:.4f}")
        print("-" * 60)
        print(f"AI JUDGEMENT   | {verdict_color}{ai['verdict']} (Prob: {ai['cheat_probability']:.2f}){c_reset}")
        if ai['contributing_factors']:
            print(f"ALERTS         | {ai['contributing_factors']}")
        print("=" * 60)

if __name__ == "__main__":
    # Boot Sequence
    try:
        app = AEGIS_Kernel()
        app.run_audit()
    except Exception as e:
        logger.critical(f"FATAL SYSTEM ERROR: {e}")
        raise
