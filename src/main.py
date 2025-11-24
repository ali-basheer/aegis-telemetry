"""
PROJECT: A.E.G.I.S. (Automated Emissions Governance & Intelligence System)
FILE: MAIN_ORCHESTRATOR
PROFILE: FORD_SCORPION_V8 (PRODUCTION)
STATUS: DEPLOYED

DESCRIPTION:
    The central nervous system of the forensic suite.
    Initializes the hardware link (Ford 6.7L), runs the physics kernels,
    and renders the live 'Hacker' dashboard with real-time EGT and Regen tracking.
"""

import sys
import os
import time
import logging
import threading
import queue
import uuid
from datetime import datetime
from typing import Dict, Any

# --- CRITICAL PATH SETUP ---
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- MODULE IMPORTS ---
from src.hardware.obd_interface import OBDInterface
from src.physics.combustion import ForensicCombustionValidator
# (Optional subsystems - kept for compatibility)
from src.core.database import ForensicDatabase

# --- LOGGING CONFIG ---
log_dir = os.path.join(project_root, "data", "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "system_debug.log"))]
)
logger = logging.getLogger("AEGIS.MAIN")

class TelemetryBus:
    """Thread-safe data interchange."""
    def __init__(self):
        self.current_snapshot: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def update(self, data: Dict[str, Any]):
        with self._lock:
            self.current_snapshot.update(data)

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.current_snapshot.copy()

class AEGIS_Kernel:
    """
    The Master Controller (Ford 6.7L Edition).
    """
    def __init__(self):
        self.logger = logger
        self.bus = TelemetryBus()
        self.running = False
        self.session_id = str(uuid.uuid4())[:8]
        
        print("\n[BOOT] INITIALIZING A.E.G.I.S. FORENSIC SUITE...")
        
        # 1. Hardware Layer (The Ford Interface)
        # We pass a config to prefer real hardware over simulation
        self.hal = OBDInterface({'simulation_mode': False})
        
        # 2. Physics Kernel (The V8 Validator)
        self.phys_combustion = ForensicCombustionValidator()
        
        # 3. Persistence (Optional)
        try:
            db_path = os.path.join(project_root, "data", "forensics.db")
            self.db = ForensicDatabase(db_path)
            self.db.start_session(self.session_id, "FORD_F250_6.7L", "CAL_UNKNOWN")
        except:
            self.logger.warning("Database unavailable. Running in volatile mode.")
            self.db = None

        print("[BOOT] SYSTEMS ONLINE. STARTING AUDIT LOOP.")

    def _acquisition_loop(self):
        """Polls the OBD interface for fresh data."""
        if not self.hal.connect():
            self.logger.error("Failed to initialize HAL.")
            return

        while self.running:
            # Fetch frame (Standard + Extended Ford PIDs)
            frame = self.hal.get_data()
            if frame:
                self.bus.update(frame)
            time.sleep(0.05) # 20Hz Polling

    def run_audit(self):
        """Main Analysis Loop (1Hz UI Refresh)."""
        self.running = True
        
        # Start Data Thread
        acq_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        acq_thread.start()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 1. Get Telemetry
                frame = self.bus.get_snapshot()
                if not frame:
                    time.sleep(0.1)
                    continue
                
                # 2. Run Physics Validation (The "Lie Detector")
                phys_res = self.phys_combustion.validate_snapshot(frame)
                frame.update(phys_res) # Merge results back to frame
                
                # 3. Log Evidence
                if self.db:
                    self.db.log_packet(self.session_id, frame)
                
                # 4. Update Dashboard
                self._render_dashboard(frame)
                
                # Frequency Control (Refresh UI every 0.5s)
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, 0.5 - elapsed))
                
        except KeyboardInterrupt:
            print("\n[STOP] User Halt. Closing Session.")
        finally:
            self.running = False
            if self.hal: self.hal.close()

    def _render_dashboard(self, f: Dict):
        """
        Renders the Hacker/Engineer UI specialized for Ford 6.7L.
        """
        # ANSI Colors
        os.system('cls' if os.name == 'nt' else 'clear')
        c_reset = "\033[0m"
        c_green = "\033[92m"
        c_red = "\033[91m"
        c_yellow = "\033[93m"
        c_cyan = "\033[96m"
        
        # --- STATUS FLAGS ---
        is_regen = f.get('regen_status', 0.0) > 0.5
        egt = f.get('exhaust_temp', 0.0)
        
        # Dynamic Header Color
        header_color = c_red if is_regen else c_cyan
        status_msg = "⚠ ACTIVE REGENERATION IN PROGRESS ⚠" if is_regen else "MONITORING - PASSIVE"
        
        print(f"{header_color}=== A.E.G.I.S. LIVE MONITOR (FORD SCORPION V8) ==={c_reset}")
        print(f"SESSION: {self.session_id} | HAL: {self.hal.status} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{header_color}STATUS:  {status_msg}{c_reset}")
        print("=" * 60)
        
        # --- ROW 1: ENGINE VITALS ---
        print(f"{c_yellow}[ ENGINE VITALS ]{c_reset}")
        print(f"RPM:   {f.get('rpm', 0):>5.0f}  | SPEED: {f.get('speed', 0):>3.0f} km/h | LOAD:  {f.get('load', 0):>3.0f}%")
        print(f"TEMP:  {f.get('temp', 0):>5.0f}C | FUEL:  {f.get('fuel_rate', 0):>5.1f} L/h  | MAF:   {f.get('maf', 0):>5.1f} g/s")
        print("-" * 60)
        
        # --- ROW 2: PHYSICS VALIDATION ---
        # Comparing Theoretical NOx (Zeldovich) vs Reported (Sensor)
        p_nox = f.get('physics_nox_ppm', 0)
        r_nox = f.get('actual_nox', 0)
        delta = f.get('delta_percent', 0) * 100
        
        delta_color = c_green if abs(delta) < 50 else c_red
        
        print(f"{c_yellow}[ PHYSICS VERIFICATION ]{c_reset}")
        print(f"THEORETICAL NOX (MODEL):  {p_nox:>6.1f} ppm")
        print(f"REPORTED NOX (SENSOR):    {r_nox:>6.1f} ppm")
        print(f"VARIANCE SIGMA:           {delta_color}{delta:>+6.1f}%{c_reset}")
        print("-" * 60)
        
        # --- ROW 3: AFTERTREATMENT (FORD SPECIFIC) ---
        print(f"{c_yellow}[ AFTERTREATMENT HEALTH ]{c_reset}")
        
        # EGT Alert Logic
        egt_color = c_red if egt > 600 else c_reset
        print(f"EGT (POST-TURBO):         {egt_color}{egt:>6.0f} C{c_reset}")
        
        # DPF Stats
        dist_reg = f.get('dist_since_regen', 0)
        print(f"DIST SINCE LAST REGEN:    {dist_reg:>6.1f} Miles")
        
        if is_regen:
             print(f"{c_red}>>> DPF BURN ACTIVE - DO NOT SHUTDOWN <<<{c_reset}")
        else:
             print(f"DPF STATUS:               Accumulating Soot")

        print("=" * 60)

if __name__ == "__main__":
    try:
        app = AEGIS_Kernel()
        app.run_audit()
    except Exception as e:
        logger.critical(f"FATAL SYSTEM ERROR: {e}")
        raise