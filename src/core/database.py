"""
MODULE: FORENSIC_PERSISTENCE_LAYER
STATUS: PRODUCTION_READY (SQLITE WAL MODE)

DESCRIPTION:
    A high-throughput logging engine designed for 10-50Hz telemetry ingestion.
    It uses SQLite in 'Write-Ahead Logging' (WAL) mode to allow concurrent 
    Read/Write operations, ensuring the dashboard never freezes while 
    evidential data is being committed to disk.

    SCHEMA:
    Optimized for Ford 6.7L Powerstroke data points (EGT, DPF, NOx).
"""

import sqlite3
import time
import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("AEGIS.DB")

class ForensicDatabase:
    """
    The 'Black Box' recorder.
    """
    def __init__(self, db_path: str):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.session_id = "UNKNOWN"

    def start_session(self, session_id: str, vin: str, cal_id: str):
        """
        Initializes the database connection and optimized schema.
        """
        self.session_id = session_id
        
        # Connect to SQLite
        # check_same_thread=False allows the acquisition thread to write
        # while the UI thread reads (Safe only in WAL mode)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # --- PERFORMANCE OPTIMIZATION (CRITICAL) ---
        # Enable Write-Ahead Logging.
        # This prevents "database is locked" errors during high-speed logging.
        self.conn.execute("PRAGMA journal_mode=WAL;") 
        self.conn.execute("PRAGMA synchronous=NORMAL;") # Reduces SD card wear
        
        self._init_schema()
        self._log_metadata(vin, cal_id)
        
        logger.info(f"[DB] Session {session_id} Recording Started. Mode: WAL")

    def _init_schema(self):
        """
        Creates the specialized table for Ford 6.7L Telemetry.
        """
        # 1. Main Telemetry Table
        # We store specific columns for fast querying, plus a JSON blob for extra data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                rpm REAL,
                load_pct REAL,
                coolant_temp REAL,
                maf_g_s REAL,
                fuel_rate_lph REAL,
                
                -- EMISSIONS DATA
                nox_actual_ppm REAL,
                nox_physics_ppm REAL,
                nox_delta_pct REAL,
                
                -- FORD AFTERTREATMENT (SCORPION V8)
                egt_temp_c REAL,
                regen_active INTEGER,
                dist_since_regen_mi REAL,
                
                -- METADATA
                raw_blob TEXT
            );
        """)
        
        # 2. Events Table (For AI Verdicts)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS forensic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                event_type TEXT,
                severity TEXT,
                details TEXT
            );
        """)
        self.conn.commit()

    def _log_metadata(self, vin, cal_id):
        """Log the 'Header' information for this drive cycle."""
        self.conn.execute("""
            INSERT INTO forensic_events (session_id, timestamp, event_type, severity, details)
            VALUES (?, ?, ?, ?, ?)
        """, (self.session_id, time.time(), "SESSION_START", "INFO", 
              json.dumps({"vin": vin, "calibration": cal_id})))
        self.conn.commit()

    def log_packet(self, session_id: str, frame: Dict[str, Any]):
        """
        Writes a single telemetry frame to the ledger.
        """
        if not self.conn:
            return

        try:
            # Extract Ford Specifics with defaults
            rpm = frame.get('rpm', 0)
            load = frame.get('load', 0)
            temp = frame.get('temp', 0)
            maf = frame.get('maf', 0)
            fuel = frame.get('fuel_rate', 0)
            
            nox_act = frame.get('actual_nox', 0)
            nox_phys = frame.get('physics_nox_ppm', 0)
            delta = frame.get('delta_percent', 0)
            
            egt = frame.get('exhaust_temp', 0)
            # Normalize boolean/float regen status to Integer (0/1)
            regen_stat = 1 if frame.get('regen_status', 0) > 0.5 else 0
            dist_regen = frame.get('dist_since_regen', 0)

            # Insert Record
            self.conn.execute("""
                INSERT INTO telemetry_frames 
                (session_id, timestamp, rpm, load_pct, coolant_temp, maf_g_s, fuel_rate_lph,
                 nox_actual_ppm, nox_physics_ppm, nox_delta_pct,
                 egt_temp_c, regen_active, dist_since_regen_mi, raw_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, time.time(), rpm, load, temp, maf, fuel,
                nox_act, nox_phys, delta,
                egt, regen_stat, dist_regen, json.dumps(frame)
            ))
            
            # Commit is implicit in WAL mode (periodically synced), 
            # but we force commit every frame for forensic integrity in this demo.
            # In extremely high load (100Hz+), you would move this to a batch commit.
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Write Failure: {e}")

    def log_event(self, event_type: str, severity: str, message: str):
        """Log distinct events (e.g. 'Defeat Device Detected')."""
        if not self.conn: return
        self.conn.execute("""
            INSERT INTO forensic_events (session_id, timestamp, event_type, severity, details)
            VALUES (?, ?, ?, ?, ?)
        """, (self.session_id, time.time(), event_type, severity, message))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("[DB] Ledger Sealed.")