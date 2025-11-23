"""
MODULE: FORENSIC_PERSISTENCE_LAYER
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-05
CLASSIFICATION: RESTRICTED / AUDIT-GRADE
"""

import sqlite3
import time
import json
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import asdict

# Configure module-level logger
logger = logging.getLogger("AEGIS.CORE.DB")

class IntegrityError(Exception):
    """Raised when chain hash verification fails."""
    pass

class ForensicDatabase:
    """
    ACID-compliant storage engine with enforced immutability.
    """
    
    _SCHEMA_VERSION = 2
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._configure_pragma()
        self._init_schema()
        
    def _configure_pragma(self):
        pragmas = [
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA foreign_keys = ON;",
            "PRAGMA encoding = 'UTF-8';",
            "PRAGMA temp_store = MEMORY;"
        ]
        for p in pragmas:
            self.cursor.execute(p)

    def _init_schema(self):
        logger.info("Verifying Forensic Schema Definition...")
        try:
            self.cursor.execute("BEGIN TRANSACTION;")
            
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_registry (
                session_uuid TEXT PRIMARY KEY,
                start_time_utc REAL NOT NULL,
                technician_id TEXT,
                vin_hash TEXT,
                ecu_calibration_id TEXT,
                notes TEXT
            );
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_chain (
                seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT NOT NULL,
                timestamp_utc REAL NOT NULL,
                rpm REAL,
                load_pct REAL,
                intake_temp_c REAL,
                maf_g_s REAL,
                nox_actual_ppm REAL,
                nox_expected_ppm REAL,
                urea_dosing_mg_s REAL,
                scr_efficiency REAL,
                row_hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                FOREIGN KEY(session_uuid) REFERENCES session_registry(session_uuid)
            );
            """)
            
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_chain_session ON telemetry_chain(session_uuid);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_chain_time ON telemetry_chain(timestamp_utc);")

            self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_chain_update
            BEFORE UPDATE ON telemetry_chain
            BEGIN
                SELECT RAISE(ABORT, 'FORENSIC ALERT: Attempted to modify immutable evidence ledger.');
            END;
            """)
            
            self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_chain_delete
            BEFORE DELETE ON telemetry_chain
            BEGIN
                SELECT RAISE(ABORT, 'FORENSIC ALERT: Attempted to destroy evidence.');
            END;
            """)
            
            self.cursor.execute("COMMIT;")
            logger.info("Schema deployed. Immutability locks active.")
            
        except sqlite3.Error as e:
            self.cursor.execute("ROLLBACK;")
            logger.critical(f"Schema Initialization Failed: {e}")
            raise

    def start_session(self, session_uuid: str, vin: str, calibration_id: str):
        vin_hash = hashlib.sha256(vin.encode()).hexdigest()
        query = """
        INSERT INTO session_registry 
        (session_uuid, start_time_utc, technician_id, vin_hash, ecu_calibration_id)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._lock:
            self.cursor.execute(query, (
                session_uuid, time.time(), "AEGIS_AUTO", vin_hash, calibration_id
            ))
    
    def get_last_hash(self, session_uuid: str) -> str:
        query = "SELECT row_hash FROM telemetry_chain WHERE session_uuid = ? ORDER BY seq_id DESC LIMIT 1"
        res = self.cursor.execute(query, (session_uuid,)).fetchone()
        if res:
            return res[0]
        return "0" * 64 

    def log_packet(self, session_uuid: str, data: Dict):
        with self._lock:
            prev_hash = self.get_last_hash(session_uuid)
            
            # --- FIX: Corrected Key Mappings ---
            # 'nox_actual' -> 'actual_nox'
            # 'intake_t' -> 'temp'
            
            actual_nox = data.get('actual_nox', 0.0)
            expected_nox = data.get('nox_expected', 0.0)
            
            canonical_str = (
                f"{prev_hash}|{data['t']}|{data['rpm']}|{data['load']}|"
                f"{actual_nox}|{expected_nox}"
            )
            
            current_hash = hashlib.sha256(canonical_str.encode()).hexdigest()
            
            query = """
            INSERT INTO telemetry_chain (
                session_uuid, timestamp_utc, 
                rpm, load_pct, intake_temp_c, maf_g_s,
                nox_actual_ppm, nox_expected_ppm, urea_dosing_mg_s, scr_efficiency,
                row_hash, prev_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            try:
                self.cursor.execute(query, (
                    session_uuid, data['t'],
                    data['rpm'], data['load'], data.get('temp', 20.0), data.get('maf', 0),
                    actual_nox, expected_nox, 
                    data.get('reductant_rate', 0), data.get('scr_efficiency_model', 0),
                    current_hash, prev_hash
                ))
            except sqlite3.IntegrityError as e:
                logger.error(f"Ledger Insertion Failed: {e}")

    def run_chain_verification(self, session_uuid: str) -> bool:
        logger.info(f"Auditing Chain Integrity for Session {session_uuid}...")
        query = """
        SELECT seq_id, timestamp_utc, rpm, load_pct, nox_actual_ppm, nox_expected_ppm, 
               row_hash, prev_hash
        FROM telemetry_chain
        WHERE session_uuid = ?
        ORDER BY seq_id ASC
        """
        rows = self.cursor.execute(query, (session_uuid,)).fetchall()
        expected_prev = "0" * 64
        
        for row in rows:
            if row['prev_hash'] != expected_prev:
                logger.critical(f"BROKEN LINK at SeqID {row['seq_id']}. Evidence Tampered.")
                return False
                
            canonical_str = (
                f"{row['prev_hash']}|{row['timestamp_utc']}|{row['rpm']}|{row['load_pct']}|"
                f"{row['nox_actual_ppm']}|{row['nox_expected_ppm']}"
            )
            recalc_hash = hashlib.sha256(canonical_str.encode()).hexdigest()
            
            if recalc_hash != row['row_hash']:
                logger.critical(f"HASH MISMATCH at SeqID {row['seq_id']}. Data Corrupted.")
                return False
            expected_prev = row['row_hash']
            
        logger.info(f"Audit Passed. {len(rows)} blocks verified.")
        return True

    def export_evidence_package(self, session_uuid: str) -> str:
        query = "SELECT * FROM telemetry_chain WHERE session_uuid = ? ORDER BY seq_id ASC"
        rows = self.cursor.execute(query, (session_uuid,)).fetchall()
        export_path = f"data/logs/evidence_{session_uuid}.jsonl"
        with open(export_path, 'w') as f:
            for row in rows:
                r_dict = dict(row)
                f.write(json.dumps(r_dict) + "\n")
        return export_path

    def analyze_cheat_windows(self, session_uuid: str) -> List[Dict]:
        query = """
        WITH calculated_deviation AS (
            SELECT 
                timestamp_utc,
                (nox_actual_ppm - nox_expected_ppm) as delta_ppm,
                AVG(nox_actual_ppm - nox_expected_ppm) OVER (
                    ORDER BY timestamp_utc 
                    ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
                ) as rolling_avg_delta
            FROM telemetry_chain
            WHERE session_uuid = ?
        )
        SELECT * FROM calculated_deviation
        WHERE rolling_avg_delta > 150.0; 
        """
        results = self.cursor.execute(query, (session_uuid,)).fetchall()
        return [dict(r) for r in results]
