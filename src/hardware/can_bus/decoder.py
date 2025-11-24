"""
MODULE: CAN_BUS_TOPOLOGY_FORENSICS (THE GHOST HUNTER)
PROFILE: FORD_SCORPION_V8 (HS-CAN / MS-CAN / J1939)

DESCRIPTION:
    A passive network analyzer that maps the 'Digital Fingerprint' of the vehicle.
    
    Modern "Delete Kits" and "Piggyback Tuners" often introduce anomalous 
    nodes onto the CAN bus or silence legitimate ones (like the Reductant Control Unit).

    This module scans for:
    1. ZOMBIE NODES: Expected ECUs that are silent (e.g., Unplugged EGR).
    2. GHOST NODES: Unknown CAN IDs (e.g., A 'Tuner Box' intercepting traffic).
    3. JITTER: Timing anomalies caused by signal interception latency.
"""

import time
import logging
from typing import Dict, List, Set

# Configure module-level logger
logger = logging.getLogger("AEGIS.HARDWARE.DECODER")

class ProprietarySniffer:
    """
    Decodes low-level bus topology to find hardware tampering.
    """
    
    # --- FORD 6.7L "KNOWN GOOD" TOPOLOGY (ISO 15765-4) ---
    # The Standard ID Map for a stock 2011+ Super Duty
    EXPECTED_NODES = {
        0x7E8: "PCM (Powertrain Control Module)",
        0x7E9: "TCM (Transmission Control Module)",
        0x7EA: "GPCM (Glow Plug Control Module)", 
        0x7EB: "DCU (Dosing Control Unit / Reductant)",
        0x7EC: "NOx_A (Upstream Sensor)",
        0x7ED: "NOx_B (Downstream Sensor)"
    }

    # --- J1939 "HEAVY DUTY" MAP (Packet Headers) ---
    # Used on F-350/450/550 Chassis Cabs
    EXPECTED_PGNS = {
        61444: "EEC1 (Engine Speed/Torque)",
        64984: "AT1T (Aftertreatment Gas)",
        65262: "ET1 (Engine Temperature)",
        64923: "AT1IC1 (Aftertreatment Injector)"
    }

    def __init__(self):
        self.seen_ids: Set[int] = set()
        self.seen_pgns: Set[int] = set()
        self.frame_intervals: Dict[int, List[float]] = {}
        self.last_seen: Dict[int, float] = {}
        
    def ingest_frame(self, frame_id: int, protocol: str = "CAN"):
        """
        Logs a raw CAN ID to the topology map.
        """
        current_t = time.time()
        
        # 1. TRACK PRESENCE
        self.seen_ids.add(frame_id)
        
        # 2. TRACK TIMING (Jitter Detection)
        # Real ECUs are crystal-controlled and very precise. 
        # Tuners (cheap microcontrollers) drift.
        if frame_id in self.last_seen:
            delta = current_t - self.last_seen[frame_id]
            if frame_id not in self.frame_intervals:
                self.frame_intervals[frame_id] = []
            self.frame_intervals[frame_id].append(delta)
            # Keep buffer small
            if len(self.frame_intervals[frame_id]) > 50:
                self.frame_intervals[frame_id].pop(0)
                
        self.last_seen[frame_id] = current_t

    def generate_topology_report(self) -> Dict:
        """
        Compare seen nodes against the 'Golden Image' of a stock truck.
        """
        report = {
            "status": "CLEAN",
            "missing_nodes": [],
            "ghost_nodes": [],
            "jitter_anomalies": []
        }
        
        # 1. CHECK FOR MISSING HARDWARE (HARD PART DELETE)
        # We expect the PCM and TCM to always be there.
        # If DCU (0x7EB) is missing on a Diesel, the SCR is likely unplugged.
        critical_nodes = [0x7E8, 0x7E9, 0x7EB]
        
        for node in critical_nodes:
            if node not in self.seen_ids:
                name = self.EXPECTED_NODES.get(node, "UNKNOWN")
                report["missing_nodes"].append(f"{name} [ID: {hex(node)}]")
                if node == 0x7EB:
                    report["status"] = "HARDWARE_TAMPER (DEF_MODULE_OFFLINE)"

        # 2. CHECK FOR GHOST HARDWARE (TUNER BOX)
        # Any ID in the 0x7Ex range that isn't in our map is suspicious.
        # Tuners often occupy 0x7E7 or 0x7EF to avoid collision.
        for seen in self.seen_ids:
            if 0x7E0 <= seen <= 0x7EF:
                if seen not in self.EXPECTED_NODES:
                    report["ghost_nodes"].append(f"UNIDENTIFIED_ECU [ID: {hex(seen)}]")
                    report["status"] = "HARDWARE_TAMPER (UNKNOWN_DEVICE_ON_BUS)"

        # 3. CHECK TIMING INTEGRITY (MAN-IN-THE-MIDDLE)
        # A "Piggyback" intercepts signals and re-transmits them.
        # This adds latency variance (jitter).
        # A healthy ECU has <5% variance. A Tuner often has >20%.
        for node_id, intervals in self.frame_intervals.items():
            if len(intervals) > 10:
                avg = sum(intervals) / len(intervals)
                variance = sum((x - avg) ** 2 for x in intervals) / len(intervals)
                std_dev = variance ** 0.5
                
                jitter_pct = (std_dev / avg) * 100.0
                
                if jitter_pct > 15.0: # 15% Jitter Threshold
                    name = self.EXPECTED_NODES.get(node_id, "UNKNOWN")
                    report["jitter_anomalies"].append(
                        f"{name} [ID: {hex(node_id)}] Jitter: {jitter_pct:.1f}% (Signal Injection?)"
                    )
                    if report["status"] == "CLEAN":
                        report["status"] = "SIGNAL_INTEGRITY_FAIL"

        return report

# --- STANDALONE SCANNER MODE ---
if __name__ == "__main__":
    print("--- AEGIS TOPOLOGY SCANNER (FORD 6.7L) ---")
    sniffer = ProprietarySniffer()
    
    # Simulate a "Delete Tuner" Scenario
    # 1. PCM is present (0x7E8)
    # 2. TCM is present (0x7E9)
    # 3. DCU is MISSING (0x7EB) - User unplugged it
    # 4. Tuner Box is injecting fake signals (0x7E5)
    
    print("[SIM] Injecting Traffic...")
    sim_traffic = [0x7E8, 0x7E9, 0x7EA, 0x7E5] # 0x7E5 is the Ghost
    
    for _ in range(100):
        for node in sim_traffic:
            sniffer.ingest_frame(node)
            
    report = sniffer.generate_topology_report()
    
    print(f"\nSTATUS: {report['status']}")
    if report['missing_nodes']:
        print(f"MISSING: {report['missing_nodes']}")
    if report['ghost_nodes']:
        print(f"DETECTED: {report['ghost_nodes']}")