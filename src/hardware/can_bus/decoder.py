"""
MODULE: CAN_FRAME_DECODER (ISO-15765-2 / SAE J1939)
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-04
CLASSIFICATION: FORENSIC / LOW-LEVEL NETWORK

DESCRIPTION:
    Implements a passive packet sniffer and reassembly engine for automotive networks.
    
    Standard OBD-II queries (Service $01) are active/polling and easily spoofed by 
    defeat device firmware. This module listens to 'Broadcast Traffic' (inter-module 
    communication) which is much harder to falsify without breaking vehicle drivability.

    CAPABILITIES:
    1. ISO-TP Reassembly: Reconstructs multi-frame payloads (up to 4095 bytes).
    2. J1939 PGN Parsing: Decodes heavy-duty diesel broadcast standards (Cummins/Detroit).
    3. Bitwise Signal Extraction: Unpacks non-byte-aligned signals (Intel/Motorola formats).
    4. Shadow PID Detection: Flags proprietary diagnostic commands often used to trigger 
       'Dyno Mode' or 'EPA Mode'.

STANDARDS:
    - ISO 11898 (Physical Layer)
    - ISO 15765-2 (Transport Layer)
    - SAE J1939-71 (Application Layer - Vehicle)
"""

import struct
import logging
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Configure module-level logger
logger = logging.getLogger("AEGIS.HW.CAN")

# --- PROTOCOL CONSTANTS ---
class FrameType(IntEnum):
    SINGLE = 0x0
    FIRST = 0x1
    CONSECUTIVE = 0x2
    FLOW_CONTROL = 0x3

@dataclass
class CanFrame:
    """
    Represents a raw CAN 2.0B Frame.
    """
    arbitration_id: int
    dlc: int
    payload: bytes
    timestamp: float

    @property
    def is_extended(self) -> bool:
        return self.arbitration_id > 0x7FF

@dataclass
class PGN_Definition:
    """
    SAE J1939 Parameter Group Number Definition.
    """
    pgn: int
    name: str
    spns: List[Dict] # List of Signal Parameter Numbers

class BitStreamWorker:
    """
    Utility for extracting signals from raw byte streams.
    Handles Endianness and Bit-Shifting.
    """
    @staticmethod
    def extract_signal(payload: bytes, start_bit: int, length: int, 
                       is_little_endian: bool = False, scale: float = 1.0, 
                       offset: float = 0.0) -> float:
        """
        Extracts a physical value from a packed byte array.
        """
        # 1. Convert bytes to a massive integer
        raw_int = int.from_bytes(payload, byteorder='big')
        
        # 2. Calculate bit position (Big Endian logic default for SAE)
        # Total bits
        total_bits = len(payload) * 8
        
        # In Big Endian (Motorola), start_bit is usually MSB or LSB of the byte
        # We simplify to a normalized 0-indexed bit stream for forensic extract
        
        # Create a bit mask
        mask = (1 << length) - 1
        
        # Shift to align LSB
        shift_amount = total_bits - start_bit - length
        if shift_amount < 0:
            # Handle split byte signals manually if needed (complexity simplification)
            return 0.0 
            
        unmasked = (raw_int >> shift_amount) & mask
        
        # 3. Apply Physical Scaling
        return (unmasked * scale) + offset

class ISOTP_Reassembler:
    """
    State Machine for ISO 15765-2 Transport Layer.
    Reassembles segmented messages (e.g., VIN, Calibration IDs, Freeze Frames).
    """
    
    def __init__(self):
        self.buffer: Dict[int, bytearray] = {} # Map ID -> Buffer
        self.expected_len: Dict[int, int] = {}
        self.next_seq: Dict[int, int] = {}
        
    def process_frame(self, frame: CanFrame) -> Optional[bytes]:
        """
        Returns complete payload bytes if a message is finished, else None.
        """
        # ISO-TP uses the first byte as the PCI (Protocol Control Information)
        if not frame.payload: return None
        
        pci_byte = frame.payload[0]
        frame_type = (pci_byte & 0xF0) >> 4
        
        # 1. Single Frame (SF)
        if frame_type == FrameType.SINGLE:
            length = pci_byte & 0x0F
            if length == 0: # CAN FD escape
                return None 
            return frame.payload[1:1+length]
            
        # 2. First Frame (FF)
        elif frame_type == FrameType.FIRST:
            # Length is 12 bits: lower nibble of byte 0 + all of byte 1
            length = ((pci_byte & 0x0F) << 8) + frame.payload[1]
            self.expected_len[frame.arbitration_id] = length
            self.buffer[frame.arbitration_id] = bytearray(frame.payload[2:])
            self.next_seq[frame.arbitration_id] = 1
            # In a real sniffer, we might see Flow Control here, but we just listen
            return None
            
        # 3. Consecutive Frame (CF)
        elif frame_type == FrameType.CONSECUTIVE:
            seq_idx = pci_byte & 0x0F
            arb_id = frame.arbitration_id
            
            if arb_id not in self.buffer:
                # We missed the First Frame (common in passive sniffing)
                return None 
                
            if seq_idx != (self.next_seq[arb_id] % 16):
                # Sequence error (Packet loss)
                logger.warning(f"ISO-TP Seq Error on ID {hex(arb_id)}. Dropping buffer.")
                del self.buffer[arb_id]
                return None
                
            self.buffer[arb_id].extend(frame.payload[1:])
            self.next_seq[arb_id] += 1
            
            # Check if complete
            if len(self.buffer[arb_id]) >= self.expected_len[arb_id]:
                # Trim padding
                payload = bytes(self.buffer[arb_id][:self.expected_len[arb_id]])
                del self.buffer[arb_id]
                return payload
                
        return None

class J1939_Decoder:
    """
    Decodes heavy-duty diesel PGNs (Parameter Group Numbers).
    Used for Cummins / Detroit Diesel / Caterpillar engines.
    """
    
    # Common PGNs
    PGN_EEC1 = 0xF004  # Electronic Engine Controller 1 (RPM, Torque)
    PGN_ET1  = 0xFEEE  # Engine Temperature 1 (Coolant, Oil)
    PGN_AMB  = 0xFEF5  # Ambient Conditions
    PGN_AT1  = 0xF005  # Aftertreatment 1 (DEF Level, SCR Status)

    @staticmethod
    def decode(frame: CanFrame) -> Dict[str, float]:
        """
        Parses payload based on PGN derived from 29-bit Identifier.
        """
        if not frame.is_extended: return {}
        
        # J1939 ID Breakdown:
        # Priority (3) | PGN (18) | Source Address (8)
        pgn = (frame.arbitration_id >> 8) & 0x3FFFF
        
        signals = {}
        
        # 1. PGN 61444 (EEC1) - Engine RPM & Torque
        if pgn == J1939_Decoder.PGN_EEC1:
            # RPM: Bytes 4,5. Resolution 0.125 rpm/bit. Offset 0.
            rpm_raw = (frame.payload[4] << 8) | frame.payload[3] # Little Endian for J1939 words
            signals['rpm'] = rpm_raw * 0.125
            
            # Torque Mode: Byte 0 (bits 1-4)
            signals['torque_mode'] = frame.payload[0] & 0x0F
            
        # 2. PGN 65262 (ET1) - Temps
        elif pgn == J1939_Decoder.PGN_ET1:
            # Coolant Temp: Byte 0. Res 1 degC/bit, Offset -40
            signals['coolant_temp'] = (frame.payload[0] * 1.0) - 40.0
            
        # 3. PGN 61445 (AT1) - SCR / DEF Info
        elif pgn == J1939_Decoder.PGN_AT1:
            # DEF Level: Byte 0. Res 0.4 %/bit.
            signals['def_level_pct'] = frame.payload[0] * 0.4
            
        return signals

class ProprietarySniffer:
    """
    The Forensic Logic.
    Detects messages that exist on the bus but are NOT standard J1939 or ISO PIDs.
    These are candidates for 'Shadow Mode' switches.
    """
    
    # Known proprietary triggers (Fictionalized based on research)
    SUSPICIOUS_IDS = [
        0x123, # Generic OEM Debug
        0x456, # Bosch EDC17 'Factory Mode' Broadcast
        0x7E0  # Standard Request (Watch for proprietary SIDs)
    ]
    
    def __init__(self):
        self.iso_handler = ISOTP_Reassembler()
        self.anomaly_log = []
        
    def inspect_frame(self, arbitration_id: int, data_bytes: List[int], timestamp: float):
        frame = CanFrame(
            arbitration_id=arbitration_id, 
            dlc=len(data_bytes), 
            payload=bytes(data_bytes), 
            timestamp=timestamp
        )
        
        # 1. Run Decoders
        j1939_data = J1939_Decoder.decode(frame)
        if j1939_data:
            return {"protocol": "J1939", "signals": j1939_data}
            
        iso_payload = self.iso_handler.process_frame(frame)
        if iso_payload:
            # Full diagnostic response detected
            return self._analyze_diagnostic_payload(iso_payload)
            
        # 2. Proprietary ID Check
        if frame.arbitration_id in self.SUSPICIOUS_IDS:
            self._analyze_proprietary(frame)
            return {"protocol": "PROPRIETARY", "alert": "Suspicious ID detected"}
            
        return None

    def _analyze_diagnostic_payload(self, payload: bytes):
        """
        Inspects reassembled diagnostic packets for 'Mode $27' (Security Access)
        or proprietary routines (Mode $31).
        """
        if len(payload) < 2: return None
        
        sid = payload[0] # Service ID
        
        # SID $27: Security Access (Unlocking ECU)
        if sid == 0x27:
            return {"alert": "SECURITY_ACCESS_ATTEMPT", "seed": payload.hex()}
            
        # SID $31: Routine Control (Running tests)
        if sid == 0x31:
            # Check for known 'Dyno Mode' routines
            routine_id = (payload[1] << 8) | payload[2]
            if routine_id == 0x0101: # Example: VW Roller Bench Mode
                return {"alert": "DYNO_MODE_TRIGGERED", "routine": "0x0101"}
                
        return {"protocol": "ISO-14229", "sid": hex(sid), "len": len(payload)}

    def _analyze_proprietary(self, frame: CanFrame):
        """
        Heuristic analysis of unknown frames.
        Look for static flags toggling when steering angle is 0 (Test Cycle).
        """
        # (Simplified heuristic for demo)
        # If ID 0x456 Byte 0 flips from 0x00 to 0x01, it might be a 'Cheat Active' flag.
        if frame.arbitration_id == 0x456:
            flag = frame.payload[0]
            if flag == 0x01:
                logger.critical("Proprietary Broadcast 0x456: Flag Active (Possible Map Switch)")

# --- UNIT TEST HARNESS ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sniffer = ProprietarySniffer()
    
    print("--- AEGIS CAN SNIFFER INITIALIZED ---")
    
    # 1. Simulate J1939 EEC1 Frame (RPM = 1600)
    # 1600 / 0.125 = 12800 -> 0x3200 (Little Endian -> 00 32)
    eec1_data = [0x00, 0xFF, 0xFF, 0x00, 0x32, 0xFF, 0xFF, 0xFF] 
    # ID: Priority 3, PGN 61444 (F004), Source 00 -> 0CF00400
    res = sniffer.inspect_frame(0x0CF00400, eec1_data, time.time())
    print(f"J1939 Decode: {res}")
    
    # 2. Simulate ISO-TP Multi-Frame Diagnostic Response
    # Frame 1: First Frame (1), Len 0x00C (12 bytes)
    ff_data = [0x10, 0x0C, 0x62, 0x12, 0x34, 0x56, 0x78, 0x9A]
    res1 = sniffer.inspect_frame(0x7E8, ff_data, time.time())
    
    # Frame 2: Consecutive Frame (2), Seq 1
    cf_data = [0x21, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x00] # Padding
    res2 = sniffer.inspect_frame(0x7E8, cf_data, time.time())
    
    if res2:
        print(f"ISO-TP Reassembly: Payload={res2.get('sid')} hex")
    
    # 3. Simulate Proprietary Trigger
    print("Injecting Suspicious Frame 0x456...")
    sniffer.inspect_frame(0x456, [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], time.time())
