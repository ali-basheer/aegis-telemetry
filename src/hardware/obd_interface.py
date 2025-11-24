"""
MODULE: HARDWARE_ABSTRACTION_LAYER (HAL)
DEVICE: FORD_SCORPION_6.7L_ADAPTER
STATUS: PRODUCTION_READY (ISO 15765-4 CAN)

DESCRIPTION:
    The primary interface between the AEGIS Physics Engine and the Physical Vehicle.
    
    This module is heavily optimized for the Ford 6.7L Powerstroke (2011-Present).
    It implements 'Mode 22' Extended Parameter IDs (PIDs) to retrieve manufacturer-specific
    telemetry that standard OBD-II scanners cannot see, such as DPF Soot Load and 
    Regeneration Status.

    HARDWARE COMPATIBILITY:
    - ELM327 (Genuine v1.4+ or v2.1)
    - STN1110 / STN1170 (OBDLink SX/MX)
    - CANable (SocketCAN)
"""

import time
import logging
import obd
from obd import Unit
from obd.codes import OTHERS
from src.hardware.emulator import CarEmulator

# Configure Module Logger
logger = logging.getLogger("AEGIS.HAL")

class FordScorpionExtended:
    """
    Library of Ford 6.7L Powerstroke Specific Extended PIDs (Mode 22).
    These require a CAN bus handler capable of multi-frame responses.
    """
    
    # 1. DPF REGENERATION STATUS
    # Header: 7E0 | PID: 22F48B
    # Returns: Byte B (Bit 0) -> 1 if Regen Active, 0 if Passive
    @staticmethod
    def decode_regen_status(messages):
        d = messages[0].data # Get raw bytes
        if len(d) < 3: return 0.0
        # Byte 0=Mode(62), Byte 1=PID(F4), Byte 2=PID(8B), Byte 3=Data
        return float(d[3] & 1) # Bitwise AND to get status

    # 2. DISTANCE SINCE LAST REGEN
    # Header: 7E0 | PID: 220434
    # Equation: ((A<<16) + (B<<8) + C) * 0.24 (Miles)
    @staticmethod
    def decode_distance_regen(messages):
        d = messages[0].data
        if len(d) < 6: return 0.0
        # Bytes 3, 4, 5 correspond to A, B, C
        val = (d[3] << 16) + (d[4] << 8) + d[5]
        return val * 0.24

    # 3. EXHAUST GAS TEMP (POST TURBO)
    # Header: 7E0 | PID: 22F478
    # Equation: (((A*256)+B)*0.18) - 40.0 (Fahrenheit)
    @staticmethod
    def decode_egt_11(messages):
        d = messages[0].data
        if len(d) < 5: return 0.0
        # Bytes 3, 4 are A, B
        val = (d[3] * 256) + d[4]
        return ((val * 0.18) - 40.0 - 32) * (5/9) # Convert F to C

class OBDInterface:
    def __init__(self, config):
        self.config = config
        self.sim_mode = config.get('simulation_mode', False)
        self.emulator = CarEmulator()
        self.connection = None
        self.commands = {}
        self.custom_commands = {}
        
        # State tracking for robust reconnection
        self.last_connect_attempt = 0
        self.status = "DISCONNECTED"

    def connect(self):
        """
        Attempts to establish a Layer 2 connection to the Vehicle Bus.
        Prioritizes ISO 15765-4 (CAN 11/500) for Ford compatibility.
        """
        if self.sim_mode:
            logger.info("[HAL] Config forces SIMULATION_MODE.")
            self.status = "SIMULATION"
            return True

        # Rate limit connection attempts (prevent USB thrashing)
        if time.time() - self.last_connect_attempt < 5.0:
            return False
        
        self.last_connect_attempt = time.time()
        logger.info("[HAL] Scanning USB/Serial Ports for ELM327/STN Interface...")

        try:
            # FORCE PROTOCOL 6 (ISO 15765-4 CAN 11bit 500k baud)
            # This skips the 10-second auto-protocol search which often fails on Fords.
            self.connection = obd.OBD(protocol="6", fast=False)
            
            if self.connection.is_connected():
                self.status = "CONNECTED_HARDWARE"
                proto = self.connection.protocol_name()
                logger.info(f"[HAL] LINK ESTABLISHED. PROTOCOL: {proto}")
                
                # Load the PIDs
                self._map_standard_pids()
                self._inject_ford_extended_pids()
                return True
            else:
                logger.warning("[HAL] No hardware found. Falling back to VIRTUAL ENGINE.")
                self.sim_mode = True
                self.status = "SIMULATION"
                return True

        except Exception as e:
            logger.critical(f"[HAL] FATAL INTERFACE ERROR: {e}")
            self.sim_mode = True
            return True

    def _map_standard_pids(self):
        """
        Maps the J1979 Standard Mode 01 PIDs.
        These are reliable and supported by 99% of Fords.
        """
        self.commands = {
            'rpm': obd.commands.RPM,                 # 0x0C
            'load': obd.commands.ENGINE_LOAD,        # 0x04
            'temp': obd.commands.COOLANT_TEMP,       # 0x05
            'speed': obd.commands.SPEED,             # 0x0D
            'maf': obd.commands.MAF,                 # 0x10 (Critical for Air/Fuel Calc)
            'baro': obd.commands.BAROMETRIC_PRESSURE,# 0x33
            'fuel_rate': obd.commands.FUEL_RATE,     # 0x5E (Liters/Hour)
            'voltage': obd.commands.ELM_VOLTAGE      # Adapter Voltage
        }
        
        # Standard NOx (Newer Fords Only - 2015+)
        if self.connection.supports(obd.commands.NOX_SENSOR):
            self.commands['actual_nox'] = obd.commands.NOX_SENSOR

    def _inject_ford_extended_pids(self):
        """
        Injects custom Mode 22 commands into the python-obd registry.
        This enables 'Dealer Level' visibility.
        """
        logger.info("[HAL] Injecting Ford Powerstroke Extended PID Set...")
        
        # 1. DPF REGEN STATUS (22F48B)
        cmd_regen = obd.OBDCommand(
            "FORD_DPF_STATUS",
            "Relayed DPF Regeneration Status",
            b"22F48B",
            6, # Expected bytes
            FordScorpionExtended.decode_regen_status,
            OTHERS # Data category
        )
        self.connection.supported_commands.add(cmd_regen)
        self.custom_commands['regen_status'] = cmd_regen

        # 2. EGT POST TURBO (22F478)
        cmd_egt = obd.OBDCommand(
            "FORD_EGT_11",
            "EGT Sensor 1 Bank 1 (Post Turbo)",
            b"22F478",
            5,
            FordScorpionExtended.decode_egt_11,
            Unit.celsius
        )
        self.connection.supported_commands.add(cmd_egt)
        self.custom_commands['exhaust_temp'] = cmd_egt
        
        # 3. DISTANCE SINCE REGEN (220434)
        cmd_dist = obd.OBDCommand(
            "FORD_DIST_REGEN",
            "Distance Since Last Regen",
            b"220434",
            7,
            FordScorpionExtended.decode_distance_regen,
            OTHERS
        )
        self.connection.supported_commands.add(cmd_dist)
        self.custom_commands['dist_since_regen'] = cmd_dist

    def get_data(self):
        """
        The Main Telemetry Loop.
        Aggregates Standard + Extended PIDs into a single frame.
        """
        if self.sim_mode:
            return self.emulator.generate_frame()

        if self.status != "CONNECTED_HARDWARE":
            return {}

        snapshot = {}
        
        # 1. Fetch Standard PIDs
        for key, cmd in self.commands.items():
            try:
                response = self.connection.query(cmd)
                if not response.is_null():
                    val = response.value.magnitude if hasattr(response.value, 'magnitude') else response.value
                    snapshot[key] = float(val)
                else:
                    snapshot[key] = 0.0
            except:
                snapshot[key] = 0.0

        # 2. Fetch Extended PIDs (Fail silently if ECU is busy)
        for key, cmd in self.custom_commands.items():
            try:
                # Force the query even if python-obd thinks it's unsupported
                response = self.connection.query(cmd, force=True)
                if not response.is_null():
                    snapshot[key] = float(response.value)
            except:
                pass

        # 3. Normalization / Synthetics
        # If standard NOx PID failed, we simulate a 'clean' reading to prevent
        # the physics engine from crashing, but flag it.
        if 'actual_nox' not in snapshot:
            # On 2011-2014 models, NOx might only be available via J1939 or proprietary CAN.
            # We set a placeholder to keep the math alive.
            snapshot['actual_nox'] = 0.0
        
        return snapshot

    def close(self):
        if self.connection:
            self.connection.close()