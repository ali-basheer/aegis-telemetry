"""
MODULE: REFERENCE_MAP_VALIDATOR (MAP SWITCHING DETECTION)
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-06
CLASSIFICATION: FORENSIC / CALIBRATION
"""

import math
import logging
import json
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Configure module-level logger
logger = logging.getLogger("AEGIS.COMPLIANCE.MAPS")

@dataclass
class CalibrationTable:
    """
    Represents a 3D Engine Map (Z = f(X, Y)).
    Example: Injection Quantity = f(RPM, Torque)
    """
    name: str
    x_axis: List[float] # e.g., RPM Breakpoints
    y_axis: List[float] # e.g., Load Breakpoints
    z_values: List[List[float]] # The Grid
    units: str
    hash_signature: str = ""

    def __post_init__(self):
        if len(self.z_values) != len(self.y_axis):
            raise ValueError(f"Map {self.name}: Y-axis len {len(self.y_axis)} != Rows {len(self.z_values)}")
        if len(self.z_values[0]) != len(self.x_axis):
            raise ValueError(f"Map {self.name}: X-axis len {len(self.x_axis)} != Cols {len(self.z_values[0])}")
        self.hash_signature = self._compute_signature()

    def _compute_signature(self) -> str:
        content = f"{self.name}|{self.x_axis}|{self.y_axis}|{self.z_values}"
        return hashlib.sha256(content.encode()).hexdigest()

class MathKernel:
    @staticmethod
    def bilinear_interpolation(x: float, y: float, 
                             x_axis: List[float], y_axis: List[float], 
                             z_grid: List[List[float]]) -> float:
        idx_x = MathKernel._bisect_left(x_axis, x)
        idx_y = MathKernel._bisect_left(y_axis, y)
        
        idx_x = max(0, min(idx_x, len(x_axis) - 2))
        idx_y = max(0, min(idx_y, len(y_axis) - 2))
        
        x1, x2 = x_axis[idx_x], x_axis[idx_x+1]
        y1, y2 = y_axis[idx_y], y_axis[idx_y+1]
        
        q11 = z_grid[idx_y][idx_x]
        q21 = z_grid[idx_y][idx_x+1]
        q12 = z_grid[idx_y+1][idx_x]
        q22 = z_grid[idx_y+1][idx_x+1]
        
        if x2 == x1: return q11
        if y2 == y1: return q11
        
        r1 = ((x2 - x) / (x2 - x1)) * q11 + ((x - x1) / (x2 - x1)) * q21
        r2 = ((x2 - x) / (x2 - x1)) * q12 + ((x - x1) / (x2 - x1)) * q22
        
        p = ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2
        return p

    @staticmethod
    def _bisect_left(a: List[float], x: float) -> int:
        lo, hi = 0, len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1 if lo > 0 else 0

class GoldenMapDatabase:
    def __init__(self):
        self.maps: Dict[str, CalibrationTable] = {}
        self._seed_default_maps()
        
    def _seed_default_maps(self):
        rpm_axis = [800.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 4500.0]
        load_axis = [10.0, 30.0, 50.0, 70.0, 90.0, 100.0]
        
        urea_grid = [
            [0.0, 0.0, 0.5, 1.0, 2.0, 5.0, 5.0],
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0],
            [5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 25.0],
            [10.0, 15.0, 25.0, 35.0, 45.0, 55.0, 60.0],
            [20.0, 30.0, 50.0, 70.0, 90.0, 100.0, 110.0],
            [25.0, 40.0, 60.0, 80.0, 100.0, 120.0, 130.0]
        ]
        
        self.maps['TargetUrea'] = CalibrationTable(
            "TargetUrea", rpm_axis, load_axis, urea_grid, "mg/stroke"
        )

    def get_reference_value(self, map_name: str, x_rpm: float, y_load: float) -> float:
        if map_name not in self.maps:
            return 0.0
        m = self.maps[map_name]
        return MathKernel.bilinear_interpolation(x_rpm, y_load, m.x_axis, m.y_axis, m.z_values)

class ShadowGovernor:
    TOLERANCE_PCT = 0.10
    
    def __init__(self):
        self.db = GoldenMapDatabase()
        self.map_switch_counter = 0
        
    def audit_cycle(self, frame: Dict) -> Dict:
        rpm = frame.get('rpm', 800)
        load = frame.get('load', 0)
        
        strokes_per_sec = (rpm * 4) / 120.0
        ref_urea_mg_strk = self.db.get_reference_value('TargetUrea', rpm, load)
        ref_urea_mg_s = ref_urea_mg_strk * strokes_per_sec
        
        actual_urea_mg_s = frame.get('reductant_rate', 0.0)
        
        urea_deviation = 0.0
        if ref_urea_mg_s > 1.0:
            urea_deviation = (ref_urea_mg_s - actual_urea_mg_s) / ref_urea_mg_s
            
        is_map_switched = False
        if urea_deviation > self.TOLERANCE_PCT:
            self.map_switch_counter += 1
        else:
            self.map_switch_counter = max(0, self.map_switch_counter - 1)
            
        if self.map_switch_counter > 20:
            is_map_switched = True

        return {
            "ref_urea_mg_s": ref_urea_mg_s,
            "actual_urea_mg_s": actual_urea_mg_s,
            "urea_deviation_pct": urea_deviation * 100.0,
            "map_switch_flag": is_map_switched
        }
