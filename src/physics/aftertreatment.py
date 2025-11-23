"""
MODULE: SCR_KINETIC_SOLVER
AUTHOR: ALI BASHEER (A.E.G.I.S. LEAD)
DATE: 2025-01-03
CLASSIFICATION: FORENSIC / CHEMICAL ENGINEERING
"""

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# Configure module-level logger
logger = logging.getLogger("AEGIS.PHYSICS.SCR")

# --- CHEMICAL CONSTANTS ---
MOLAR_MASS_UREA = 60.06  # g/mol
MOLAR_MASS_NH3 = 17.031  # g/mol
MOLAR_MASS_NO = 30.01    # g/mol
R_GAS = 8.314            # J/(mol*K)

@dataclass
class CatalystGeometry:
    diameter_m: float = 0.3
    length_m: float = 0.3
    cell_density_cpsi: int = 400
    
    @property
    def total_volume(self):
        return math.pi * (self.diameter_m / 2)**2 * self.length_m

class ArrheniusRate:
    def __init__(self, A_factor, Ea_joules):
        self.A = A_factor
        self.Ea = Ea_joules

    def get_k(self, temp_k):
        return self.A * math.exp(-self.Ea / (R_GAS * temp_k))

class UreaDosingUnit:
    def __init__(self):
        self.efficiency_map = {150: 0.20, 200: 0.60, 250: 0.95, 300: 0.99}

    def inject(self, dosing_rate_mg_s, exhaust_temp_c, exhaust_flow_kg_hr):
        eff = 0.0
        temps = sorted(self.efficiency_map.keys())
        for i in range(len(temps)-1):
            if temps[i] <= exhaust_temp_c <= temps[i+1]:
                t_low, t_high = temps[i], temps[i+1]
                slope = (self.efficiency_map[t_high] - self.efficiency_map[t_low]) / (t_high - t_low)
                eff = self.efficiency_map[t_low] + slope * (exhaust_temp_c - t_low)
                break
        if exhaust_temp_c > temps[-1]: eff = 1.0
        
        moles_urea = (dosing_rate_mg_s * 0.325 / 1000.0) / MOLAR_MASS_UREA
        return moles_urea * 2.0 * eff

class SCRReactor:
    def __init__(self, slices=5):
        self.geo = CatalystGeometry()
        self.slices = slices
        self.dosing_unit = UreaDosingUnit()
        self.theta_storage = np.zeros(slices)
        self.max_storage_capacity = 50.0 
        
        self.k_ads = ArrheniusRate(1.5e4, 0.0) 
        self.k_des = ArrheniusRate(4.2e6, 95000.0)
        self.k_std = ArrheniusRate(3.8e5, 54000.0)

    def step_simulation(self, dt_sec: float, inlet_no_ppm: float, inlet_temp_c: float, 
                        mass_flow_kg_hr: float, dosing_mg_s: float) -> dict:
        temp_k = inlet_temp_c + 273.15
        rho_gas = (101325.0) / (287.0 * temp_k)
        vol_flow_m3_s = (mass_flow_kg_hr / 3600.0) / rho_gas
        c_tot = 101325.0 / (R_GAS * temp_k)
        
        c_no_curr = (inlet_no_ppm * 1e-6) * c_tot
        c_nh3_curr = self.dosing_unit.inject(dosing_mg_s, inlet_temp_c, mass_flow_kg_hr) / vol_flow_m3_s
        
        slice_vol = self.geo.total_volume / self.slices
        residence_time = slice_vol / vol_flow_m3_s
        
        for i in range(self.slices):
            theta = self.theta_storage[i]
            k_a = self.k_ads.get_k(temp_k)
            k_d = self.k_des.get_k(temp_k)
            k_s = self.k_std.get_k(temp_k)
            
            r_ads = k_a * c_nh3_curr * (1.0 - theta)
            r_des = k_d * theta
            r_scr = k_s * c_no_curr * theta
            r_net_storage = r_ads - r_des
            
            consumption_no = min(c_no_curr, r_scr * residence_time)
            c_no_next = c_no_curr - consumption_no
            
            transfer_nh3 = min(c_nh3_curr, r_net_storage * residence_time)
            c_nh3_next = c_nh3_curr - transfer_nh3
            
            # --- FIX IS HERE: Changed 'dt' to 'dt_sec' ---
            d_theta = (r_net_storage - r_scr) * dt_sec / self.max_storage_capacity
            self.theta_storage[i] = max(0.0, min(1.0, theta + d_theta))
            
            c_no_curr = c_no_next
            c_nh3_curr = c_nh3_next

        tailpipe_no_ppm = (c_no_curr / c_tot) * 1e6
        ammonia_slip_ppm = (c_nh3_curr / c_tot) * 1e6
        conversion_eff = 1.0 - (tailpipe_no_ppm / inlet_no_ppm) if inlet_no_ppm > 0 else 0.0
            
        return {
            "tailpipe_no_ppm": tailpipe_no_ppm,
            "ammonia_slip_ppm": ammonia_slip_ppm,
            "conversion_efficiency": conversion_eff,
            "avg_ammonia_storage": np.mean(self.theta_storage)
        }

class AftertreatmentForensics:
    def __init__(self):
        self.reactor = SCRReactor(slices=10)
        
    def audit_frame(self, frame: dict, dt: float) -> dict:
        engine_out_nox = frame.get('nox_raw_est', 500.0) 
        
        result = self.reactor.step_simulation(
            dt_sec=dt,
            inlet_no_ppm=engine_out_nox,
            inlet_temp_c=frame.get('exhaust_temp', 100.0), # Fallback if missing
            mass_flow_kg_hr=frame.get('mass_flow', 300.0),
            dosing_mg_s=frame.get('reductant_rate', 0.0)
        )
        
        expected_nox = result['tailpipe_no_ppm']
        actual_nox = frame.get('actual_nox', 0.0)
        
        discrepancy = 0.0
        if actual_nox > 10.0:
            discrepancy = (actual_nox - expected_nox) / actual_nox
            
        flag = "PASS"
        if discrepancy > 0.40:
            flag = "FAIL: DOSING CURTAILMENT SUSPECTED"
            
        return {
            "sim_tailpipe_nox": expected_nox,
            "sim_efficiency": result['conversion_efficiency'],
            "sim_nh3_storage": result['avg_ammonia_storage'],
            "discrepancy": discrepancy,
            "status": flag
        }
