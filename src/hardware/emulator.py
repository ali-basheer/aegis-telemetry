"""
MODULE: HARDWARE_EMULATOR
PROFILE: FORD_SCORPION_V8 (VIRTUAL)

DESCRIPTION:
    A high-fidelity emulator that mimics the OBD-II data stream of a 
    Ford 6.7L Powerstroke driving through a standard test cycle.
    
    It intentionally injects specific forensic anomalies (e.g. Thermal Windowing)
    to demonstrate the detection capabilities of the AEGIS suite when 
    hardware is unavailable.
"""

import time
import math
import random

class CarEmulator:
    def __init__(self):
        self.start_time = time.time()
        self.speed = 0.0
        self.rpm = 600.0
        self.temp_c = 20.0 # Ambient start
        self.egt_c = 100.0
        self.soot_load = 0.0
        
    def generate_frame(self):
        t = time.time() - self.start_time
        
        # --- SCENARIO: COLD START TO HIGHWAY ---
        # 0-60s: Idle (Warmup)
        # 60-300s: City Driving (Stop & Go)
        # 300s+: Highway Cruise
        
        # 1. PHYSICS SIMULATION
        if t < 60:
            target_speed = 0
            target_rpm = 700 + random.uniform(-10, 10)
            load = 15.0
        elif t < 300:
            # Sine wave traffic
            target_speed = 30 + 20 * math.sin(t / 20.0)
            target_rpm = 1000 + 1000 * (target_speed / 50.0)
            load = 30 + 20 * math.sin(t / 15.0)
        else:
            # Highway
            target_speed = 105 + random.uniform(-2, 2)
            target_rpm = 1800 + random.uniform(-50, 50)
            load = 60 + random.uniform(-5, 5)

        # Lag (Physics Inertia)
        self.speed = self.speed * 0.9 + target_speed * 0.1
        self.rpm = self.rpm * 0.8 + target_rpm * 0.2
        
        # 2. THERMAL MODEL (V8 Diesel)
        # Coolant warms up slowly
        target_temp = 90.0
        self.temp_c += (target_temp - self.temp_c) * 0.001 
        
        # EGT responds fast to load
        target_egt = 150 + (load * 4.5)
        self.egt_c = self.egt_c * 0.9 + target_egt * 0.1
        
        # 3. EMISSIONS & CHEAT LOGIC
        # Base NOx generation (Physics)
        # High Load + High Temp = High NOx
        physics_nox = (load * 5.0) + (self.egt_c * 0.2)
        
        # CHEAT TRIGGER: THERMAL WINDOW
        # If we are cruising (Highway) but EGT is "safe", some cheats turn off dosing
        # to save fluid.
        # Real dosing should be ~150 mg/s at highway load.
        dosing_rate = 0.0
        
        if self.egt_c > 200:
            # We are in the "Safe Zone" -> Should dose!
            # BUT... let's simulate a cheat at t > 320s
            if t > 320:
                dosing_rate = 0.0 # CHEAT ACTIVE (Thermal Window Abuse)
            else:
                dosing_rate = 150.0 # Normal Dosing
        
        # Fake Sensor Readings
        # If dosing is active, NOx is low. If cheat is active, NOx is high.
        if dosing_rate > 0:
            reported_nox = physics_nox * 0.1 # 90% Reduction
        else:
            reported_nox = physics_nox # Raw emissions out the pipe
            
        # 4. FORD SPECIFIC PIDS
        # Generate the extended data we programmed the HAL to find
        return {
            'rpm': self.rpm,
            'speed': self.speed,
            'load': load,
            'temp': self.temp_c,
            'maf': (self.rpm/60) * 3.35 * (load/100) * 1.2, # Approx V8 Airflow
            'fuel_rate': (load * 0.4) + 1.0, # L/h
            'actual_nox': reported_nox,
            'reductant_rate': dosing_rate,
            'exhaust_temp': self.egt_c,
            'regen_status': 0.0,
            'dist_since_regen': 450.0 + (t * 0.01)
        }