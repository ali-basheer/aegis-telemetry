import time
import math
import random

class CarEmulator:
    def __init__(self):
        self.start_time = time.time()
        
    def generate_frame(self):
        elapsed = time.time() - self.start_time
        rpm = 800 + (math.sin(elapsed * 0.2) + 1) * 1350
        load = 20 + (math.sin(elapsed * 0.2) + 1) * 35 + random.uniform(-2, 2)
        temp = 30 + math.log(elapsed + 1) * 5
        maf = (rpm / 100) * (load / 50) + random.uniform(0, 5)
        
        # Base NOx calculation
        actual_nox = (load * 1.8) + (temp * 0.5) + random.uniform(-10, 10)
        
        # Inject 'Defeat Device' anomaly every 30s
        status_flag = 'NORMAL'
        if 25 < (elapsed % 60) < 35:
            actual_nox = actual_nox * 0.1 
            status_flag = 'SIMULATED_DEFEAT_ACTIVE'

        return {
            'rpm': rpm,
            'load': load,
            'temp': temp,
            'maf': maf,
            'actual_nox': max(0, actual_nox),
            'status_flag': status_flag
        }
