import math

class ZeldovichEngine:
    @staticmethod
    def calculate_limit(temp_c, load_pct):
        temp_k = temp_c + 273.15
        t_combustion = temp_k * 2.8 
        
        if t_combustion < 1800:
            return 50.0 
            
        theoretical_ppm = (math.exp(t_combustion / 300) * 0.05) * (load_pct / 100)
        return min(theoretical_ppm, 3000)
