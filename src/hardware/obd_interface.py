import obd
from src.hardware.emulator import CarEmulator

class OBDInterface:
    def __init__(self, config):
        self.sim_mode = config.get('simulation_mode', True)
        self.emulator = CarEmulator() if self.sim_mode else None
        self.connection = None
    
    def connect(self):
        if self.sim_mode:
            print('[HAL] Simulation Mode Active. Virtual Engine Started.')
            return True
        try:
            self.connection = obd.OBD()
            return self.connection.is_connected()
        except Exception as e:
            print(f'[HAL] Connection Failed: {e}')
            return False

    def get_data(self):
        if self.sim_mode:
            return self.emulator.generate_frame()
        return None
