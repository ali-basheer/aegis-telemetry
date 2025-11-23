import time
import logging
from src.ai.inference_engine import EmissionsModel
from src.physics.thermodynamics import ZeldovichEngine
from src.hardware.obd_interface import OBDInterface
from src.core.database import DatabaseHandler

class AegisSystem:
    def __init__(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AEGIS.CORE')
        self.config = config
        
        self.ai = EmissionsModel(config['ai']['model_path'])
        self.hal = OBDInterface(config['hardware'])
        self.db = DatabaseHandler(config['database']['path'])
        
    def run(self):
        self.logger.info('Initializing AEGIS Telemetry...')
        if not self.hal.connect():
            self.logger.critical('Failed to connect to OBD Interface.')
            return

        self.logger.info('System Online. Beginning Shadow Mode Audit.')
        
        try:
            while True:
                frame = self.hal.get_data()
                
                ai_pred = self.ai.predict(
                    frame['rpm'], frame['load'], frame['temp'], frame['maf']
                )
                
                phys_limit = ZeldovichEngine.calculate_limit(frame['temp'], frame['load'])
                
                deviation = 0
                if ai_pred > 10:
                    deviation = abs(frame['actual_nox'] - ai_pred) / ai_pred
                
                is_anomaly = deviation > self.config['thresholds']['nox_deviation_limit']
                
                status = ' [OK] '
                if is_anomaly:
                    status = ' [!! DEFECT DETECTED !!] '
                elif frame.get('status_flag') == 'SIMULATED_DEFEAT_ACTIVE':
                    status = ' [!! SIMULATED DEFEAT !!] '
                
                print(f'RPM: {frame['rpm']:.0f} | Load: {frame['load']:.1f}% | Rpt NOx: {frame['actual_nox']:.1f} | AI Exp: {ai_pred:.1f} | Phys Max: {phys_limit:.1f} || {status}')
                
                self.db.log(frame, ai_pred, phys_limit, is_anomaly)
                
                time.sleep(1.0 / self.config['system']['sampling_rate_hz'])
                
        except KeyboardInterrupt:
            self.logger.info('Shutdown signal received.')
