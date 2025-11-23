import numpy as np
import logging
import os
import tensorflow as tf 

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class EmissionsModel:
    def __init__(self, model_path='data/models/emissions_v1.tflite'):
        self.logger = logging.getLogger('AEGIS.AI')
        self.model_path = model_path
        self.interpreter = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.logger.warning(f'Model missing at {self.model_path}. Run tools/train_dummy_model.py')
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.logger.info('TFLite Model loaded successfully.')
        except Exception as e:
            self.logger.error(f'Failed to load AI model: {e}')

    def predict(self, rpm, load, temp, maf):
        if not self.interpreter:
            return 0.0
        input_data = np.array([[rpm, load, temp, maf]], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return float(output_data[0][0])
