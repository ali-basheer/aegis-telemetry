import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

def generate_data(samples=5000):
    print('Generating synthetic engine data...')
    rpm = np.random.uniform(800, 5000, samples)
    load = np.random.uniform(10, 100, samples)
    temp = np.random.uniform(20, 90, samples)
    maf = (rpm / 6000) * (load / 100) * 150
    
    # Physics-ish formula for 'True' NOx to train against
    nox = (load * 2) + (temp * 1.5) + (rpm * 0.05) + np.random.normal(0, 5, samples)
    
    X = np.column_stack((rpm, load, temp, maf))
    y = nox
    return X, y

def train_and_export():
    X, y = generate_data()
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print('Training model...')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    
    print('Converting to TFLite...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    save_path = 'data/models/emissions_v1.tflite'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f'SUCCESS: Model saved to {save_path}')

if __name__ == '__main__':
    train_and_export()
