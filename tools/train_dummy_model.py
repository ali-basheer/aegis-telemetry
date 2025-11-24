"""
TOOL: AI_MODEL_TRAINER
PROFILE: FORD_SCORPION_6.7L_DIESEL
OUTPUT: TFLite Forensic Classifier

DESCRIPTION:
    Generates synthetic telemetry data representing a Ford 6.7L Powerstroke engine
    under both 'COMPLIANT' and 'NON_COMPLIANT' (Cheating) conditions.

    It trains a Neural Network to recognize the specific signature of emissions fraud:
    specifically, the divergence between 'Engine Load' and 'Urea Dosing' that occurs
    when a defeat device is active.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- CONFIGURATION: FORD 6.7L SPECS ---
# The AI needs to know what "Normal" looks like for this specific truck.
RPM_MIN = 600.0   # Idle
RPM_MAX = 3500.0  # Redline
DISPLACEMENT_L = 6.7

def generate_ford_profile(samples=10000):
    print(f'Generating {samples} samples of Ford 6.7L Telemetry...')
    
    # 1. GENERATE OPERATING CONDITIONS
    # Diesel engines spend most time at low RPM but high torque
    rpm = np.random.triangular(RPM_MIN, 1500, RPM_MAX, samples)
    load = np.random.uniform(0, 100, samples) # % Load
    
    # Intake Temp (Intercooled Turbo)
    temp_c = np.random.normal(45, 15, samples)
    temp_c = np.clip(temp_c, 20, 90)

    # 2. CALCULATE PHYSICS-BASED FEATURES
    # MAF (Mass Air Flow) estimation for 6.7L V8
    # Airflow scales linearly with RPM and Load (Turbo boost effect)
    # A 6.7L engine flows massive air compared to a car.
    maf_g_s = (rpm / 60.0) * (DISPLACEMENT_L / 2.0) * (load / 100.0 * 2.5) * 1.2 / 1000.0 * 1000
    # Simplified: (Rev/s) * (Disp/2) * (VolEff * Boost) * Density
    # Result roughly 10 g/s (Idle) to 400 g/s (Full Throttle)
    
    # 3. GENERATE LABELS (THE "TEACHING" STEP)
    # We want the AI to predict "NOx Mass" to check against the sensor.
    
    # Base Physics: NOx raises exponentially with Load & Temp
    true_nox_ppm = 50 + (load * 3.5) + (np.exp(temp_c / 25.0) * 10)
    
    # 4. INJECT "DEFEAT DEVICE" ANOMALIES
    # We simulate a "Cycle Beating" strategy:
    # If the engine is under high load but the "reported" NOx is miraculously low,
    # that is the signature of software manipulation.
    
    is_cheating = np.zeros(samples)
    reported_nox = true_nox_ppm.copy()
    
    for i in range(samples):
        # Randomly inject cheat logic (15% of dataset)
        if np.random.random() < 0.15:
            # CHEAT MODE: Engine working hard, but sensor reports clean air
            if load[i] > 60: 
                reported_nox[i] = reported_nox[i] * 0.15 # Mask emissions by 85%
                is_cheating[i] = 1.0 # Label as FRAUD

    # Feature Vector: [RPM, LOAD, TEMP, MAF, REPORTED_NOX]
    X = np.column_stack((rpm, load, temp_c, maf_g_s, reported_nox))
    
    # Target Vector: [IS_CHEATING] (Binary Classification)
    y = is_cheating
    
    return X, y

def train_and_export():
    # 1. Get Data
    X, y = generate_ford_profile(samples=20000)
    
    # 2. Build Forensics Model
    # A simple Feed-Forward Network is sufficient for this signature detection
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(5,)), # 5 Input Features
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') # Probability of Cheating (0-1)
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    # 3. Train
    print('\n--- STARTING NEURAL NETWORK TRAINING ---')
    print('Objective: Distinguish "Thermal Windowing" from "Active Fraud"')
    model.fit(X, y, epochs=15, batch_size=64, validation_split=0.2)
    
    # 4. Export to TFLite (Embedded Ready)
    print('\n--- EXPORTING TO EDGE FORMAT ---')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Ensure directory exists
    save_path = os.path.join('data', 'models', 'emissions_v1.tflite')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f'SUCCESS: Ford 6.7L Forensic Model saved to: {save_path}')
    print('Ready for deployment in src/main.py')

if __name__ == '__main__':
    try:
        train_and_export()
    except ImportError:
        print("CRITICAL: TensorFlow not found.")
        print("Please run: pip install tensorflow")