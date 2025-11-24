# A.E.G.I.S. SYSTEM CONFIGURATION
# PROFILE: PRODUCTION_READY (FORD 6.7L POWERSTROKE)

# --- SYSTEM IDENTITY ---
system:
  id: "AEGIS_UNIT_01"
  version: "2.0.0-SCORPION"
  mode: "FORENSIC_AUDIT" # Options: MONITOR, AUDIT, DEBUG

# --- HARDWARE ABSTRACTION LAYER (HAL) ---
hardware:
  # Set to 'false' to attempt connection to real vehicle first
  # Set to 'true' to force the Virtual Emulator
  simulation_mode: false
  
  # Interface Type: 'obd' (ELM327/STN) or 'socketcan' (Linux/Native)
  interface_type: "obd"
  
  # Connection parameters
  port: "AUTO" # "AUTO" scans all COM ports. Or specify "/dev/ttyUSB0"
  baudrate: 500000 # 500k is standard for J1939/ISO15765
  timeout: 1.0

# --- VEHICLE PHYSICS PROFILE ---
# Target: 2011-2024 Ford 6.7L Powerstroke V8 Turbo Diesel
vehicle:
  name: "Ford F-Series Super Duty"
  engine_code: "6.7L Scorpion"
  
  geometry:
    cylinders: 8
    bore_mm: 99.06
    stroke_mm: 107.95
    compression_ratio: 16.2
    firing_order: "1-3-7-2-6-5-4-8"
    
  forced_induction:
    type: "VGT" # Variable Geometry Turbo
    max_boost_psi: 32.0
    intercooled: true

# --- COMPLIANCE THRESHOLDS (THE RED LINES) ---
# Sensitivity settings for the forensic detectors
compliance:
  # 1. Stoichiometry (Smoke Limiter)
  smoke_limit_lambda: 1.10
  max_o2_sensor_deviation: 0.15 # 15% mismatch allowed before flagging "Sensor Spoofing"

  # 2. Reference Map (Urea Dosing)
  min_dosing_ratio: 0.015 # 1.5% of Fuel
  max_dosing_ratio: 0.045 # 4.5% of Fuel
  allowable_map_deviation: 0.20 # 20% tolerance

  # 3. Temporal (Thermal Windowing)
  thermal_window:
    min_exhaust_temp_c: 200.0
    min_ambient_temp_c: -7.0
    grace_period_sec: 60 # Time allowed to turn on SCR after entering safe zone

# --- AI META-CLASSIFIER ---
intelligence:
  model_path: "data/models/emissions_v1.tflite"
  confidence_threshold: 0.85 # Minimum probability to trigger "NON_COMPLIANT" verdict
  
# --- DATA PERSISTENCE ---
database:
  path: "data/logs/forensics.db"
  retention_days: 30
  encryption: false # Set true for chain-of-custody mode