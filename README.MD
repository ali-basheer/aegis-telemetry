# A.E.G.I.S. (Automated Emissions Governance & Intelligence System)

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A forensic telemetry suite for detecting defeat devices and emissions fraud in modern diesel engines.**

---

## The Why

**Fact 1:** City dwellers die sooner due to tailpipe emissions. Children and elderly are most affected.

**Fact 2:** Every single automotive manufacturer uses software to bypass emissions standards. (They just haven't been caught *yet*.)

**Fact 3:** Their methods of cheating are harder than ever to detect.

The reasons why these frauds occur are complex; no single person, group, or company can bear the blame alone.

Basic tools like A.E.G.I.S. can help you understand if your vehicle is cheating.

This tool is not about carbon emissions; it's about helping understand our direct contributions air quality related mortality. 

## The Mechanism

AEGIS treats the ECU as a hostile witness. It runs a parallel physics simulation (Shadow Mode) to validate sensor integrity.

### 1. Thermodynamic Validation
The system solves the Zeldovich Mechanism using live data. It calculates the theoretical adiabatic flame temperature to determine if the reported NOx levels are physically possible.

### 2. Chemical Stoichiometry
AEGIS models the SCR (Selective Catalytic Reduction) reaction rates. If the engine load is high but the Urea (AdBlue) dosing rate drops below the stoichiometric requirement, the system flags a map switch.

### 3. Cycle Recognition Forensics
Modern cheat software detects when the car is being tested (e.g., on a rolling road). AEGIS implements duration basedd metrics to detect if the current driving pattern matches known regulatory cycles (FTP-75, NEDC). If a match is found, it alerts that the vehicle may be in a Clean Mode masquerade.

### 4. AI Meta-Classification
A **Random Forest Classifier** acts as the system's adjudicator. It aggregates signals from the physics kernels, compliance monitors, and sensor fusion layers into a feature vector. This model renders a final verdict (`COMPLIANT` or `NON_COMPLIANT`) and provides an Explainable AI (XAI) output, citing specific factors (e.g., "Kinematic Lock" or "Thermodynamic Divergence") as evidence.

---

## Forensic Dashboard
*System status during a simulated "Dyno Detection" event. Note the discrepancy between Engine Load and Urea Dosing.*

```text
=== A.E.G.I.S. FORENSIC MONITOR (LIVE) ===
SESSION: 26f9179e... | TIME: 18:16:39
------------------------------------------------------------
ENGINE STATE   | RPM: 3460 | LOAD:  87% | SPEED:    0 km/h
EMISSIONS      | NOx ACT:  176 ppm | NOx EXP: [HIGH VARIANCE]
UREA DOSING    | ACT:  0.0 mg/s | MAP DEV: +100.0%
------------------------------------------------------------
AI JUDGEMENT   | NON_COMPLIANT (CRITICAL) (Prob: 1.00)
ALERTS         | ['KINEMATIC_LOCK_DETECTED', 'CALIBRATION_MAP_MISMATCH']

---
Deployment & Hardware Guide
Phase 1: Simulation (Hardware-in-the-Loop)
You do not need a truck to test the logic. The repository includes a physics-based emulator (src/hardware/emulator.py) that generates noisy sensor data to validate the compliance algorithms.

Run in Developer Mode:

Bash

# Install dependencies
pip install -r requirements.txt

# Train the local AI model (Random Forest)
python tools/train_dummy_model.py

# Launch the Orchestrator
python src/main.py
Phase 2: Production (Vehicle Integration)
To deploy this on a physical vehicle, you must switch the Hardware Abstraction Layer (HAL) from simulation to live.

1. The Bill of Materials
Compute: Raspberry Pi 4 Model B (or NVIDIA Jetson Nano for faster inference).

CAN Interface:

Option A (USB): CANable 2.0 (candleLight firmware recommended).

Option B (HAT): PiCAN 2 or Waveshare RS485 CAN HAT.

Cabling: DB9 to OBD-II Cable.

2. Wiring the OBD-II Port
You need to tap into the Primary CAN Bus (J1939). On a standard OBD-II connector:

Pin 6: CAN High

Pin 14: CAN Low

Pin 16: +12V (Power for Pi)

Pin 4/5: Ground

3. OS Configuration (Linux/Raspbian)
AEGIS uses socketcan for high-performance packet capture.

Bash

# 1. Bring up the CAN interface (Bitrate 500k is standard for J1939)
sudo ip link set can0 up type can bitrate 500000

# 2. Verify traffic is flowing
candump can0
4. AEGIS Configuration
Edit config/settings.yaml to disable the emulator and bind to the live socket.

YAML

hardware:
  simulation_mode: false
  interface_type: "socketcan"
  channel: "can0"
Once configured, run python src/main.py. The system will begin logging to the cryptographic ledger immediately.

Contribution & Data
This project is open for academic and forensic use.

I am actively seeking datasets from Euro 6 / EPA Tier 4 compliant heavy-duty trucks to refine the reference map baselines. If you have .blf, .asc, or .csv logs from dyno runs, please open an issue or contact me directly.

License: MIT