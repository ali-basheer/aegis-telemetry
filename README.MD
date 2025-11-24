
````markdown
# A.E.G.I.S. (Automated Emissions Governance & Intelligence System)

![Status](https://img.shields.io/badge/Status-Production_Ready-red) ![Profile](https://img.shields.io/badge/Profile-Ford_6.7L_Powerstroke-blue) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A forensic telemetry framework for detecting "Defeat Devices" and emissions fraud on the Ford 6.7L Powerstroke (2011-Present).**

> **WARNING:** This tool communicates with vehicle subsystems using manufacturer-specific protocols. Use responsibly.

---

## The Mechanism

AEGIS treats the Engine Control Unit (ECU) as a hostile witness. It runs a parallel physics simulation to validate sensor integrity in real-time.

### 1. Thermodynamic Validation (Zeldovich Solver)
Calculates the theoretical adiabatic flame temperature of the cylinders to determine if reported NOx levels are physically possible.

### 2. Thermal Window Forensics (Bosch Denoxtronic)
Monitors the Selective Catalytic Reduction (SCR) system for illegal shutdowns. It compares the SCR's active status against the known "Safe Operating Area" (Exhaust > 200°C) to detect "Thermal Window" cheats.

### 3. Hardware Topology Scanning
Scans the CAN bus for "Ghost Nodes" (Piggyback Tuners) or missing nodes (Unplugged EGR/DPF modules).

---

## 🛠 Deployment Guide

### 1. Hardware Requirements
* **Vehicle:** Ford F-250/F-350/F-450 (6.7L Diesel)
* **Interface:** USB-to-OBDII Adapter (OBDLink SX or generic ELM327).
* **OS:** Linux (Raspberry Pi) or Windows (Laptop).

### 2. Installation
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Generate the AI Model (CRITICAL STEP)
# This creates the 'brain' that detects fraud patterns
python tools/train_dummy_model.py

# 3. Verify Physics Engine
# Runs a virtual dyno test to ensure math is correct
python tests/test_physics.py
```

### 3. Go Live (Forensic Mode)
Plug the adapter into the truck and the computer.

**Option A: Visual Dashboard (Laptop)**
```bash
python src/main.py
```
*Displays live telemetry, fraud probability, and alerts.*

**Option B: Headless Daemon (Raspberry Pi)**
```bash
python src/core/orchestrator.py
```
*Runs in the background. Logs evidence to `data/logs/forensics.db`.*

---

## Configuration
Edit `config/settings.yaml` to tune the detection thresholds:

* `smoke_limit_lambda`: 1.10 (Lower value = Allow more smoke before flagging)
* `confidence_threshold`: 0.85 (AI certainty required to flag fraud)

## License
MIT
````