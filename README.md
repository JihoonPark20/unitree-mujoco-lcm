# Unitree MuJoCo LCM Interface

This repository provides an **LCM-based interface** for `unitree_mujoco`.

The original `unitree_mujoco` repository relies on a **Unitree SDK2–based bridge**, which is primarily supported on Linux.  
This repository implements an alternative communication layer based on **LCM**, allowing the MuJoCo simulation to be used **on macOS as well as other platforms**.

Original repository:  
https://github.com/unitreerobotics/unitree_mujoco

---

## Background

- `unitree_mujoco` uses a communication bridge based on **Unitree SDK2**
- That bridge has platform limitations
- This repository replaces that layer with an **LCM-based interface**
- The interface works on **macOS** and is not tied to Linux-only dependencies

---

## Requirements

- Python 3
- MuJoCo (as required by `unitree_mujoco`)
- LCM (Python package)