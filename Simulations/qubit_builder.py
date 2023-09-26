## Setup
import numpy as np
import matplotlib.pyplot as plt

import json

# sys.path.append("..")

# Paths and imports
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/test"

# Load devices/system
from devices.device import SimpleQubit, Resonator
from devices.system import QubitResonatorSystem, QubitSystem
from devices.pulses import SquareCosinePulse

from devices.system import dispersive_shift
from analysis.auto import automatic_analysis

# config = json.load(open("qubit_calibration.json", "r"))
# timescale = 1e-9  # ns


def build_qubit(config, timescale):
    return SimpleQubit(
        frequency=config["f01"] * timescale,
        anharmonicity=config["alpha"] * timescale,
        T1=config["T1"] / timescale,
        Tphi=config["Tphi"] / timescale,
    )


def build_resonator(config, timescale, levels=20):
    return Resonator(
        frequency=config["fr"] * timescale,
        kappa=config["kappa"] * timescale,
        levels=levels,
    )


def build_pulse(amplitude=0, frequency=0):
    return SquareCosinePulse(
        amplitude=amplitude,
        frequency=frequency,
        phase=0.0,
    )


def build_qubit_system(config, timescale):
    qubit = build_qubit(config, timescale)
    return QubitSystem(qubit)


def build_qubit_resonator_system(
    config, timescale, resonator_levels=10, pulse=False, amplitude=0, frequency=0
):
    qubit = build_qubit(config, timescale)
    resonator = build_resonator(config, timescale, levels=resonator_levels)
    if pulse:
        pulse = build_pulse(amplitude=amplitude, frequency=frequency)
        return QubitResonatorSystem(qubit, resonator, resonator_pulse=pulse)
    else:
        return QubitResonatorSystem(qubit, resonator)
