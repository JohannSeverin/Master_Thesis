# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os, pickle

sys.path.append("..")

config = json.load(open("../qubit_calibration.json", "r"))

config["g"] *= 2  # Looks like a small error in the coupling strength.
config["eta"] = None
config["fr"] = 7.552e9

save_path = "data/"
name = "dispersive_approx"

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=20)

from devices.system import QubitResonatorSystem
from simulation.experiment import LindbladExperiment, SchroedingerExperiment
from analysis.auto import automatic_analysis
from devices.pulses import SquareCosinePulse

pulse_frequency = config["fr"] * timescale + 0.0025

resonator_pulse = SquareCosinePulse(amplitude=25e-3, frequency=pulse_frequency)

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=None,
)

dispersive_system = system.dispersive_approximation()

times = np.arange(0, 1001, 0.1)

initial_state = [system.get_states(0, 0), system.get_states(1, 0)]

# Experiment
if not os.path.exists(os.path.join(save_path, name + "_full.pkl")):
    experiment = LindbladExperiment(
        system,
        initial_state,
        times,
        expectation_operators=[
            system.resonator_I(),
            system.resonator_Q(),
            system.photon_number_operator(),
        ],
        only_store_final=False,
        store_states=True,
        save_path=os.path.join(save_path, name + "_full.pkl"),
    )

    results = experiment.run()

    # Analysis
    automatic_analysis(results)
else:
    results = pickle.load(open(os.path.join(save_path, name + "_full.pkl"), "rb"))

# # Experiment
if not os.path.exists(os.path.join(save_path, name + "_dipsersive.pkl")):
    experiment = LindbladExperiment(
        dispersive_system,
        initial_state,
        times,
        expectation_operators=[
            system.resonator_I(),
            system.resonator_Q(),
            system.photon_number_operator(),
        ],
        only_store_final=False,
        store_states=True,
        save_path=os.path.join(save_path, name + "_dipsersive.pkl"),
    )

    results = experiment.run()

    # Analysis
    automatic_analysis(results)
else:
    results = pickle.load(open(os.path.join(save_path, name + "_dipsersive.pkl"), "rb"))


# # Plotting
plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")
plt.rcParams["font.size"] = 16

fig, ax = plt.subplots()

from analysis.Q_func import Q_of_rho

interval = 20
resolution = 200

# Q Function at 0, 100, 200 ns
