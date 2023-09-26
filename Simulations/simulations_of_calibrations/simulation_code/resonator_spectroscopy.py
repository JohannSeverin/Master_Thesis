# To do actual experiment
# - Run 100 steps from 7.55 to 7.56 GHz
# - Duration has to be 5 µs or change experiment


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os

sys.path.append("../..")

config = json.load(open("../../qubit_calibration.json", "r"))

config["fr"] = 7.552e9  # Hz - Uncoupled resonator frequency.
config["g"] *= 2  # Looks like a small error in the coupling strength.
# config["alpha"] = -config["alpha"]

save_path = "../data/"

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_pulse, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=10)
pulse = build_pulse(amplitude=2e-3, frequency=np.linspace(7.55, 7.56, 20))

from devices.system import QubitResonatorSystem
from simulation.experiment import (
    SchroedingerExperiment,
    MonteCarloExperiment,
    LindbladExperiment,
)
from analysis.auto import automatic_analysis

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=1,
)

from devices.system import dispersive_shift

dispersive_shift(system)


initial_states = [system.get_states(0), system.get_states(1)]
times = np.linspace(0, 5000, 15000)  # Need to do 5 µs

# Schroedinger Experiment
experiment = SchroedingerExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=True,
    store_states=True,
    save_path=os.path.join(save_path, "resonator_spectroscopy_schroedinger.pkl"),
)

results = experiment.run()

plt.plot(results.exp_vals[:, 0])
plt.plot(results.exp_vals[:, 1])
# plt.plot(results.exp_vals[:, 2])
plt.show()

automatic_analysis(results)

# Monte Carlo Experiment
experiment = MonteCarloExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=True,
    store_states=True,
    ntraj=10,
    save_path=os.path.join(save_path, "resonator_spectroscopy_monte_carlo.pkl"),
)

results = experiment.run()

automatic_analysis(results)

# # Lindblad Experiment
experiment = LindbladExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=True,
    store_states=True,
    save_path=os.path.join(save_path, "resonator_spectroscopy_lindblad.pkl"),
)

results = experiment.run()

automatic_analysis(results)
