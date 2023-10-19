# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os
import time

sys.path.append("../..")

config = json.load(open("../../qubit_calibration_2.json", "r"))

# config["fr"] = 7.552e9  # Hz - Uncoupled resonator frequency.
# config["g"] *= 2  # Looks like a small error in the coupling strength.

save_path = "../data/"
name = "qubit_T1"

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_pulse, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=10)
pulse = build_pulse(amplitude=0, frequency=0)

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
    coupling_strength=config["g"] * timescale,
    readout_efficiency=1,
)


initial_states = [system.get_states(1)]
times = np.linspace(0, 10000, 20000)

# Schroedinger Experiment
experiment = SchroedingerExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.qubit_state_occupation_operator()],
    only_store_final=False,
    store_states=False,
    save_path=os.path.join(save_path, name + "_schoedinger.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Schroedinger: {time.time() - start_time:.2f} s")

# automatic_analysis(results)

# Monte Carlo Experiment
experiment = MonteCarloExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.qubit_state_occupation_operator()],
    only_store_final=False,
    ntraj=100,
    exp_val_method="average",
    store_states=False,
    save_path=os.path.join(save_path, name + "_monte_carlo.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Monte Carli: {time.time() - start_time}")


# automatic_analysis(results)

# # Lindblad Experiment
experiment = LindbladExperiment(
    system,
    initial_states,
    times,
    only_store_final=False,
    store_states=False,
    expectation_operators=[system.qubit_state_occupation_operator()],
    save_path=os.path.join(save_path, name + "_lindblad.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Lindblad: {time.time() - start_time}")


automatic_analysis(results)
