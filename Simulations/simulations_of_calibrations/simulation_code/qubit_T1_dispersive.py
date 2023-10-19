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
    StochasticMasterEquationExperiment,
)
from analysis.auto import automatic_analysis

from devices.pulses import SquareCosinePulse

dummy_pulse = SquareCosinePulse(
    amplitude=0,
    frequency=config["fr"] * timescale,
    phase=0,
)


system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=dummy_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=None,
).dispersive_approximation(config["chi"] * timescale)


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
    save_path=os.path.join(save_path, name + "_schoedinger_dispersive.pkl"),
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
    ntraj=10,
    exp_val_method="average",
    store_states=False,
    save_path=os.path.join(save_path, name + "_monte_carlo_dispersive.pkl"),
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
    expectation_operators=[system.qubit_state_occupation_operator()],
    only_store_final=False,
    store_states=False,
    save_path=os.path.join(save_path, name + "_lindblad_dispersive.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Lindblad: {time.time() - start_time}")

automatic_analysis(results)


# # SME Experiment
times = np.linspace(0, 10000, 1000 + 1)
dummy_pulse = SquareCosinePulse(
    amplitude=config["drive_power"] * timescale * 2 * np.pi / 5,
    frequency=config["fr"] * timescale,
    phase=0,
)

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=dummy_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=0.50,
).dispersive_approximation(config["chi"] * timescale)

experiment = StochasticMasterEquationExperiment(
    system,
    initial_states,
    times,
    # expectation_operators=[system.qubit_state_occupation_operator()],
    only_store_final=False,
    store_states=False,
    store_measurements=True,
    save_path=os.path.join(save_path, name + "_sme_dispersive.pkl"),
    ntraj=10,
    nsubsteps=20,
)

start_time = time.time()
results = experiment.run()
print(f"Time for SME: {time.time() - start_time}")


automatic_analysis(results)
