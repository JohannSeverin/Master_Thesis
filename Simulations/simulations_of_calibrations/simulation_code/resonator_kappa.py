# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os
import time

sys.path.append("../..")

config = json.load(open("../../qubit_calibration.json", "r"))

config["fr"] = 7.552e9  # Hz - Uncoupled resonator frequency.
config["g"] *= 2  # Looks like a small error in the coupling strength.
config["eta"] = 1

save_path = "../data/"
name = "resonator_kappa"

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=20)

from devices.system import QubitResonatorSystem
from simulation.experiment import (
    MonteCarloExperiment,
    LindbladExperiment,
    StochasticMasterEquationExperiment,
)

from analysis.auto import automatic_analysis

from devices.pulses import SquareCosinePulse

resonator_pulse = SquareCosinePulse(amplitude=5e-3, frequency=7.5545, duration=400)

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=config["eta"],
)


initial_states = [system.get_states(0)]
times = np.linspace(0, 1500, 15000)


# Monte Carlo Experiment
experiment = MonteCarloExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=False,
    ntraj=10,
    exp_val_method="average",
    store_states=True,
    save_path=os.path.join(save_path, name + "_monte_carlo.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Monte Carlo: {time.time() - start_time}")


automatic_analysis(results)

# # Lindblad Experiment
experiment = LindbladExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=False,
    store_states=True,
    save_path=os.path.join(save_path, name + "_lindblad.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Lindblad: {time.time() - start_time}")


automatic_analysis(results)


# Stochastic Master Equation Experiment
experiment = StochasticMasterEquationExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=False,
    store_states=True,
    ntraj=1,
    nsubsteps=50,
    save_path=os.path.join(save_path, name + "_sme.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for SME: {time.time() - start_time}")

automatic_analysis(results)
