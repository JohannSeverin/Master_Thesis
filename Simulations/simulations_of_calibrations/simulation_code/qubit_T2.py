## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys

sys.path.append("..")

config = json.load(open("../qubit_calibration.json", "r"))
timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_pulse, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=10)
pulse = build_pulse(amplitude=0, frequency=0)

from devices.system import QubitResonatorSystem, QubitSystem
from simulation.experiment import (
    SchroedingerExperiment,
    MonteCarloExperiment,
    LindbladExperiment,
)
from analysis.auto import automatic_analysis

system = QubitSystem(
    qubit,
    # resonator,
    # coupling_strength=config["g"] * timescale,
    # readout_efficiency=1,
)


initial_states = [1j * (system.get_states(1) + system.get_states(0)) / np.sqrt(2)]
times = np.linspace(0, 1000, 5000)

import qutip

x_oprerator = qutip.create(3) + qutip.destroy(3)
measure = qutip.tensor(x_oprerator, qutip.qeye(10))

# Schroedinger Experiment
# experiment = SchroedingerExperiment(
#     system,
#     initial_states,
#     times,
#     expectation_operators=[measure_x],
#     only_store_final=False,
#     store_states=True,
# )

# results = experiment.run()


# automatic_analysis(results)

# Monte Carlo Experiment
# experiment = MonteCarloExperiment(
#     system,
#     initial_states,
#     times,
#     expectation_operators=[measure_x],
#     only_store_final=False,
#     ntraj=10,
# )

# results = experiment.run()


# automatic_analysis(results)

# # Schroedinger Experiment
experiment = LindbladExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[
        system.qubit_state_occupation_operator(0),
        system.qubit_state_occupation_operator(1),
        x_oprerator,
    ],
    only_store_final=False,
    store_states=True,
)

results = experiment.run()


automatic_analysis(results)
