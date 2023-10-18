# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os
import time

sys.path.append("../..")

config = json.load(open("../../qubit_calibration_2.json", "r"))

save_path = "../Figs/"
name = "qubit_T1_theroy "

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_pulse, build_resonator

from devices.system import QubitSystem
from simulation.experiment import LindbladExperiment, SchroedingerExperiment
from analysis.auto import automatic_analysis

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=10)
pulse = build_pulse(amplitude=0, frequency=0)


system = QubitSystem(
    qubit,
)


initial_states = [system.get_states(0) / np.sqrt(2) + system.get_states(1) / np.sqrt(2)]
times = np.linspace(0, 2000, 2000)

# Schroedinger Experiment
experiment = LindbladExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[
        system.qubit_state_occupation_operator(0),
        system.qubit_state_occupation_operator(1),
    ],
    only_store_final=False,
    store_states=True,
)

start_time = time.time()
results = experiment.run()
# automatic_analysis(results)


rho_01 = [results.states[0][t][0, 1] for t in range(len(results.states[0]))]
demod = np.exp(-1j * 2 * np.pi * config["f01"] * timescale * times)

plt.plot((demod * np.array(rho_01)).real)

fig, ax = plt.subplots(figsize=(5, 5))


# T2 slow component
plt.figure()
config = json.load(open("../../qubit_calibration_2.json", "r"))
config["Tphi"] = 0
config["T1"] = 0
true_f01 = config["f01"] * timescale
config["f01"] += 1e5
config["alpha"] = None

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=10)
pulse = build_pulse(amplitude=0, frequency=0)

system = QubitSystem(
    qubit,
)


from qutip import ket2dm, Qobj
from qutip import sigmax, sigmay

initial_states = [system.get_states(0) / np.sqrt(2) + system.get_states(1) / np.sqrt(2)]
times = np.linspace(0, 2000, 2000)

# Schroedinger Experiment
experiment = SchroedingerExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[
        sigmax(),
        sigmay(),
        system.qubit_state_occupation_operator(0),
        system.qubit_state_occupation_operator(1),
    ],
    only_store_final=False,
    store_states=True,
)

results = experiment.run()

# automatic_analysis(results)

rho_01 = results.exp_vals[0] + 1j * results.exp_vals[1]

demod = np.exp(1j * 2 * np.pi * true_f01 * times)


plt.figure()
plt.plot((demod * rho_01).real)

# plt.hlines(1 / 2, 0, 1000, linestyle="--", color="black")

# plt.plot(np.array(rho_01).imag)


# plt.plot(np.array(rho_01).imag)
