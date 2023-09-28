# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os
import time

sys.path.append("..")

config = json.load(open("../qubit_calibration.json", "r"))

config["g"] *= 2  # Looks like a small error in the coupling strength.
config["eta"] = 1
# config["kappa"] *= 10

save_path = "data/"
name = "qfunc_trajectories"

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

resonator_pulse = SquareCosinePulse(amplitude=12e-3, frequency=config["fr"] * timescale)

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=None,
).dispersive_approximation(config["chi"] * timescale)

from qutip import ket2dm

initial_temperature_prob = 0.125
initial_ground = (1 - initial_temperature_prob) * ket2dm(
    system.get_states(0)
) + initial_temperature_prob * ket2dm(system.get_states(1))
initial_excited = (1 - initial_temperature_prob) * ket2dm(
    system.get_states(1)
) + initial_temperature_prob * ket2dm(system.get_states(0))

initial_states = [initial_ground, initial_excited]
times = np.arange(0, 1000, 10)

# # Lindblad Experiment
experiment = LindbladExperiment(
    system,
    initial_states,
    times,
    expectation_operators=[
        system.resonator_I(),
        system.resonator_Q(),
        system.photon_number_operator(),
    ],
    only_store_final=False,
    store_states=True,
    save_path=os.path.join(save_path, name + "_lindblad.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for Lindblad: {time.time() - start_time}")

from analysis.Q_func import qfunc_plotter_with_time_slider

qfunc_plotter_with_time_slider(results, interval=10, resolution=100, time_steps=1)

automatic_analysis(results)

import gc

gc.collect()


# Stochastic Master Equation Experiment with perfect efficiency

times = np.arange(0, 1000, 10, dtype=np.float64)
system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=1,
).dispersive_approximation(config["chi"] * timescale)


experiment = StochasticMasterEquationExperiment(
    system,
    initial_states,
    times,
    # expectation_operators=[
    #     system.resonator_I(),
    #     system.resonator_Q(),
    #     system.photon_number_operator(),
    # ],
    only_store_final=False,
    store_states=False,
    ntraj=250,
    nsubsteps=5,
    save_path=os.path.join(save_path, name + "_sme_perfect.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for SME: {time.time() - start_time}")

import gc

gc.collect()


# Stochastic Master Equation Experiment

times = np.arange(0, 1000, 10, dtype=np.float64)
system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=0.10,
).dispersive_approximation(config["chi"] * timescale)


experiment = StochasticMasterEquationExperiment(
    system,
    initial_states,
    times,
    # expectation_operators=[
    #     system.resonator_I(),
    #     system.resonator_Q(),
    #     system.photon_number_operator(),
    # ],
    only_store_final=False,
    store_states=False,
    ntraj=250,
    nsubsteps=5,
    save_path=os.path.join(save_path, name + "_sme.pkl"),
)

start_time = time.time()
results = experiment.run()
print(f"Time for SME: {time.time() - start_time}")

import gc

gc.collect()

# automatic_analysis(results)

measurements = np.squeeze(results.measurements) / np.sqrt(config["kappa"] * timescale)

plt.figure()
plt.plot(
    measurements[0, :, :, 0].mean(axis=1), -measurements[0, :, :, 1].mean(axis=1), "o"
)
plt.plot(
    measurements[1, :, :, 0].mean(axis=1), -measurements[1, :, :, 1].mean(axis=1), "o"
)


plt.figure()
plt.plot(
    results.times[::],
    measurements[0, :, :, 0].mean(axis=0),
)  # .mean(axis = 0))
plt.plot(
    results.times[::],
    measurements[1, :, :, 0].mean(axis=0),
)  # .mean(axis = 0))

plt.plot(results.times[::], measurements[0, 0, :, 0])  # .mean(axis = 0))
plt.plot(results.times[::], measurements[1, 0, :, 0])  # .mean(axis = 0))

plt.figure()
plt.plot(
    measurements.mean(axis=1)[0, :, 0].cumsum(),
    measurements.mean(axis=1)[0, :, 1].cumsum(),
    "o--",
)
plt.plot(
    measurements.mean(axis=1)[1, :, 0].cumsum(),
    measurements.mean(axis=1)[1, :, 1].cumsum(),
    "o--",
)


plt.figure()
for i in range(2):
    for j in range(measurements.shape[1]):
        plt.plot(
            measurements[i, j, :, 0].cumsum()[::],
            measurements[i, j, :, 1].cumsum()[::],
            color="C" + str(i),
            alpha=0.5,
        )
