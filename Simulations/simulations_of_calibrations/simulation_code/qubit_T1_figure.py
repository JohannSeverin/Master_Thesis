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

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=10)
pulse = build_pulse(amplitude=0, frequency=0)

from devices.system import QubitSystem
from simulation.experiment import LindbladExperiment
from analysis.auto import automatic_analysis

system = QubitSystem(
    qubit,
)


initial_states = [system.get_states(0), system.get_states(1)]
times = np.linspace(0, 10000, 40000)

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
    store_states=False,
)

start_time = time.time()
results = experiment.run()
automatic_analysis(results)

# Steady state
steady_state = qubit.gamma_up / qubit.gamma_down

# Plotting
fig, ax = plt.subplots(nrows=2, figsize=(5, 7), sharex=True, sharey=True)

ax[0].plot(times * 1e-3, results.exp_vals[0, 0], label=r"$\rho_{00}$")
ax[0].plot(times * 1e-3, results.exp_vals[0, 1], label=r"$\rho_{11}$")

ax[0].set(
    title="Initialized in $|0\\rangle$",
    ylabel="Occupation probability",
    ylim=(-0.10, 1.10),
)

ax[0].legend(fontsize=14, loc="upper right")
ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])

ax[1].plot(times * 1e-3, results.exp_vals[1, 0], label="Ground state occupation")
ax[1].plot(times * 1e-3, results.exp_vals[1, 1], label="Excited state occupation")

ax[1].set(
    title="Initialized in $|1\\rangle$",
    ylabel="Occupation probability",
)

ax[1].set_xlabel("Time (µs)")


ax[0].hlines(1 - steady_state, 0, 10, linestyle="--", color="C0", alpha=0.5)
ax[0].text(
    5.0,
    1 - steady_state - 0.05,
    r"$\rho_{00}(\infty)$",
    color="C0",
    fontsize=14,
    va="top",
)
ax[0].hlines(steady_state, 0, 10, linestyle="--", color="C1", alpha=0.5)
ax[0].text(
    5.0,
    steady_state + 0.05,
    r"$\rho_{11}(\infty)$",
    color="C1",
    fontsize=14,
    va="bottom",
)

ax[1].hlines(1 - steady_state, 0, 10, linestyle="--", color="C0", alpha=0.5)
ax[1].text(
    5.0,
    1 - steady_state + 0.05,
    r"$\rho_{00}(\infty)$",
    color="C0",
    fontsize=14,
    va="bottom",
)
ax[1].hlines(steady_state, 0, 10, linestyle="--", color="C1", alpha=0.5)
ax[1].text(
    5.0,
    steady_state - 0.05,
    r"$\rho_{11}(\infty)$",
    color="C1",
    fontsize=14,
    va="top",
)

fig.tight_layout()

fig.savefig("../Figs/qubit_T1_theroy.pdf")
