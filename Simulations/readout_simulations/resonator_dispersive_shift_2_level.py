## Setup
import json, sys, os, pickle, gc
import time
import numpy as np
from scipy.constants import hbar, Boltzmann
from qutip import ket2dm

sys.path.append("..")
from qubit_builder import build_qubit, build_resonator

# from analysis.auto import automatic_analysis
from devices.device import SimpleQubit
from devices.system import QubitResonatorSystem
from devices.pulses import SquareCosinePulse
from simulation.experiment import (
    SchroedingerExperiment,
)


config = json.load(open("../qubit_calibration_2.json", "r"))
# config["fr"] = 7.552e9

timescale = 1e-9  # ns

qubit = SimpleQubit(
    frequency=config["f01"] * timescale,
)

resonator = build_resonator(config, timescale, levels=40)

frequencies = np.linspace(config["fr"] * 1e-9 - 0.01, config["fr"] * 1e-9 + 0.01, 100)
resonator_pulse = SquareCosinePulse(
    amplitude=config["drive_power"] * timescale,
    frequency=frequencies,
)

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
).dispersive_approximation()

times = np.arange(0, 310, 10)

states = [system.get_states(0, 0), system.get_states(1, 0)]

experiment = SchroedingerExperiment(
    system=system,
    times=times,
    states=states,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=True,
)

results = experiment.run()

# plt.rcParams["figure.figsize"] = (6, 6)
import matplotlib.pyplot as plt

plt.style.use("../../code/matplotlib_style/margin_figure.mplstyle")
fig, ax = plt.subplots()

ax.plot(frequencies, results.exp_vals[:, 0], label=r"$|0\rangle$")
ax.plot(frequencies, results.exp_vals[:, 1], label=r"$|1\rangle$")

dispersive_shift = system.calculate_dispersive_shift() / 2 / np.pi
ax.set_xticks(
    [
        config["fr"] * 1e-9 + dispersive_shift[0],
        config["fr"] * 1e-9,
        config["fr"] * 1e-9 + dispersive_shift[1],
    ]
)

ax.set_xticklabels([r"$f_r - \chi$", r"$f_r$", r"$f_r + \chi$"], fontsize=18)
ax.set_ylabel(r"Expectation Photons, $\langle a^\dagger a \rangle$ (a .u.)")
ax.set_yticks([])

ax.tick_params(labelsize=24, axis="both")

ax.set(
    title="Dispersive Shift - 2 Levels",
    xlabel="Drive Frequency",
    xlim=(config["fr"] * 1e-9 - 0.01, config["fr"] * 1e-9 + 0.01),
    ylim=(0, 1.1),
)

ax.legend(loc="upper right", fontsize=14)

fig.tight_layout()
fig.savefig("dispersive_shift_2_level.pdf", bbox_inches="tight")
