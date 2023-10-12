## Setup
import json, sys, os, pickle, gc
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, Boltzmann
from qutip import ket2dm

sys.path.append("..")
from qubit_builder import build_qubit, build_resonator
from analysis.auto import automatic_analysis
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
    anharmonicity=config["alpha"] * timescale,
)

resonator = build_resonator(config, timescale, levels=40)

frequencies = np.linspace(config["fr"] * 1e-9 - 0.03, config["fr"] * 1e-9 + 0.01, 100)
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

states = [system.get_states(0, 0), system.get_states(1, 0), system.get_states(2, 0)]

experiment = SchroedingerExperiment(
    system=system,
    times=times,
    states=states,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=True,
)

results = experiment.run()

plt.rcParams["figure.figsize"] = (6, 6)
fig, ax = plt.subplots()

ax.plot(frequencies, results.exp_vals[:, 0], label=r"$|0\rangle$")
ax.plot(frequencies, results.exp_vals[:, 1], label=r"$|1\rangle$")
ax.plot(frequencies, results.exp_vals[:, 2], label=r"$|2\rangle$")

dispersive_shift = system.calculate_dispersive_shift() / 2 / np.pi
ax.set_xticks(
    [
        config["fr"] * 1e-9 + dispersive_shift[0],
        config["fr"] * 1e-9,
        config["fr"] * 1e-9 + dispersive_shift[1],
        config["fr"] * 1e-9 + dispersive_shift[2],
    ]
)

ax.set_xticklabels(
    [r"$f_r + \chi_0$", r"$f_r$", r"$f_r + \chi_1$", "$f_r + \chi_2$"],
    fontsize=18,
    rotation=45,
)
ax.set_ylabel("Expectation Photons (a .u.)")
ax.set_yticks([])
ax.set(
    title="Dispersive Shift",
    xlabel="Drive Frequency",
    xlim=(config["fr"] * 1e-9 - 0.01, config["fr"] * 1e-9 + 0.01),
)

ax.legend()

fig.tight_layout()
fig.savefig("dispersive_shift_3_level.pdf")
