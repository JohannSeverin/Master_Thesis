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
from devices.device import SimpleQubit, Transmon
from devices.system import QubitResonatorSystem
from devices.pulses import SquareCosinePulse
from simulation.experiment import (
    SchroedingerExperiment,
)

config = json.load(open("../qubit_calibration_2.json", "r"))
# config["fr"] = 7.552e9

timescale = 1e-9  # ns


qubit_frequencies = [0, config["f01"], config["f01"] * 2 - config["alpha"]]
qubit = Transmon(EC=2, EJ=100, levels=5)

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

states = [system.get_states(i, 0) for i in range(5)]

experiment = SchroedingerExperiment(
    system=system,
    times=times,
    states=states,
    expectation_operators=[system.photon_number_operator()],
    only_store_final=True,
)

results = experiment.run()

# plt.rcParams["figure.figsize"] = (6, 6)
plt.style.use("../../code/matplotlib_style/margin_figure.mplstyle")

fig, ax = plt.subplots()

for i in range(5):
    ax.plot(frequencies, results.exp_vals[:, i], label=f"$|{i}\\rangle$")
# ax.plot(frequencies, results.exp_vals[:, 1], label=r"$|1\rangle$")
# ax.plot(frequencies, results.exp_vals[:, 2], label=r"$|2\rangle$")

dispersive_shift = system.calculate_dispersive_shift() / 2 / np.pi
xticks = config["fr"] * 1e-9 + dispersive_shift
xticks = np.append(xticks, config["fr"])

xticklabels = [f"$f_r + \\chi_{i}$" for i in range(5)]
xticklabels.append("$f_r$")

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=18, rotation=45)

# ax.set_xticks(
#     [
#         config["fr"] * 1e-9 + dispersive_shift[0],
#         config["fr"] * 1e-9,
#         config["fr"] * 1e-9 + dispersive_shift[1],
#         config["fr"] * 1e-9 + dispersive_shift[2],
#     ]
# )

# ax.set_xticklabels(
#     [r"$f_r + \chi_0$", r"$f_r$", r"$f_r + \chi_1$", "$f_r + \chi_2$"],
#     fontsize=18,
#     rotation=45,
# )
ax.set_ylabel("Expectation Photons (a .u.)")
ax.set_yticks([])
ax.set(
    title="Dispersive Shift - 2 Levels",
    xlabel="Drive Frequency",
    xlim=(config["fr"] * 1e-9 - 0.01, config["fr"] * 1e-9 + 0.012),
    ylim=(0, 1.1),
)

ax.legend(loc="upper right", fontsize=14)

# fig.tight_layout()
fig.savefig("dispersive_shift_5_level.pdf")
