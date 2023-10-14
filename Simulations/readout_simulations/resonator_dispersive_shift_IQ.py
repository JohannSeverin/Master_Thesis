## Setup
import json, sys, os, pickle, gc
import time
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")
from scipy.constants import hbar, Boltzmann
from qutip import ket2dm

sys.path.append("..")
config = json.load(open("../qubit_calibration_2.json", "r"))

timescale = 1e-9  # ns
start, end = 0, 200

omega_r = config["fr"] * 2 * np.pi * timescale
omega_d = config["fr"] * 2 * np.pi * timescale
chi = config["chi"] * 2 * np.pi * timescale
drive_amplitude = config["drive_power"] * 2 * np.pi * timescale / 5


def func(t, alpha, state, omega_r):
    state_depdenent_shift = 2 * (state - 0.5) * chi
    return (
        -1j * ((omega_r - omega_d + state_depdenent_shift) * alpha)
        + 1j * drive_amplitude
    )


from scipy.integrate import solve_ivp

times = np.linspace(start, end, 100)
results_ground = solve_ivp(
    func, (start, end), y0=[0 + 0j], args=(0, omega_r), t_eval=times
)
results_excited = solve_ivp(
    func, (start, end), y0=[0 + 0j], args=(1, omega_r), t_eval=times
)

fig, ax = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
ax[0].plot(results_ground.y[0].real, results_ground.y[0].imag)
ax[0].plot(results_excited.y[0].real, results_excited.y[0].imag)

from matplotlib.patches import Circle

circ = Circle(
    [results_ground.y[0, -1].real, results_ground.y[0, -1].imag], radius=0.5, color="C0"
)
ax[0].add_patch(circ)

circ = Circle(
    [results_excited.y[0, -1].real, results_excited.y[0, -1].imag],
    radius=0.5,
    color="C1",
)
ax[0].add_patch(circ)

# Distance between circles
# ax[0].plot(
#     [
#         results_ground.y[0, -1].real,
#         results_excited.y[0, -1].real,
#     ],
#     [
#         results_ground.y[0, -1].imag,
#         results_excited.y[0, -1].imag,
#     ],
#     color="C3",
#     ls = "-",
#     linewidth = 5,
# )

# distance = np.sqrt(
#     (results_ground.y[0, -1].real - results_excited.y[0, -1].real) ** 2
#     + (results_ground.y[0, -1].imag - results_excited.y[0, -1].imag) ** 2
# )

# ax[0].text(
#     (results_ground.y[0, -1].real + results_excited.y[0, -1].real) / 2,
#     (results_ground.y[0, -1].imag + results_excited.y[0, -1].imag) / 2,
#     f"{distance:.2f}",
#     color="C3",
#     fontsize=20,
#     horizontalalignment="center",
#     verticalalignment="bottom",
# )


ax[0].set_aspect("equal")

ax[0].set(
    xlabel="I (photon count)",
    ylabel="Q (photon count)",
    title="Drive at $f_r$",
)

results_ground = solve_ivp(
    func, (start, end), y0=[0 + 0j], args=(0, omega_r - chi), t_eval=times
)
results_excited = solve_ivp(
    func, (start, end), y0=[0 + 0j], args=(1, omega_r - chi), t_eval=times
)

ax[1].plot(results_ground.y[0].real, results_ground.y[0].imag)
ax[1].plot(results_excited.y[0].real, results_excited.y[0].imag)

circ = Circle(
    [results_ground.y[0, -1].real, results_ground.y[0, -1].imag], radius=0.5, color="C0"
)
ax[1].add_patch(circ)

circ = Circle(
    [results_excited.y[0, -1].real, results_excited.y[0, -1].imag],
    radius=0.5,
    color="C1",
)
ax[1].add_patch(circ)

ax[1].set_aspect("equal")

ax[1].set(
    title="Drive at $f_r - \chi$",
    xlabel="I (photon count)",
)
# Distance between circles
# ax[1].plot(
#     [
#         results_ground.y[0, -1].real,
#         results_excited.y[0, -1].real,
#     ],
#     [
#         results_ground.y[0, -1].imag,
#         results_excited.y[0, -1].imag,
#     ],
#     color="C3",
#     ls="-",
#     linewidth=5,
# )

# distance = np.sqrt(
#     (results_ground.y[0, -1].real - results_excited.y[0, -1].real) ** 2
#     + (results_ground.y[0, -1].imag - results_excited.y[0, -1].imag) ** 2
# )

# ax[1].text(
#     (results_ground.y[0, -1].real + results_excited.y[0, -1].real) / 2,
#     (results_ground.y[0, -1].imag + results_excited.y[0, -1].imag) / 2,
#     f"{distance:.2f}",
#     color="C3",
#     fontsize=20,
#     horizontalalignment="center",
#     verticalalignment="bottom",
# )


fig.tight_layout()

# from qubit_builder import build_qubit, build_resonator
# from analysis.auto import automatic_analysis
# from devices.device import SimpleQubit
# from devices.system import QubitResonatorSystem
# from devices.pulses import SquareCosinePulse
# from simulation.experiment import (
#     SchroedingerExperiment,
# )


# # config["fr"] = 7.552e9

# timescale = 1e-9  # ns

# qubit = SimpleQubit(
#     frequency=config["f01"] * timescale,
# )

# resonator = build_resonator(config, timescale, levels=40)

# resonator_pulse = SquareCosinePulse(
#     amplitude=config["drive_power"] * timescale,
#     frequency=config["fr"] * timescale,
# )

# system = QubitResonatorSystem(
#     qubit,
#     resonator,
#     resonator_pulse=resonator_pulse,
#     coupling_strength=config["g"] * timescale,
# ).dispersive_approximation(config["chi"] * timescale)

# times = np.arange(0, 310, 10)

# states = [system.get_states(0, 0), system.get_states(1, 0)]

# experiment = SchroedingerExperiment(
#     system=system,
#     times=times,
#     states=states,
#     expectation_operators=[system.photon_number_operator()],
#     only_store_final=False,
#     store_states=True,
# )

# results = experiment.run()


# from analysis.Q_func import Q_of_rho, qfunc_plotter
