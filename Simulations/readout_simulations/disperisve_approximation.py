# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt

import json, sys, os, pickle

sys.path.append("..")

config = json.load(open("../qubit_calibration.json", "r"))

config["g"] *= 2  # Looks like a small error in the coupling strength.
config["eta"] = None
config["fr"] = 7.552e9

save_path = "data/"
name = "dispersive_approx"

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=20)

from devices.system import QubitResonatorSystem
from simulation.experiment import LindbladExperiment, SchroedingerExperiment
from analysis.auto import automatic_analysis
from devices.pulses import SquareCosinePulse

pulse_frequency = config["fr"] * timescale + 0.0025

resonator_pulse = SquareCosinePulse(amplitude=25e-3, frequency=pulse_frequency)

system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=None,
)

dispersive_system = system.dispersive_approximation()

times = np.arange(0, 201, 0.1)

initial_state = [system.get_states(0, 0), system.get_states(1, 0)]

# Experiment
if not os.path.exists(os.path.join(save_path, name + "_full.pkl")):
    experiment = LindbladExperiment(
        system,
        initial_state,
        times,
        expectation_operators=[
            system.resonator_I(),
            system.resonator_Q(),
            system.photon_number_operator(),
        ],
        only_store_final=False,
        store_states=False,
        save_path=os.path.join(save_path, name + "_full.pkl"),
    )

    results_full = experiment.run()

    # Analysis
    automatic_analysis(results_full)
else:
    results_full = pickle.load(open(os.path.join(save_path, name + "_full.pkl"), "rb"))

# # # Experiment
if not os.path.exists(os.path.join(save_path, name + "_dipsersive.pkl")):
    experiment = LindbladExperiment(
        dispersive_system,
        initial_state,
        times,
        expectation_operators=[
            system.resonator_I(),
            system.resonator_Q(),
            system.photon_number_operator(),
        ],
        only_store_final=False,
        store_states=True,
        save_path=os.path.join(save_path, name + "_dipsersive.pkl"),
    )

    results_dispersive = experiment.run()

    # Analysis
    automatic_analysis(results_dispersive)
else:
    results_dispersive = pickle.load(
        open(os.path.join(save_path, name + "_dipsersive.pkl"), "rb")
    )


# # Plotting
plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")
plt.rcParams["font.size"] = 16

from matplotlib.colors import LinearSegmentedColormap

cmap_0 = LinearSegmentedColormap.from_list("cmap_0", ["white", "C0"])
cmap_1 = LinearSegmentedColormap.from_list("cmap_1", ["white", "C1"])

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)

from analysis.Q_func import Q_of_rho

interval = 5
resolution = 200

xs, ys = np.linspace(-interval, interval, resolution), np.linspace(
    -interval, interval, resolution
)


# DO A PTRACE
def get_Q_func(rhos, interval, resolution, results, rotations=0):
    qubit_dims, resonator_dims = (
        results.dimensions["qubit"],
        results.dimensions["resonator"],
    )

    reshaped = np.reshape(
        rhos, (-1, qubit_dims, resonator_dims, qubit_dims, resonator_dims)
    )
    ptraced = np.einsum("ijklm -> ikm", reshaped)

    x = np.linspace(-interval, interval, resolution)
    y = np.linspace(-interval, interval, resolution)

    X, Y = np.meshgrid(x, y)

    Qs = Q_of_rho(ptraced, X.flatten(), Y.flatten(), rotate=rotations).reshape(
        -1, resolution, resolution
    )

    return Qs
    # Q_ground = Q_of_rho(rhos_ground, X.flatten(), Y.flatten())


# Q Function at 0, 100, 200 ns
snapshots = [500, 2500, 5000]
states_ground = results_dispersive.states[0][snapshots]
states_excited = results_dispersive.states[1][snapshots]


Qs_ground = get_Q_func(states_ground, interval, resolution, results_dispersive)
Qs_excited = get_Q_func(states_excited, interval, resolution, results_dispersive)


for i, time in enumerate(snapshots):
    ax[0].contour(xs, ys, Qs_ground[i], cmap=cmap_0, levels=10, alpha=0.75)
    ax[0].contour(xs, ys, Qs_excited[i], cmap=cmap_1, levels=10, alpha=0.75)

states_ground = results_full.states[0][snapshots]
states_excited = results_full.states[1][snapshots]

rotations = (config["fr"] * timescale + 0.0025) * results_full.times[snapshots] - 1 / 4

Qs_ground = []
for state, rotation in zip(states_ground, rotations):
    state = np.expand_dims(state, axis=0)
    Qs_ground.append(
        get_Q_func(state, interval, resolution, results_full, rotations=rotation)[0]
    )
Qs_ground = np.array(Qs_ground)

Qs_excited = []
for state, rotation in zip(states_excited, rotations):
    state = np.expand_dims(state, axis=0)
    Qs_excited.append(
        get_Q_func(state, interval, resolution, results_full, rotations=rotation)[0]
    )
Qs_excited = np.array(Qs_excited)


for i, time in enumerate(snapshots):
    ax[1].contour(xs, ys, Qs_ground[i], cmap=cmap_0, levels=10, alpha=0.75)
    ax[1].contour(xs, ys, Qs_excited[i], cmap=cmap_1, levels=10, alpha=0.75)


ax[0].plot(
    results_dispersive.exp_vals[0, 0, :] / 2,
    results_dispersive.exp_vals[0, 1, :] / 2,
    "C0",
)
ax[0].plot(
    results_dispersive.exp_vals[1, 0, :] / 2,
    results_dispersive.exp_vals[1, 1, :] / 2,
    "C1",
)

# Q Function at 0, 100, 200 ns
snapshots = [1000, 3000, 5000]

times = results_full.times


demodulate = np.exp(
    1j * 2 * np.pi * times * (config["fr"] * timescale + 0.0025) - 1j * np.pi / 2
)
IQ_expect_ground = results_full.exp_vals[0, 0, :] + 1j * results_full.exp_vals[0, 1, :]
I_ground = (IQ_expect_ground * demodulate).real / 2
Q_ground = (IQ_expect_ground * demodulate).imag / 2

IQ_expect_excited = results_full.exp_vals[1, 0, :] + 1j * results_full.exp_vals[1, 1, :]
I_excited = (IQ_expect_excited * demodulate).real / 2
Q_excited = (IQ_expect_excited * demodulate).imag / 2

ax[1].plot(I_ground, Q_ground, "C0")
ax[1].plot(I_excited, Q_excited, "C1")

ax[0].set_aspect("equal")
ax[1].set_aspect("equal")

ax[0].set(
    title="Dispersive Approximation",
    xlim=(-interval, interval),
    ylim=(-interval, interval),
    xlabel="I (photons)",
    ylabel="Q (photons)",
)

ax[1].set(
    title="Full Simulation",
    xlim=(-interval, interval),
    ylim=(-interval, interval),
    xlabel="I (photons)",
    # ylabel="Q (photons)",
)


fig.savefig("figures/dispersive_approx.pdf")
