# To do actual experiment
# - Run for a long time (35 µs)


## Setup
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("../../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["font.size"] = 16

import json, sys, os, pickle
import time

config = json.load(open("../qubit_calibration.json", "r"))

sys.path.append("..")

save_path = "data/"
name = "qfunc_trajectories_test"

timescale = 1e-9  # ns


lindblad_results = pickle.load(
    open(
        "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Simulations/readout_simulations/data/qfunc_trajectories_lindblad.pkl",
        "rb",
    )
)

from analysis.Q_func import Q_of_rho


# At 0, 200 and 400 ns
rhos_ground = lindblad_results.states[0][[0, 20, 40]]
rhos_excited = lindblad_results.states[1][[0, 20, 40]]

interval = 40
resolution = 200

from matplotlib.colors import LinearSegmentedColormap

cmap_0 = LinearSegmentedColormap.from_list("cmap_0", ["white", "C0"])
cmap_1 = LinearSegmentedColormap.from_list("cmap_1", ["white", "C1"])


# DO A PTRACE
def get_Q_func(rhos, interval, resolution, results):
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

    Qs = Q_of_rho(ptraced, X.flatten(), Y.flatten()).reshape(-1, resolution, resolution)

    return Qs
    # Q_ground = Q_of_rho(rhos_ground, X.flatten(), Y.flatten())


Qs_ground = get_Q_func(rhos_ground, interval, resolution, lindblad_results)
Qs_excited = get_Q_func(rhos_excited, interval, resolution, lindblad_results)


x = np.linspace(-interval, interval, resolution)
y = np.linspace(-interval, interval, resolution)


from scipy.stats import multivariate_normal
from scipy.signal import convolve2d

X, Y = np.meshgrid(x, y)
gaussian = multivariate_normal(mean=[0, 0], cov=[[200, 0], [0, 200]]).pdf(
    np.dstack([X, Y])
)

sme_results = pickle.load(
    open(
        "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Simulations/readout_simulations/data/qfunc_trajectories_sme.pkl",
        "rb",
    )
)

measurements_ground = np.squeeze(sme_results.measurements[0]) / np.sqrt(
    config["kappa"] * timescale * 0.10 * 2
)
measurements_excited = np.squeeze(sme_results.measurements[1]) / np.sqrt(
    config["kappa"] * timescale * 0.10 * 2
)

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["figure.figsize"] = (18, 10)
fig, ax = plt.subplots(ncols=3, nrows=2, sharey="row")

for i in range(Qs_ground.shape[0]):
    ax[0, i].contour(x, y, Qs_ground[i], cmap=cmap_0)
    ax[0, i].contour(x, y, Qs_excited[i], cmap=cmap_1)

    ax[1, i].contour(x, y, convolve2d(Qs_ground[i], gaussian, mode="same"), cmap=cmap_0)
    ax[1, i].contour(
        x, y, convolve2d(Qs_excited[i], gaussian, mode="same"), cmap=cmap_1
    )


for i, time in enumerate([0, 20, 40]):
    ax[0, i].set(
        xlim=(-5, 5),
        ylim=(-5, 5),
        title=f"QFunc - {time * 10} ns",
    )

    ax[1, i].scatter(
        measurements_ground[time, :, 0],
        measurements_ground[time, :, 1],
        color="C0",
        alpha=0.5,
    )

    ax[1, i].scatter(
        measurements_excited[time, :, 0],
        measurements_excited[time, :, 1],
        color="C1",
        alpha=0.5,
    )

    ax[1, i].set(
        xlabel="I (photons)",
        title=f"Convolved QFunc and 10 ns Record",
    )

for i in range(2):
    ax[i, 0].set(ylabel="Q (photons)")


fig.suptitle(f"Readout Q Functions and 10 ns Records - $\eta = 10\%$")
fig.tight_layout()

fig.savefig("figures/qfunc_trajectories_perfect.pdf")
