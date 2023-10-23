import numpy as np
import matplotlib.pyplot as plt

import json, sys, os, pickle
import time

sys.path.append("..")

config = json.load(open("../qubit_calibration_2.json", "r"))

# config["g"] *= 2  # Looks like a small error in the coupling strength.
# config[
#     "eta"
# ] *= 9  # This is to make up for the fact that the experiment has a steady state photon count of 30

ntraj = 500

save_path = "data/"
name = "realistic"
overwrite = False

timescale = 1e-9  # ns

from qubit_builder import build_qubit, build_resonator

qubit = build_qubit(config, timescale)
resonator = build_resonator(config, timescale, levels=20)

from devices.system import QubitResonatorSystem
from simulation.experiment import (
    LindbladExperiment,
    StochasticMasterEquationExperiment,
)

from analysis.auto import automatic_analysis

from devices.pulses import SquareCosinePulse

resonator_pulse = SquareCosinePulse(amplitude=25e-3, frequency=config["fr"] * timescale)

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

# # # Lindblad Experiment
# experiment = LindbladExperiment(
#     system,
#     initial_states,
#     times,
#     expectation_operators=[
#         system.resonator_I(),
#         system.resonator_Q(),
#         system.photon_number_operator(),
#     ],
#     only_store_final=False,
#     store_states=True,
#     save_path=os.path.join(save_path, name + "_lindblad.pkl"),
# )

# start_time = time.time()
# results = experiment.run()
# print(f"Time for Lindblad: {time.time() - start_time}")

# from analysis.Q_func import qfunc_plotter_with_time_slider

# automatic_analysis(results)
# plt.show()
# # qfunc_plotter_with_time_slider(results, interval=10, resolution=100, time_steps=1)

# import gc

# gc.collect()


# Stochastic Master Equation Experiment
times = np.arange(0, 1000, 10, dtype=np.float64)
system = QubitResonatorSystem(
    qubit,
    resonator,
    resonator_pulse=resonator_pulse,
    coupling_strength=config["g"] * timescale,
    readout_efficiency=config["eta"],
).dispersive_approximation(config["chi"] * timescale)


if not os.path.exists(os.path.join(save_path, name + "_sme.pkl")) or overwrite:
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
        ntraj=ntraj,
        nsubsteps=5,
        save_path=os.path.join(save_path, name + "_sme.pkl"),
    )

    start_time = time.time()
    results = experiment.run()
    print(f"Time for SME: {time.time() - start_time}")

    import gc

    gc.collect()

    automatic_analysis(results)
else:
    results = pickle.load(open(os.path.join(save_path, name + "_sme.pkl"), "rb"))

    automatic_analysis(results)

measurements = (
    np.squeeze(results.measurements)
    / np.sqrt(config["kappa"] * timescale * config["eta"] * 2)
).real

plt.rcParams["axes.titlesize"] = 18  # because of three subplots this size is better
plt.rcParams["figure.figsize"] = (18, 6)  # Make them more square

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("mycmap", ["C0", "C1"], N=2)

from matplotlib.gridspec import GridSpec

data = measurements

fig = plt.figure(tight_layout=True)
gs = GridSpec(2, 3, figure=fig)

ax_trajectory_Q = fig.add_subplot(gs[1, 0])
ax_trajectory_I = fig.add_subplot(gs[0, 0], sharex=ax_trajectory_Q)
ax_traj = fig.add_subplot(gs[:, 0], visible=False)
ax_scatter = fig.add_subplot(gs[:, 1])
ax_histogram = fig.add_subplot(gs[:, 2])


sample = 1

weights_I = np.abs(
    data[0, :, :, 0].mean(axis=0) - data[1, :, :, 0].mean(axis=0)
) / np.var(data[0, :, :, 0].mean(axis=0) - data[1, :, :, 0].mean(axis=0))

weights_Q = np.abs(
    data[0, :, :, 1].mean(axis=0) - data[1, :, :, 1].mean(axis=0)
) / np.var(data[0, :, :, 1].mean(axis=0) - data[1, :, :, 1].mean(axis=0))


# Normalize the integration to give same units
weights_I *= len(weights_I) / weights_I.sum()
weights_Q *= len(weights_Q) / weights_Q.sum()

data[:, :, :, 0] *= weights_I
data[:, :, :, 1] *= weights_Q

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

all_I = np.concatenate(
    [
        data.sum(axis=2)[0, :, 0],
        data.sum(axis=2)[1, :, 0],
    ]
)
all_Q = np.concatenate(
    [
        data.sum(axis=2)[0, :, 1],
        data.sum(axis=2)[1, :, 1],
    ]
)
states = np.concatenate(
    [
        np.zeros_like(data.sum(axis=2)[0, :, 1]),
        np.ones_like(data.sum(axis=2)[0, :, 1]),
    ]
)

lda = LDA()
lda.fit(np.stack([all_I, all_Q]).T, states)


# Scatter plot
def make_scatter_plot():
    all_I = np.concatenate(
        [
            data.sum(axis=2)[0, :, 0],
            data.sum(axis=2)[1, :, 0],
        ]
    )
    all_Q = np.concatenate(
        [
            data.sum(axis=2)[0, :, 1],
            data.sum(axis=2)[1, :, 1],
        ]
    )

    states = np.concatenate(
        [
            np.zeros_like(data.sum(axis=2)[0, :, 1]),
            np.ones_like(data.sum(axis=2)[0, :, 1]),
        ]
    )

    ax_scatter.scatter(
        all_I,
        all_Q,
        c=states,
        alpha=0.5,
        cmap=cmap,
        rasterized=True,
    )

    ax_scatter.plot(
        [], [], color="C0", label="Ground", alpha=0.75, marker="o", linestyle="None"
    )
    ax_scatter.plot(
        [], [], color="C1", label="Excited", alpha=0.75, marker="o", linestyle="None"
    )

    ax_scatter.set(
        xlabel="I (mV)",
        ylabel="Q (mV)",
        title="Integrated Signal w Weights",
    )

    ax_scatter.set(xlim=ax_scatter.get_xlim(), ylim=ax_scatter.get_ylim())
    ax_scatter.autoscale(False)

    projection_line_x = np.linspace(*ax_scatter.get_xlim(), 100)
    projection_line_y = (
        lda.coef_[0][1] * (projection_line_x - lda.xbar_[0]) + lda.xbar_[1]
    )

    ax_scatter.plot(
        projection_line_x,
        projection_line_y,
        color="black",
        linestyle="--",
        label="projection_axis",
    )
    ax_scatter.legend(fontsize=12)
    # ax_scatter.plot(projection_line_y, projection_line_x, color="gray", linestyle="--")


make_scatter_plot()


# Trajectory Plot
def make_weight_plot():
    # Example Trajectory
    time = results.times

    ax_trajectory_I.plot(time, weights_I, label="Weights", alpha=0.75, color="k")

    ax_trajectory_Q.plot(time, weights_Q, alpha=0.75, color="k")

    ax_trajectory_I.tick_params(labelbottom=False)

    ax_trajectory_Q.set(
        xlabel="t (ns)",
        ylabel="Weights Q",
    )
    ax_trajectory_I.set(
        ylabel="Weights I",
        title="Calculated Weights",
    )

    ax_trajectory_I.legend(fontsize=12)


make_weight_plot()


# Histogram Plot
from sklearn.metrics import det_curve


def make_histogram_plot():
    transformed = lda.transform(np.stack([all_I, all_Q]).T)

    ax_histogram.hist(
        transformed[states == 0],
        bins=30,
        color="C0",
        alpha=0.5,
        label="Ground",
        density=True,
    )
    ax_histogram.hist(
        transformed[states == 1],
        bins=30,
        color="C1",
        alpha=0.5,
        label="Excited",
        density=True,
    )

    ax_histogram.set(
        xlabel="LDA - projection",
        ylabel="Density",
        title="Histogram of the Projected Data",
        ylim=(0, 1.3 * ax_histogram.get_ylim()[1]),
    )

    fpr, fnr, threshholds = det_curve(states, transformed)

    ax_fidelity = ax_histogram.twinx()
    ax_fidelity.plot(
        threshholds, 1 - fpr - fnr, color="C2", label="Fidelity", linestyle="--"
    )
    ax_fidelity.set_ylabel("Fidelity")
    ax_fidelity.set_ylim(0, 1)

    ax_fidelity.vlines(
        threshholds[np.argmax(1 - fpr - fnr)],
        *ax_fidelity.get_ylim(),
        linestyle="-",
        color="C2",
        label="optimal threshold",
    )

    ax_histogram.legend(fontsize=12, loc="upper left")
    ax_fidelity.legend(fontsize=12, loc="upper right")

    print(f"Optimal threshold: {threshholds[np.argmax(1 - fpr - fnr)]}")
    print(
        f"Optimal fidelity: {1 - fpr[np.argmax(1 - fpr - fnr)] - fnr[np.argmax(1 - fpr - fnr)]}"
    )
    with open("log/weighted.txt", "w") as f:
        f.write(f"Optimal threshold: {threshholds[np.argmax(1 - fpr - fnr)]}\n")
        f.write(
            f"Optimal fidelity: {1 - fpr[np.argmax(1 - fpr - fnr)] - fnr[np.argmax(1 - fpr - fnr)]}\n"
        )


make_histogram_plot()


fig.savefig("figures/weigted_simulation.pdf")
