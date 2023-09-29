import matplotlib.pyplot as plt
import numpy as np
import pickle, os, sys, json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import det_curve


sys.path.append("..")
plt.style.use("../../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["font.size"] = 16
plt.rcParams["figure.figsize"] = (18, 6)
cmap = LinearSegmentedColormap.from_list("mycmap", ["C0", "C1"], N=2)
config = json.load(open("../qubit_calibration.json", "r"))

config["g"] *= 2  # Looks like a small error in the coupling strength.
config["eta"] *= 9
# This is to make up for the fact that the experiment has a steady state photon count of 30

timescale = 1e-9  # ns

path = "data/"
save_path = "figures/"


def load_measurements(name, eta):
    results = pickle.load(open(os.path.join(path, name), "rb"))
    measurements = (
        np.squeeze(results.measurements)
        / np.sqrt(config["kappa"] * timescale * eta * 2)
    ).real
    return measurements, results


def calculate_weights(measurements):
    # calculate weight for I
    mean_path_ground_I = measurements[0, :, :, 0].mean(axis=0)
    mean_path_excited_I = measurements[1, :, :, 0].mean(axis=0)

    var_path_ground_I = measurements[0, :, :, 0].var(axis=0)
    var_path_excited_I = measurements[1, :, :, 0].var(axis=0)

    weights_I = np.abs(mean_path_ground_I - mean_path_excited_I) / (
        var_path_ground_I + var_path_excited_I
    )

    # repeat for Q
    mean_path_ground_Q = measurements[0, :, :, 1].mean(axis=0)
    mean_path_excited_Q = measurements[1, :, :, 1].mean(axis=0)

    var_path_ground_Q = measurements[0, :, :, 1].var(axis=0)
    var_path_excited_Q = measurements[1, :, :, 1].var(axis=0)

    weights_Q = np.abs(mean_path_ground_Q - mean_path_excited_Q) / (
        var_path_ground_Q + var_path_excited_Q
    )

    # Normalize the integration to give same units and return
    weights_I *= len(weights_I) / weights_I.sum()
    weights_Q *= len(weights_Q) / weights_Q.sum()
    return weights_I, weights_Q


def apply_weights(measurements, weights_I, weights_Q):
    all_I = np.concatenate(
        [
            measurements[0, :, :, 0],
            measurements[1, :, :, 0],
        ]
    )
    all_Q = np.concatenate(
        [
            measurements[0, :, :, 1],
            measurements[1, :, :, 1],
        ]
    )

    I = (weights_I * all_I).sum(axis=1)
    Q = (weights_Q * all_Q).sum(axis=1)

    states = np.concatenate(
        [
            np.zeros(measurements.shape[1]),
            np.ones(measurements.shape[1]),
        ]
    )

    return I, Q, states


def lda_transformation(I, Q, states):
    # Train and transform
    lda = LDA()
    lda.fit(np.vstack([I, Q]).T, states)

    transformed = lda.transform(np.stack([I, Q]).T)

    return transformed, lda


def max_fidelity_score(score, truth, return_all=False):
    fpr, fnr, threshholds = det_curve(truth, score)
    fidelity = 1 - fpr - fnr
    max_fidelity = fidelity.max()
    if not return_all:
        return max_fidelity, threshholds[np.argmax(fidelity)]
    else:
        return max_fidelity, threshholds[np.argmax(fidelity)], fidelity


def calculate_fidelity_and_create_plots(name, eta):
    measurements, results = load_measurements(name, eta)
    weights_I, weights_Q = calculate_weights(measurements)
    I, Q, states = apply_weights(measurements, weights_I, weights_Q)
    transformed, lda = lda_transformation(I, Q, states)
    max_fidelity = max_fidelity_score(transformed, states)

    # Setup Figure
    fig = plt.figure(tight_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax_trajectory_Q = fig.add_subplot(gs[1, 0])
    ax_trajectory_I = fig.add_subplot(gs[0, 0], sharex=ax_trajectory_Q)
    ax_traj = fig.add_subplot(gs[:, 0], visible=False)
    ax_scatter = fig.add_subplot(gs[:, 1])
    ax_histogram = fig.add_subplot(gs[:, 2])

    # Weight Plots
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

    # Scatter Plot
    ax_scatter.scatter(
        I,
        Q,
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
        xlabel="I (a. u.)",
        ylabel="Q (a. u.)",
        title="Integrated Signal w Weights",
    )

    x_min, x_max = ax_scatter.get_xlim()
    y_min, y_max = ax_scatter.get_ylim()

    new_min = min(x_min, y_min)
    new_max = max(x_max, y_max)

    ax_scatter.set(xlim=(new_min, new_max), ylim=(new_min, new_max))
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

    # Histogram
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
    ax_fidelity.text(
        0.6,
        0.5,
        f"Fidelity: {max_fidelity[0]:.3f}",
    )

    ax_histogram.legend(fontsize=12, loc="upper left")
    ax_fidelity.legend(fontsize=12, loc="upper right")

    fig.suptitle(
        f"Fidelity Analysis for {name[:-8]}",
    )

    return fig, max_fidelity


### Run Following options of config files
config_dicts = {
    "realistic": {},
    "decay_only": {"eta": 1, "temperature": 0, "T1": config["T1"]},
    "efficiency_only": {"eta": config["eta"], "temperature": 0, "T1": 0},
    "thermal_only": {"eta": 1, "temperature": config["temperature"], "T1": 0},
    "no_decay": {
        "eta": config["eta"],
        "temperature": config["temperature"],
        "T1": 0,
    },
    "perfect_efficiency": {
        "eta": 1,
        "temperature": config["temperature"],
        "T1": config["T1"],
    },
    "zero_temperature": {
        "eta": config["eta"],
        "temperature": 0,
        "T1": config["T1"],
    },
    "perfect": {"eta": 1, "temperature": 0, "T1": 0},
}

for name, config_dict in config_dicts.items():
    fig, max_fidelity = calculate_fidelity_and_create_plots(name + "_sme.pkl", 1)
    fig.savefig(
        os.path.join(save_path, name + "_sme.pdf"),
    )
    # plt.close(fig)

    if os.path.exists("log.txt"):
        with open("log.txt", "a") as f:
            f.write(f"{name} - max fidelity:  {max_fidelity[0]:.3f} +-\n")
    else:
        with open("log.txt", "w") as f:
            f.write(f"{name} - max fidelity:  {max_fidelity[0]:.3f} +-\n")

    # break
