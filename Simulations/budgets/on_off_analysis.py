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
config = json.load(open("../qubit_calibration_2.json", "r"))

# config["g"] *= 2  # Looks like a small error in the coupling strength.
# config["eta"] *= 9
# # This is to make up for the fact that the experiment has a steady state photon count of 30

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
    best_fpr, best_fnr = fpr[np.argmax(fidelity)], fnr[np.argmax(fidelity)]
    fidelity_error = np.sqrt(
        best_fpr * (1 - best_fpr) + best_fnr * (1 - best_fnr)
    ) / np.sqrt(len(truth) / 2)
    if not return_all:
        return max_fidelity, threshholds[np.argmax(fidelity)], fidelity_error
    else:
        return max_fidelity, threshholds[np.argmax(fidelity)], fidelity, fidelity_error


def calculate_fidelity_and_create_plots(
    name, config_dict, ax_scatter_big_figure=None, return_transformed=False
):
    eta = config_dict["eta"] if "eta" in config_dict else config["eta"]
    measurements, results = load_measurements(name, eta)
    weights_I, weights_Q = calculate_weights(measurements)
    # weights_I, weights_Q = np.ones_like(weights_I), np.ones_like(weights_Q)
    # weights_I[-20:] = 0
    # weights_Q[-20:] = 0
    # weights_I[:20] = 0
    # weights_Q[:20] = 0
    I, Q, states = apply_weights(measurements, weights_I, weights_Q)
    transformed, lda = lda_transformation(I, Q, states)
    max_fidelity = max_fidelity_score(transformed, states)

    name = name.split("_")
    name = [n.capitalize() for n in name]
    name = " ".join(name)

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
    for ax in [ax_scatter, ax_scatter_big_figure]:
        if ax:
            random_order = np.random.permutation(len(I))
            ax.scatter(
                I[random_order],
                Q[random_order],
                c=states[random_order],
                alpha=0.75,
                cmap=cmap,
                rasterized=True,
                s=15,
            )

            ax.plot(
                [],
                [],
                color="C0",
                label="Ground",
                alpha=0.75,
                marker="o",
                linestyle="None",
            )
            ax.plot(
                [],
                [],
                color="C1",
                label="Excited",
                alpha=0.75,
                marker="o",
                linestyle="None",
            )

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            new_min = min(x_min, y_min)
            new_max = max(x_max, y_max)

            ax.set(xlim=(new_min, new_max), ylim=(new_min, new_max))
            ax.autoscale(False)

            projection_line_x = np.linspace(*ax.get_xlim(), 200)
            projection_line_y = np.linspace(*ax.get_ylim(), 200)
            xx, yy = np.meshgrid(projection_line_x, projection_line_y)
            labels = (
                lda.transform(np.stack([xx.flatten(), yy.flatten()]).T).reshape(
                    xx.shape
                )
                > max_fidelity[1]
            )

            ax.contour(
                xx[:, :],
                yy[:, :],
                labels[:, :],
                levels=[0.5],
                linestyles="--",
                alpha=1.0,
                colors="k",
            )

            ax.plot([], [], color="k", linestyle="--", label="Decision Boundary")

            if ax == ax_scatter_big_figure:
                ax.set_title(f"IQ for {name[:-8]}", fontsize=22)
                ax.set_aspect("equal")
                ax.text(
                    0.05,
                    0.90,
                    f"Fidelity: {max_fidelity[0]:.3f}",
                    va="top",
                    transform=ax.transAxes,
                )
            else:
                ax.set(
                    xlabel="I (a. u.)",
                    ylabel="Q (a. u.)",
                    title="Integrated Signal w Weights",
                )
                ax.legend(fontsize=12)

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

    if return_transformed:
        return fig, max_fidelity, {"states": states, "transformed": transformed}
    else:
        return fig, max_fidelity


### Setup figure
big_fig, axes_for_big_fig = plt.subplots(
    ncols=3, nrows=2, figsize=(12, 8), sharex=False, sharey=False
)
two_fig, axes_for_two_fig = plt.subplots(
    ncols=2, nrows=1, figsize=(12, 6), sharex=True, sharey=True
)

### Run all config files
config_dicts_all = {
    "realistic": {},
    "perfect": {"eta": 1, "temperature": 0, "T1": 0},
}
config_dict_combinations = {
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
}

### Run Following options of config files
# fidelities = []
# fidelities_errors = []

for col_idx, experiments_to_loop in enumerate(
    [config_dicts_all, config_dict_combinations]
):
    # fidelities_for_experiment = []
    # fidelities_errors_for_experiment = []
    for i, (name, config_dict) in enumerate(experiments_to_loop.items()):
        if col_idx == 0:
            fig, max_fidelity = calculate_fidelity_and_create_plots(
                name + "_sme.pkl", config_dict, axes_for_two_fig[i]
            )
        else:
            fig, max_fidelity = calculate_fidelity_and_create_plots(
                name + "_sme.pkl", config_dict, axes_for_big_fig[i // 3, i % 3]
            )

        fig.savefig(
            os.path.join(save_path, name + "_sme.pdf"),
        )

        if os.path.exists("log.txt"):
            with open("log.txt", "a") as f:
                f.write(
                    f"{name} - max fidelity:  {max_fidelity[0]:.3f} +- {max_fidelity[2]:.3f}\n"
                )
        else:
            with open("log.txt", "w") as f:
                f.write(
                    f"{name} - max fidelity:  {max_fidelity[0]:.3f} +- {max_fidelity[2]:.3f}\n"
                )
        if name == "realistic":
            fig, max_fidelity, transformed = calculate_fidelity_and_create_plots(
                name + "_sme.pkl", config_dict, return_transformed=True
            )
            pickle.dump(transformed, open("transformed.pkl", "wb"))

big_fig.tight_layout()
# big_fig.suptitle("IQ Scatter Plots for Different Parameters", y=1.01)

for i in range(2):
    axes_for_big_fig[i, 0].set_ylabel("Q (a. u.)")

for i in range(3):
    axes_for_big_fig[1, i].set_xlabel("I (a. u.)")

big_fig.savefig(
    os.path.join(save_path, "iq_scatter_budgetting_on_off.pdf"), bbox_inches="tight"
)

two_fig.tight_layout()
# two_fig.suptitle("IQ Scatter Plots for Different Parameters", y=1.01, va="bottom")

for i in range(2):
    axes_for_two_fig[i].set_ylabel("Q (a. u.)")

for i in range(2):
    axes_for_two_fig[i].set_xlabel("I (a. u.)")

two_fig.savefig(
    os.path.join(save_path, "iq_scatter_budgetting_on_off_two.pdf"), bbox_inches="tight"
)


### Plot Fidelities
# fig, ax = plt.subplots(ncols=3, sharey=True)

# symbols = ["$1 - \eta$", "$1 / T_1$", "$\\tau$"]

# for i, (fidelities_for_experiment, label) in enumerate(
#     zip(fidelities, ["Efficiency", "Decay", "Temperature"])
# ):
#     ax[i].plot(
#         [1.1, 1.0, 0.9, 0.75, 0.5, 0.0],
#         fidelities_for_experiment,
#         marker="o",
#         linestyle="None",
#         label=label,
#     )

#     ax[i].errorbar(
#         [1.1, 1.0, 0.9, 0.75, 0.5, 0.0],
#         fidelities_for_experiment,
#         yerr=fidelities_errors[i],
#         linestyle="None",
#         color="k",
#     )

#     ax[i].set(
#         xlabel=f"{label} ({symbols[i]})",
#         ylabel="Fidelity",
#         title=label,
#         # ylim=(0, 1),
#     )

#     ax[i].vlines(
#         1.0,
#         *ax[i].get_ylim(),
#         linestyle="--",
#         color="k",
#         label="Original",
#         alpha=0.75,
#     )

#     ax[i].legend(fontsize=12)

# fig.savefig(os.path.join(save_path, "fidelities_at_different_parameters.pdf"))
