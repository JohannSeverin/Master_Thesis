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


def calculate_fidelity_and_create_plots(name, config_dict, ax_scatter_big_figure=None):
    eta = config_dict["eta"] if "eta" in config_dict else config["eta"]
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
    for ax in [ax_scatter, ax_scatter_big_figure]:
        if ax:
            ax.scatter(
                I,
                Q,
                c=states,
                alpha=0.5,
                cmap=cmap,
                rasterized=True,
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
                > 0.5
            )

            ax.contour(
                xx[:, 50:],
                yy[:, 50:],
                labels[:, 50:],
                levels=[0.5],
                linestyles="--",
                alpha=0.50,
                colors="k",
            )

            if ax == ax_scatter_big_figure:
                ax.set_title("")
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

    return fig, max_fidelity


def scale_eta(eta, factor):
    return 1 - (1 - eta) * factor


def scale_T1(T1, factor):
    return T1 / factor if factor != 0 else 0


### Setup figure
big_fig, axes_for_big_fig = plt.subplots(
    ncols=3, nrows=6, figsize=(12, 22), sharex=True, sharey=True
)


def scale_temperature(temperature, factor):
    return temperature * factor


### Run Following options of config files
config_dicts_efficiency = {
    r"eta_10_increase": {"eta": scale_eta(config["eta"], 1.1)},
    r"eta": {"eta": scale_eta(config["eta"], 1.0)},
    r"eta_10_reduction": {"eta": scale_eta(config["eta"], 0.9)},
    r"eta_25_reduction": {"eta": scale_eta(config["eta"], 0.75)},
    r"eta_50_reduction": {"eta": scale_eta(config["eta"], 0.50)},
    r"eta_100_reduction": {"eta": scale_eta(config["eta"], 0.00)},
}

config_dicts_decay = {
    r"T1_10_increase": {"T1": scale_T1(config["T1"], 1.1)},
    r"T1": {"T1": scale_T1(config["T1"], 1.0)},
    r"T1_10_reduction": {"T1": scale_T1(config["T1"], 0.9)},
    r"T1_25_reduction": {"T1": scale_T1(config["T1"], 0.75)},
    r"T1_50_reduction": {"T1": scale_T1(config["T1"], 0.50)},
    r"T1_100_reduction": {"T1": scale_T1(config["T1"], 0.00)},
}

config_dicts_temperature = {
    r"temperature_10_increase": {
        "temperature": scale_temperature(config["temperature"], 1.1)
    },
    r"temperature": {"temperature": scale_temperature(config["temperature"], 1.0)},
    r"temperature_10_reduction": {
        "temperature": scale_temperature(config["temperature"], 0.9)
    },
    r"temperature_25_reduction": {
        "temperature": scale_temperature(config["temperature"], 0.75)
    },
    r"temperature_50_reduction": {
        "temperature": scale_temperature(config["temperature"], 0.50)
    },
    r"temperature_100_reduction": {
        "temperature": scale_temperature(config["temperature"], 0.00)
    },
}

### Run Following options of config files

for col_idx, experiments_to_loop in enumerate(
    [config_dicts_efficiency, config_dicts_decay, config_dicts_temperature]
):
    for i, (name, config_dict) in enumerate(experiments_to_loop.items()):
        fig, max_fidelity = calculate_fidelity_and_create_plots(
            name + "_sme.pkl", config_dict, axes_for_big_fig[i, col_idx]
        )
        fig.savefig(
            os.path.join(save_path, name + "_sme.pdf"),
        )
        # plt.close(f[ig)

        if os.path.exists("log.txt"):
            with open("log.txt", "a") as f:
                f.write(f"{name} - max fidelity:  {max_fidelity[0]:.3f}\n")
        else:
            with open("log.txt", "w") as f:
                f.write(f"{name} - max fidelity:  {max_fidelity[0]:.3f}\n")

        # break


for col_idx, parameter in enumerate([r"$(1 - \eta)$", r"$(1 / T_1)$", r"$\tau$"]):
    for row_idx, amount in enumerate(
        [r"$1.1$", r"$1.0$", r"$0.9$", r"$0.75$", r"$0.5$", r"$0.0$"]
    ):
        axes_for_big_fig[row_idx, col_idx].text(
            0.05,
            1.00,
            f"{amount} $\\times$ {parameter}",
            va="top",
            transform=axes_for_big_fig[row_idx, col_idx].transAxes,
        )


big_fig.tight_layout()
big_fig.suptitle("IQ Scatter Plots for Different Parameters", y=1.01)

for i in range(6):
    axes_for_big_fig[i, 0].set_ylabel("Q (a. u.)")

for i in range(3):
    axes_for_big_fig[5, i].set_xlabel("I (a. u.)")

big_fig.savefig(
    os.path.join(save_path, "iq_scatter_budgetting.pdf"),
)
