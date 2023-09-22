# Point to the xarrays containing the excited and ground state data
path = "/mnt/c/Users/johan/Downloads/selected_files_2023920_211420/readout test_092950"

# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys, os

plt.style.use("../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["axes.titlesize"] = 18  # because of three subplots this size is better
plt.rcParams["figure.figsize"] = (18, 6)  # Make them more square

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("mycmap", ["C0", "C1"], N=2)

from matplotlib.gridspec import GridSpec

data = xr.open_dataset(os.path.join(path, "demodulated_dataset.nc")).sel(
    adc_timestamp=slice(0, 1e-6)
)

fig = plt.figure(tight_layout=True)
gs = GridSpec(2, 3, figure=fig)

ax_trajectory_Q = fig.add_subplot(gs[1, 0])
ax_trajectory_I = fig.add_subplot(gs[0, 0], sharex=ax_trajectory_Q)
ax_traj = fig.add_subplot(gs[:, 0], visible=False)
ax_scatter = fig.add_subplot(gs[:, 1])
ax_histogram = fig.add_subplot(gs[:, 2])


sample = 1


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

all_I = np.concatenate(
    [
        data.I_ground.sum("adc_timestamp").values,
        data.I_excited.sum("adc_timestamp").values,
    ]
)
all_Q = np.concatenate(
    [
        data.Q_ground.sum("adc_timestamp").values,
        data.Q_excited.sum("adc_timestamp").values,
    ]
)

states = np.concatenate(
    [
        np.zeros_like(data.I_ground.sum("adc_timestamp").values),
        np.ones_like(data.I_excited.sum("adc_timestamp").values),
    ]
)

lda = LDA()
lda.fit(np.stack([all_I, all_Q]).T, states)


# Scatter plot
def make_scatter_plot():
    all_I = np.concatenate(
        [
            data.I_ground.sum("adc_timestamp").values,
            data.I_excited.sum("adc_timestamp").values,
        ]
    )
    all_Q = np.concatenate(
        [
            data.Q_ground.sum("adc_timestamp").values,
            data.Q_excited.sum("adc_timestamp").values,
        ]
    )

    states = np.concatenate(
        [
            np.zeros_like(data.I_ground.sum("adc_timestamp").values),
            np.ones_like(data.I_excited.sum("adc_timestamp").values),
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
        title="Scatter plot of the demodulated data",
    )

    ax_scatter.indicate_inset(
        bounds=[
            data.I_ground.sel(sample=sample).sum("adc_timestamp"),
            data.Q_ground.sel(sample=sample).sum("adc_timestamp"),
            0.01,
            0.01,
        ],
        inset_ax=ax_traj,
        edgecolor="black",
    )

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
def make_trajectory_plot():
    # Example Trajectory
    time = data.I_ground.adc_timestamp.values * 1e9
    ax_trajectory_I.plot(
        time,
        data.I_ground.sel(sample=sample),
        label="Sample",
        alpha=0.75,
    )

    ax_trajectory_Q.plot(time, data.Q_ground.sel(sample=sample), alpha=0.75)

    ax_trajectory_I.tick_params(labelbottom=False)

    # Mean trajectories
    ax_trajectory_I.plot(
        time,
        data.I_ground.mean("sample"),
        color="C0",
        linestyle="--",
        label="Mean Ground",
    )

    ax_trajectory_I.plot(
        time,
        data.I_excited.mean("sample"),
        color="C1",
        linestyle="--",
        label="Mean Excited",
    )

    ax_trajectory_Q.plot(
        time,
        data.Q_ground.mean("sample"),
        color="C0",
        linestyle="--",
    )

    ax_trajectory_Q.plot(
        time,
        data.Q_excited.mean("sample"),
        color="C1",
        linestyle="--",
    )

    ax_trajectory_Q.set(
        xlabel="t (ns)",
        ylabel="Q (mV)",
    )
    ax_trajectory_I.set(
        ylabel="I (mV)",
        title="Trajectory of the demodulated data",
    )

    ax_trajectory_I.legend(fontsize=12)


make_trajectory_plot()


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

    with open("Logs/Introduction.txt", "w") as f:
        f.write(f"Optimal threshold: {threshholds[np.argmax(1 - fpr - fnr)]}\n")
        f.write(
            f"Optimal fidelity: {1 - fpr[np.argmax(1 - fpr - fnr)] - fnr[np.argmax(1 - fpr - fnr)]}\n"
        )


make_histogram_plot()


fig.savefig("Figs/Introduction.pdf")

# fig.tight_layout()
