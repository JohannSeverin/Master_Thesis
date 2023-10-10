# Point to the xarrays containing the excited and ground state data
path = "/mnt/c/Users/johan/Downloads/IQ_threshold_141420"

# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys, os

plt.style.use("../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["axes.titlesize"] = 18  # because of three subplots this size is better
plt.rcParams["figure.figsize"] = (18, 12)  # Make them more square

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("mycmap", ["C0", "C1"], N=2)

from matplotlib.gridspec import GridSpec

data = xr.open_dataset(os.path.join(path, "demodulated_dataset.nc")).sel(
    adc_timestamp=slice(0, 1e-6)
)

fig = plt.figure(tight_layout=True)
gs = GridSpec(4, 3, figure=fig)

ax_trajectory_Q = fig.add_subplot(gs[1, 0])
ax_trajectory_I = fig.add_subplot(gs[0, 0], sharex=ax_trajectory_Q)
ax_traj = fig.add_subplot(gs[:2, 0], visible=False)
ax_scatter = fig.add_subplot(gs[:2, 1])
ax_histogram = fig.add_subplot(gs[:2, 2])


sample = 18


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import det_curve

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
        title="Scatter plot of the Integrated Signal",
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
        title="Trajectory of the Demodulated Trace",
    )

    ax_trajectory_I.legend(fontsize=12)


make_trajectory_plot()


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
    fidelity = 1 - fpr - fnr

    ax_fidelity = ax_histogram.twinx()
    ax_fidelity.plot(
        threshholds, fidelity, color="C2", label="Fidelity", linestyle="--"
    )
    ax_fidelity.set_ylabel("Fidelity")
    ax_fidelity.set_ylim(0, 1)

    ax_fidelity.vlines(
        threshholds[np.argmax(fidelity)],
        *ax_fidelity.get_ylim(),
        linestyle="-",
        color="C2",
        label="optimal threshold",
    )

    ax_histogram.legend(fontsize=12, loc="upper left")
    ax_fidelity.legend(fontsize=12, loc="upper right")

    best_fpr, best_fnr = fpr[np.argmax(fidelity)], fnr[np.argmax(fidelity)]
    fidelity_error = np.sqrt(
        best_fpr * (1 - best_fpr) / sum(states == 0)
        + best_fnr * (1 - best_fnr) / sum(states == 0)
    )

    with open("Logs/Introduction.txt", "w") as f:
        f.write("SIMPLE WEIGHTS")
        f.write(f"Optimal threshold: {threshholds[np.argmax(1 - fpr - fnr)]}\n")
        f.write(
            f"Optimal fidelity: {1 - best_fpr - best_fnr:.3f} +- {fidelity_error:.3f} \n"
        )

    print(f"Optimal threshold: {threshholds[np.argmax(fidelity)]}\n")
    print(f"Optimal fidelity: {1 - best_fpr - best_fnr:.3f} +- {fidelity_error:.3f} \n")


make_histogram_plot()


# Histogram Plot
from sklearn.metrics import det_curve
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import norm

func = lambda x, mu1, sigma1, mu2, sigma2, p: (1 - p) * norm.pdf(
    x, mu1, sigma1
) + p * norm.pdf(x, mu2, sigma2)


transformed = lda.transform(np.stack([all_I, all_Q]).T)
cost = UnbinnedNLL(transformed.flatten(), func)
minimizer = Minuit(cost, mu1=-1, sigma1=1, mu2=+1, sigma2=1, p=0.5)
minimizer.migrad()

with open("Logs/Introduction.txt", "a") as f:
    f.write(
        f"Fit of gaussians: \n mu1: {minimizer.values['mu1']} +- {minimizer.errors['mu1']} \t mu2: {minimizer.values['mu2']} +- {minimizer.errors['mu2']} \n sigma1: {minimizer.values['sigma1']} +- {minimizer.errors['sigma1']} \t sigma2: {minimizer.values['sigma2']} +- {minimizer.errors['sigma2']} \n p: {minimizer.values['p']} +- {minimizer.errors['p']} \n"
    )
    SNR = np.abs(minimizer.values["mu1"] - minimizer.values["mu2"]) / np.sqrt(
        minimizer.values["sigma1"] ** 2 + minimizer.values["sigma2"] ** 2
    )
    f.write(f"Signal to noise: {SNR:.4f} \n")

ax_trajectory_Q = fig.add_subplot(gs[3, 0])
ax_trajectory_I = fig.add_subplot(gs[2, 0], sharex=ax_trajectory_Q)
ax_traj = fig.add_subplot(gs[2:, 0], visible=False)
ax_scatter = fig.add_subplot(gs[2:, 1])
ax_histogram = fig.add_subplot(gs[2:, 2])


sample = 1

weights_I = np.abs(
    data.I_ground.mean("sample") - data.I_excited.mean("sample")
) / np.var(data.I_ground.mean("sample") - data.I_excited.mean("sample"))

weights_Q = np.abs(
    data.Q_ground.mean("sample") - data.Q_excited.mean("sample")
) / np.var(data.Q_ground.mean("sample") - data.Q_excited.mean("sample"))

# Normalize the integration to give same units
weights_I *= len(weights_I) / weights_I.sum()
weights_Q *= len(weights_Q) / weights_Q.sum()

data["I_ground"] *= weights_I
data["I_excited"] *= weights_I
data["Q_ground"] *= weights_Q
data["Q_excited"] *= weights_Q

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
    time = data.I_ground.adc_timestamp.values * 1e9

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
    fidelity = 1 - fpr - fnr

    ax_fidelity = ax_histogram.twinx()
    ax_fidelity.plot(
        threshholds, fidelity, color="C2", label="Fidelity", linestyle="--"
    )
    ax_fidelity.set_ylabel("Fidelity")
    ax_fidelity.set_ylim(0, 1)

    ax_fidelity.vlines(
        threshholds[np.argmax(fidelity)],
        *ax_fidelity.get_ylim(),
        linestyle="-",
        color="C2",
        label="optimal threshold",
    )

    ax_histogram.legend(fontsize=12, loc="upper left")
    ax_fidelity.legend(fontsize=12, loc="upper right")

    best_fpr, best_fnr = fpr[np.argmax(fidelity)], fnr[np.argmax(fidelity)]
    fidelity_error = np.sqrt(
        best_fpr * (1 - best_fpr) / sum(states == 0)
        + best_fnr * (1 - best_fnr) / sum(states == 0)
    )

    with open("Logs/Introduction.txt", "a") as f:
        f.write("OPTIMAL WEIGHTS")
        f.write(f"Optimal threshold: {threshholds[np.argmax(1 - fpr - fnr)]}\n")
        f.write(
            f"Optimal fidelity: {1 - best_fpr - best_fnr:.3f} +- {fidelity_error:.3f} \n"
        )


make_histogram_plot()

# Histogram Plot
from sklearn.metrics import det_curve
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import norm

func = lambda x, mu1, sigma1, mu2, sigma2, p: (1 - p) * norm.pdf(
    x, mu1, sigma1
) + p * norm.pdf(x, mu2, sigma2)


transformed = lda.transform(np.stack([all_I, all_Q]).T)
cost = UnbinnedNLL(transformed.flatten(), func)
minimizer = Minuit(cost, mu1=-1, sigma1=1, mu2=+1, sigma2=1, p=0.5)
minimizer.migrad()

with open("Logs/Introduction.txt", "a") as f:
    f.write(
        f"Fit of gaussians: \n mu1: {minimizer.values['mu1']} +- {minimizer.errors['mu1']} \t mu2: {minimizer.values['mu2']} +- {minimizer.errors['mu2']} \n sigma1: {minimizer.values['sigma1']} +- {minimizer.errors['sigma1']} \t sigma2: {minimizer.values['sigma2']} +- {minimizer.errors['sigma2']} \n p: {minimizer.values['p']} +- {minimizer.errors['p']} \n"
    )
    SNR = np.abs(minimizer.values["mu1"] - minimizer.values["mu2"]) / np.sqrt(
        minimizer.values["sigma1"] ** 2 + minimizer.values["sigma2"] ** 2
    )
    f.write(f"Signal to noise: {SNR:.4f}")


fig.savefig("Figs/Introduction.pdf")

# fig.tight_layout()
