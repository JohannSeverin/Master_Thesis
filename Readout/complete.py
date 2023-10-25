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


sample = 587


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

    random_order = np.random.permutation(len(all_I))

    ax_scatter.scatter(
        all_I[random_order],
        all_Q[random_order],
        c=states[random_order],
        s=15,
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

    # projection_line_x = np.linspace(*ax_scatter.get_xlim(), 100)
    # projection_line_y = (
    #     lda.coef_[0][1] * (projection_line_x - lda.xbar_[0]) + lda.xbar_[1]
    # )

    # ax_scatter.plot(
    #     projection_line_x,
    #     projection_line_y,
    #     color="black",
    #     linestyle="--",
    #     label="projection_axis",
    # )

    x_min, x_max = ax_scatter.get_xlim()
    y_min, y_max = ax_scatter.get_ylim()

    # new_min = min(x_min, y_min)
    # new_max = max(x_max, y_max)

    ax_scatter.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    # ax_scatter.autoscale(False)

    projection_line_x = np.linspace(x_min, x_max, 200)
    projection_line_y = np.linspace(y_min, y_max, 200)
    xx, yy = np.meshgrid(projection_line_x, projection_line_y)
    labels = (
        lda.transform(np.stack([xx.flatten(), yy.flatten()]).T).reshape(xx.shape) > 0
    )

    ax_scatter.contour(
        xx[:, :],
        yy[:, :],
        labels[:, :],
        levels=[0.5],
        linestyles="--",
        alpha=1.00,
        colors="k",
    )

    ax_scatter.plot([], [], color="k", linestyle="--", label="Decision Boundary")

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
        alpha=0.50,
        label="Ground",
        density=True,
    )
    ax_histogram.hist(
        transformed[states == 1],
        bins=30,
        color="C1",
        alpha=0.50,
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
        best_fpr * (1 - best_fpr) + best_fnr * (1 - best_fnr)
    ) / np.sqrt(sum(states == 0))

    print(sum(states == 0))

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
minimizer = Minuit(cost, mu1=-1.5, sigma1=0.75, mu2=+1.5, sigma2=0.75, p=0.5)
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

    random_order = np.random.permutation(len(all_I))

    ax_scatter.scatter(
        all_I[random_order],
        all_Q[random_order],
        c=states[random_order],
        s=15,
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

    x_min, x_max = ax_scatter.get_xlim()
    y_min, y_max = ax_scatter.get_ylim()

    # new_min = min(x_min, y_min)
    # new_max = max(x_max, y_max)

    ax_scatter.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    # ax_scatter.autoscale(False)

    projection_line_x = np.linspace(x_min, x_max, 200)
    projection_line_y = np.linspace(y_min, y_max, 200)
    xx, yy = np.meshgrid(projection_line_x, projection_line_y)
    labels = (
        lda.transform(np.stack([xx.flatten(), yy.flatten()]).T).reshape(xx.shape) > 0
    )

    ax_scatter.contour(
        xx[:, :],
        yy[:, :],
        labels[:, :],
        levels=[0.5],
        linestyles="--",
        alpha=1.00,
        colors="k",
    )

    ax_scatter.plot([], [], color="k", linestyle="--", label="Decision Boundary")

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
        alpha=0.50,
        label="Ground",
        density=True,
    )
    ax_histogram.hist(
        transformed[states == 1],
        bins=30,
        color="C1",
        alpha=0.50,
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
        best_fpr * (1 - best_fpr) + best_fnr * (1 - best_fnr)
    ) / np.sqrt(sum(states == 0))

    # print(best_fpr, best_fnr, sum(states == 0), fidelity_error)

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
states = np.concatenate(
    [
        np.zeros_like(data.I_ground.sum("adc_timestamp").values),
        np.ones_like(data.I_excited.sum("adc_timestamp").values),
    ]
)
cost = UnbinnedNLL(transformed.flatten(), func)
minimizer = Minuit(cost, mu1=-1.5, sigma1=0.75, mu2=+1.5, sigma2=0.75, p=0.5)
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


# Check with simulated data
import pickle

transformed_sim = pickle.load(
    open(
        "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Simulations/budgets/transformed.pkl",
        "rb",
    )
)

plt.style.use("../code/matplotlib_style/inline_figure.mplstyle")

fig, all_axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), sharex=True)

plt.rcParams["axes.titlesize"] = 22  # because of three subplots this size is better
axes = all_axes[0, :]

# axes.figure()
densities, bins, _ = axes[0].hist(
    transformed_sim["transformed"],
    density=True,
    bins=30,
    color="C4",
    label="Simulated",
    histtype="step",
    linewidth=3,
    zorder=5,
    alpha=0.8,
)
axes[0].hist(
    transformed,
    density=True,
    bins=bins,
    color="k",
    label="Experimental",
    histtype="step",
    linewidth=3,
    linestyle="--",
)

axes[0].set(
    ylabel="Density",
    title="Full Distributions",
    ylim=(axes[0].get_ylim()[0], 1.3 * axes[0].get_ylim()[1]),
)


# Cumulative distribution
densities_sim, bins, _ = axes[1].hist(
    transformed_sim["transformed"],
    density=True,
    bins=30,
    color="C4",
    label="Simulated",
    histtype="step",
    linewidth=3,
    cumulative=True,
    alpha=0.8,
    zorder=5,
)

densities_exp, _, _ = axes[1].hist(
    transformed,
    density=True,
    bins=bins,
    color="k",
    linestyle="--",
    label="Experiment",
    histtype="step",
    linewidth=3,
    cumulative=True,
)

axes[1].set(
    ylabel="Density",
    title="Cumulative Distributions (Full)",
    xlim=(np.min(transformed), 0.999 * np.max(transformed)),
)

axes[0].legend(fontsize=18, loc="upper right")
axes[1].legend(fontsize=18, loc="upper left")

# fig.savefig("Figs/Weighted_comparison_with_simmulation.pdf", bbox_inches="tight")

# Another figure where we compare the ground and excited state distributions
# fig, axes = plt.subplots(ncols=2)
axes = all_axes[1, :]

densities, bins, _ = axes[0].hist(
    transformed_sim["transformed"][transformed_sim["states"] == 0],
    density=True,
    bins=30,
    color="C0",
    label="Simulated",
    histtype="step",
    linewidth=3,
    alpha=0.8,
    zorder=5,
)
axes[0].hist(
    transformed[states == 0],
    density=True,
    bins=bins,
    color="k",
    label="Experimental",
    histtype="step",
    linewidth=3,
    linestyle="--",
)

axes[0].set(
    xlabel="LDA - projection",
    ylabel="Density",
    title="Ground State Distribution",
)
axes[0].legend(fontsize=18, loc="upper right")

densities, bins, _ = axes[1].hist(
    transformed_sim["transformed"][transformed_sim["states"] == 1],
    density=True,
    bins=30,
    color="C1",
    label="Simulated",
    histtype="step",
    alpha=0.8,
    linewidth=3,
    zorder=5,
)
axes[1].hist(
    transformed[states == 1],
    density=True,
    bins=bins,
    color="k",
    label="Experimental",
    histtype="step",
    linewidth=3,
    linestyle="--",
)

axes[1].set(
    xlabel="LDA - projection",
    ylabel="Density",
    title="Excited State Distribution",
)

axes[1].legend(fontsize=18, loc="upper left")

fig.tight_layout()
fig.savefig("Figs/Weighted_comparison_with_simmulation.pdf", bbox_inches="tight")

from scipy.stats import ks_2samp

ks_full = ks_2samp(transformed_sim["transformed"].flatten(), transformed.flatten())

ks_ground = ks_2samp(
    transformed_sim["transformed"][transformed_sim["states"] == 0].flatten(),
    transformed[states == 0].flatten(),
)

ks_excited = ks_2samp(
    transformed_sim["transformed"][transformed_sim["states"] == 1].flatten(),
    transformed[states == 1].flatten(),
)

with open("Logs/Comparison.txt", "w") as f:
    f.write(f"KS test for full data: {ks_full}\n")
    f.write(f"KS test for ground state: {ks_ground}\n")
    f.write(f"KS test for excited state: {ks_excited}\n")

print(f"KS test for full data: {ks_full}\n")
print(f"KS test for ground state: {ks_ground}\n")
print(f"KS test for excited state: {ks_excited}\n")

# print(ks_2samp(transformed_sim["transformed"].flatten(), transformed.flatten()))
# print(
#     ks_2samp(
#         transformed_sim["transformed"][transformed_sim["states"] == 0].flatten(),
#         transformed[states == 0].flatten(),
#     )
# )
# print(
#     ks_2samp(
#         transformed_sim["transformed"][transformed_sim["states"] == 1].flatten(),
#         transformed[states == 1].flatten(),
#     )
# )


# axes[0].hist(
#     transformed_sim["transformed"][transformed_sim["states"] == 0],
#     density=True,
#     bins=30,
#     linewidth=5,
#     histtype="step",
# )
# axes[0].hist(
#     transformed[states == 0], density=True, bins=30, linewidth=5, histtype="step"
# )

# axes[1].hist(
#     transformed_sim["transformed"][transformed_sim["states"] == 1],
#     density=True,
#     bins=30,
#     linewidth=5,
#     histtype="step",
# )
# axes[1].hist(
#     transformed[states == 1], density=True, bins=30, linewidth=5, histtype="step"
# )


# fig.tight_layout()
