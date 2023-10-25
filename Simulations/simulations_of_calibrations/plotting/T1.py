import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../../code/matplotlib_style/fullwidth_figure.mplstyle")

import json, sys, os, pickle
import time

sys.path.append("../..")


name = "qubit_T1"

schroedinger_data = pickle.load(open("../data/" + name + "_schoedinger.pkl", "rb"))
monte_carlo_data = pickle.load(open("../data/" + name + "_monte_carlo.pkl", "rb"))
lindblad_data = pickle.load(open("../data/" + name + "_lindblad.pkl", "rb"))

import xarray as xr

experimental_data = xr.open_dataset(
    "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Calibrations/Data/T1/dataset.nc"
)
x_data = experimental_data.readout_delay

# Setup Figure
plt.rcParams["axes.titlesize"] = 16
fig, two_row_axes = plt.subplots(
    2,
    4,
    figsize=(14, 8),
    sharey=False,
    sharex=True,
    gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1, 1, 1], "hspace": 0.30},
)

axes = two_row_axes[0, :]

axes[1].set_yticks([0.25, 0.50, 0.75, 1])
axes[1].set_yticklabels(["$1 / 4$", "$1 / 2$", "$3 / 4$", "$1$"])
axes[2].set_yticklabels([])
axes[3].set_yticklabels([])


# Plotting Simulation Data
ax = axes[1]

times = schroedinger_data.times[::250] * 1e-3
y_data = schroedinger_data.exp_vals[::250]

ax.plot(times, y_data, ".", label="Schroedinger")

ax.set(
    ylabel=r"Occupation, $\langle P_{1} \rangle$",
    ylim=(0.10, 1.05),
    xlim=(-1, 12),
    title="SE - Full",
)
ax_inset = ax.inset_axes([0.5, 0.25, 0.4, 0.4])
ax_inset.plot(times[40:50], y_data[40:50], ".", label="Schroedinger")
ax_inset.set(ylim=(0.97, 1.02))
ax_inset.xaxis.set_ticklabels([])
ax_inset.yaxis.set_ticklabels([])
ax_inset.spines["right"].set_visible(True)
ax_inset.spines["top"].set_visible(True)
ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=2)


# Plotting Simulation Data
ax = axes[2]

times = monte_carlo_data.times[::250] * 1e-3
y_data = monte_carlo_data.exp_vals[::250]

ax.plot(times, y_data, ".", label="Monte Carlo")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    ylim=(0.10, 1.05),
    title="MC - Full",
)
# ax.legend()

# Plotting Simulation Data
ax = axes[3]

times = lindblad_data.times[::250] * 1e-3
y_data = lindblad_data.exp_vals[::250]

ax.plot(times, y_data, ".", label="Lindblad")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    ylim=(0.10, 1.05),
    xlim=(-1, 12),
    title="ME - Full",
)
# ax.legend()


# # Plotting Experimental Data
title = "Experiment"
xlabel = "Waiting Time (µs)"
scale_x = 1e-6

ylabel = "Readout Signal I (a. u.)"
scale_y = 1e-3

y_data = experimental_data.readout__final__I__avg
y_err = experimental_data.readout__final__I__avg__error

ax = axes[0]
ax.plot(
    x_data[x_data < 1e-5] / scale_x, y_data[x_data < 1e-5] / scale_y, ".", label="Data"
)

ax.errorbar(
    x_data[x_data < 1e-5] / scale_x,
    y_data[x_data < 1e-5] / scale_y,
    yerr=y_err[x_data < 1e-5] / scale_y,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)

ax.set(
    title=title,
    ylabel=ylabel,
    xlim=(-1, 11),
)

# ax.legend()

# fig.tight_layout()

fig.align_ylabels(axes)
# fig.savefig("../Figs/" + name + ".pdf")


# import numpy as np
# import matplotlib.pyplot as plt

# plt.style.use("../../../code/matplotlib_style/fullwidth_figure.mplstyle")

# import json, sys, os, pickle
# import time

# sys.path.append("../..")


# name = "qubit_T1"

schroedinger_data = pickle.load(
    open("../data/" + name + "_schoedinger_dispersive.pkl", "rb")
)
monte_carlo_data = pickle.load(
    open("../data/" + name + "_monte_carlo_dispersive.pkl", "rb")
)
lindblad_data = pickle.load(open("../data/" + name + "_lindblad_dispersive.pkl", "rb"))
sme_data = pickle.load(open("../data/" + name + "_sme_dispersive.pkl", "rb"))

# Setup Figure
# plt.rcParams["axes.titlesize"] = 16
# fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)


axes = two_row_axes[1, :]
axes[1].set_yticks([0.25, 0.50, 0.75, 1])
axes[1].set_yticklabels(["$1 / 4$", "$1 / 2$", "$3 / 4$", "$1$"])
# axes[1].set_yticklabels([0, 1])
axes[2].set_yticklabels([])
axes[3].set_yticklabels([])


# Plotting Simulation Data
ax = axes[1]

times = schroedinger_data.times[::250] * 1e-3
y_data = schroedinger_data.exp_vals[::250]

ax.plot(times, y_data, ".", label="Schroedinger")

ax.set(
    ylabel=r"Occupation, $\langle P_1 \rangle$",
    xlabel="Time (µs)",
    ylim=(0.10, 1.05),
    title="SE - Dispersive",
)

ax_inset = ax.inset_axes([0.5, 0.25, 0.4, 0.4])
ax_inset.plot(times[40:50], y_data[40:50], ".", label="Schroedinger")
ax_inset.set(ylim=(0.97, 1.02))
ax_inset.xaxis.set_ticklabels([])
ax_inset.yaxis.set_ticklabels([])
ax_inset.spines["right"].set_visible(True)
ax_inset.spines["top"].set_visible(True)
ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=2)


# ax.legend()

# Plotting Simulation Data
ax = axes[2]

times = monte_carlo_data.times[::250] * 1e-3
y_data = monte_carlo_data.exp_vals[::250]

ax.plot(times, y_data, ".", label="Monte Carlo")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    xlabel="Time (µs)",
    ylim=(0.10, 1.05),
    title="MC - Dispersive",
)
# ax.legend()

# Plotting Simulation Data
ax = axes[3]

times = lindblad_data.times[::250] * 1e-3
y_data = lindblad_data.exp_vals[::250]

ax.plot(times, y_data, ".", label="Lindblad")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    xlabel="Time (µs)",
    ylim=(0.10, 1.05),
    title="ME - Dispersive",
)
# ax.legend()


# SME Plot
measurements = np.array(sme_data.measurements)
I_measurements = measurements.mean(0)[:-1, 0, 0]

measurements = I_measurements.reshape(-1, 20).mean(1).real
measurements_err = (I_measurements.reshape(-1, 20).real.std(1)) / np.sqrt(20)

ax = axes[0]
ax.plot(sme_data.times[:-1:20] * 1e-3, measurements, ".", label="SME")
ax.errorbar(
    sme_data.times[:-1:20] * 1e-3,
    measurements,
    yerr=measurements_err,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)

ax.set(
    title="SME - Dispersive",
    xlabel="Time (µs)",
    ylabel="Readout Signal I (a. u.)",
)

fig.align_ylabels(two_row_axes)
fig.tight_layout()

# fig.align_ylabels(axes)
fig.savefig("../Figs/" + name + "_dispersive.pdf")
