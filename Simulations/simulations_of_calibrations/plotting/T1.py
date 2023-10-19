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
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)


axes[1].set_yticks([0, 1])
axes[1].set_yticklabels([0, 1])
axes[2].set_yticklabels([])
axes[3].set_yticklabels([])


# Plotting Simulation Data
ax = axes[1]

times = schroedinger_data.times[::250] * 1e-3
y_data = schroedinger_data.exp_vals[::250]

ax.plot(times, y_data, "o", label="Schroedinger")

ax.set(
    ylabel=r"$\langle P_{1} \rangle$",
    ylim=(-0.05, 1.05),
    xlabel="Time (µs)",
    title="SE - Full",
)
# ax.legend()

# Plotting Simulation Data
ax = axes[2]

times = monte_carlo_data.times[::250] * 1e-3
y_data = monte_carlo_data.exp_vals[::250]

ax.plot(times, y_data, "o", label="Monte Carlo")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    xlabel="Time (µs)",
    ylim=(-0.05, 1.05),
    title="MC - Full",
)
# ax.legend()

# Plotting Simulation Data
ax = axes[3]

times = lindblad_data.times[::250] * 1e-3
y_data = lindblad_data.exp_vals[::250]

ax.plot(times, y_data, "o", label="Lindblad")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    xlabel="Time (µs)",
    ylim=(-0.05, 1.05),
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
ax.plot(x_data / scale_x, y_data / scale_y, "o", label="Data")

ax.errorbar(
    x_data / scale_x,
    y_data / scale_y,
    yerr=y_err / scale_y,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)

ax.set(
    title=title,
    xlabel="Time (µs)",
    ylabel=ylabel,
    xlim=(-1, 11),
)

# ax.legend()

fig.tight_layout()

fig.align_ylabels(axes)
fig.savefig("../Figs/" + name + ".pdf")
