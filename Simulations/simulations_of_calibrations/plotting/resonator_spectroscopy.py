import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../../code/matplotlib_style/inline_figure.mplstyle")

import json, sys, os, pickle
import time

sys.path.append("../..")


name = "resonator_spectroscopy"

schroedinger_data = pickle.load(open("../data/" + name + "_schroedinger.pkl", "rb"))
monte_carlo_data = pickle.load(open("../data/" + name + "_monte_carlo.pkl", "rb"))
lindblad_data = pickle.load(open("../data/" + name + "_lindblad.pkl", "rb"))

import xarray as xr

experimental_data = xr.open_dataset(
    "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Calibrations/Data/resonator_spectroscopy/dataset.nc"
)

# Setup Figure
fig, axes = plt.subplots(4, 1, figsize=(8, 22), sharex=True)

# # Plotting Simulation Data
ax = axes[1]

x_data = schroedinger_data.sweep_dict["resonator_pulse"]["frequency"]
y_data_g = schroedinger_data.exp_vals[:, 0]
y_data_e = schroedinger_data.exp_vals[:, 1]

ax.plot(x_data, -y_data_g, "o-")
ax.plot(x_data, -y_data_e, "o-")

ax.set(title="Simulation - Schroedinger", ylabel=r"$-\langle aa^\dagger \rangle$")

# ax.legend()

# # Plotting Simulation Data
ax = axes[2]

x_data = monte_carlo_data.sweep_dict["resonator_pulse"]["frequency"]
y_data_g = monte_carlo_data.exp_vals[:, 0]
y_data_e = monte_carlo_data.exp_vals[:, 1]

ax.plot(x_data, -y_data_g, "o-")
ax.plot(x_data, -y_data_e, "o-")

ax.set(title="Simulation - Monte Carlo", ylabel=r"$-\langle aa^\dagger \rangle$")

# # Plotting Simulation Data
ax = axes[3]

x_data = lindblad_data.sweep_dict["resonator_pulse"]["frequency"]
y_data_g = lindblad_data.exp_vals[:, 0]
y_data_e = lindblad_data.exp_vals[:, 1]

ax.plot(x_data, -y_data_g, "o-")
ax.plot(x_data, -y_data_e, "o-")

ax.set(
    title="Simulation - Lindblad",
    ylabel=r"$-\langle aa^\dagger \rangle$",
    xlabel="Frequency (GHz)",
)


# # Plotting Experimental Data
# Imports
data_folder = "../Data/resonator_spectroscopy"
title = "Resonator Spectroscopy"
xlabel = "Frequency (GHz)"
scale_x = 1e9

ylabel = "Readout Signal I (a. u.)"
scale_y = 1e-3

x_data = experimental_data.readout_frequency

y_data_g = experimental_data.sel({"sweep_1": 0}).readout__final__abs__avg
y_err_g = experimental_data.sel({"sweep_1": 0}).readout__final__abs__avg__error

y_data_e = experimental_data.sel({"sweep_1": 1}).readout__final__abs__avg
y_err_e = experimental_data.sel({"sweep_1": 1}).readout__final__abs__avg__error


# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# # Plotting
ax = axes[0]
ax.plot(x_data / scale_x, y_data_g / scale_y, "o", label="Ground Data", color="C0")

ax.plot(x_data / scale_x, y_data_e / scale_y, "o", label="Excited Data", color="C1")

ax.errorbar(
    x_data / scale_x,
    y_data_g / scale_y,
    yerr=y_err_g / scale_y,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)

ax.errorbar(
    x_data / scale_x,
    y_data_e / scale_y,
    yerr=y_err_e / scale_y,
    ls="none",
    color="C1",
    capsize=2,
    elinewidth=1,
)

ax.set(
    title=title,
    # xlabel=xlabel,
    ylabel=ylabel,
)

ax.legend()
# fig.tight_layout()

fig.align_ylabels(axes)
fig.savefig("../Figs/" + name + ".pdf")
