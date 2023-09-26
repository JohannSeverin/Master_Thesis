import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../../code/matplotlib_style/fullwidth_figure.mplstyle")

import json, sys, os, pickle
import time

sys.path.append("../..")


name = "resonator_kappa"

monte_carlo_data = pickle.load(open("../data/" + name + "_monte_carlo.pkl", "rb"))
lindblad_data = pickle.load(open("../data/" + name + "_lindblad.pkl", "rb"))
sme_data = pickle.load(open("../data/" + name + "_sme.pkl", "rb"))

import xarray as xr

experimental_data = xr.open_dataset(
    "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Calibrations/Data/kappa/dataset.nc"
)
x_data = experimental_data.adc_timestamp


# Setup Figure
fig, axes = plt.subplots(2, 2, sharex=True)

# Plotting Monte Carlo Data
ax = axes[1, 1]

times = monte_carlo_data.times[::250]
y_data = monte_carlo_data.exp_vals[::250]

ax.plot(times, y_data, "o")


# Plotting Lindblad Data
ax = axes[0, 1]

times = lindblad_data.times[::250]
y_data = lindblad_data.exp_vals[::250]

ax.plot(times, y_data, "o")


# Plotting the results of Stochastic Master Equation
ax = axes[1, 0]

course_grain = 60

coursed_times = sme_data.times[::course_grain]
measurements = np.squeeze(sme_data.measurements)

y_I, y_Q = measurements[:, 0], measurements[:, 1]
y_I = y_I.reshape((course_grain, -1)).mean(axis=0)
y_Q = y_Q.reshape((course_grain, -1)).mean(axis=0)
y_data = np.sqrt(y_I**2 + y_Q**2)

ax.plot(coursed_times, y_data, "o")


# Experimental Data
ax = axes[0, 0]
x_data = experimental_data.adc_timestamp
y_data_I = experimental_data.readout__final__adc_I__avg.sel(sweep_0=0)
y_err_I = experimental_data.readout__final__adc_I__avg__error.sel(sweep_0=0)
y_data_Q = experimental_data.readout__final__adc_Q__avg.sel(sweep_0=0)
y_err_Q = experimental_data.readout__final__adc_Q__avg__error.sel(sweep_0=0)


y_data_I -= y_data_I.mean()
y_data_Q -= y_data_Q.mean()


y_data = np.sqrt(y_data_I**2 + y_data_Q**2)
y_err = np.sqrt(y_err_I**2 * y_data_I**2 + y_err_Q**2 * y_data_Q**2) / y_data

x_data = x_data.coarsen(adc_timestamp=10).mean()
y_data = y_data.coarsen(adc_timestamp=10).mean()
y_err = y_err.coarsen(adc_timestamp=10).mean()

title = "Resonator Decay Rate"
xlabel = "Time (ns)"
scale_x = 1e-9

ylabel = "Readout Signal Abs (mV)"
scale_y = 1e-3

ax.plot(x_data / scale_x, y_data / scale_y, "o", label="$|I + iQ|$")

ax.errorbar(
    x_data / scale_x,
    y_data / scale_y,
    yerr=y_err / scale_y,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)

# .set(
#     title=title,
#     ylabel="Readout Signal (mV)",
# )

# ax.set(
#     xlabel=xlabel,
#     ylabel=ylabel,
# )

fig.align_ylabels(axes)

ax.legend()

# fig.savefig(f"../Figures/{title}.pdf")
