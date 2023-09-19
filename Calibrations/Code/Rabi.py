# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/standard_plot_style.mplstyle")

data_folder = "../Data/Qubit_spectroscopy"
title = "Qubit spectroscopy"
xlabel = "Frequency (GHz)"
scale_x = 1e9

ylabel = "Readout Signal I (a. u.)"
scale_y = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.pulse_frequency
y_data = data.readout__final__I__avg
y_err = data.readout__final__I__avg__error

fit_name = "Lorentzian"
fit_resolution = 1000


def fit_func(x, f0, A, gamma, offset):
    return offset + A * gamma**2 / (gamma**2 + (x - f0) ** 2)


# Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares

ls = LeastSquares(x_data, y_data, y_err, model=fit_func)
minimizer = Minuit(ls, f0=5.5e9, A=0.1, gamma=0.1e9, offset=0.1)
minimizer.migrad()


# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(x_data / scale_x, y_data / scale_y, "o", label="Data")

xs_fit = np.linspace(*ax.get_xlim(), fit_resolution) * scale_x
ax.plot(
    xs_fit / scale_x,
    fit_func(xs_fit, *minimizer.values) / scale_y,
    "--",
    color="k",
    label=fit_name + " fit",
)

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
    xlabel=xlabel,
    ylabel=ylabel,
)

ax.legend()

fig.savefig("../Figures/Qubit_spectroscopy.pdf")
