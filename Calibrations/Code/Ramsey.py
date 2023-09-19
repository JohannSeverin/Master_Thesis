# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/standard_plot_style.mplstyle")

data_folder = "../Data/Ramsey"
title = "Ramsey"
xlabel = "Waiting Time (ns)"
scale_x = 1e-9

ylabel = "Readout Signal I (a. u.)"
scale_y = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.wait_time
y_data = data.readout__final__I__avg
y_err = data.readout__final__I__avg__error

fit_name = "Decayingng Cosine"
fit_resolution = 1000
fit_delay = 40


def fit_func(x, Amplitude, Frequency, Phase, offset, T2):
    return offset + Amplitude * np.cos(2 * np.pi * Frequency * x + Phase) * np.exp(
        -x / T2
    )


guesses = {
    "Amplitude": 0.0001285200521324053,
    "Frequency": 5511022.044088177,
    "Phase": 0.1,
    "offset": 0.0001630993315741798,
    "T2": 10e-6,
}

# Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares

ls = LeastSquares(x_data, y_data, y_err, model=fit_func)
minimizer = Minuit(ls, **guesses)
minimizer.migrad()


# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(x_data / scale_x, y_data / scale_y, "o", label="Data")

xs_fit = np.linspace(*ax.get_xlim(), fit_resolution) * scale_x
xs_fit = xs_fit[fit_delay:]
ax.plot(
    xs_fit / scale_x,
    fit_func(xs_fit, *minimizer.values) / scale_y,
    "--",
    color="k",
    label=fit_name + " Fit",
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

fig.savefig(f"../Figures/{title}.pdf")
