# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/margin_figure.mplstyle")

data_folder = "../Data/efficiency_pulse"
title = "Efficiency Experiment Pulse"
xlabel = "Time (ns)"
scale_x = 1e-9

ylabel = "Readout Signal Abs (mV)"
scale_y = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.adc_timestamp
y_data_I = data.readout__final__adc_I__avg.sel(sweep_0=0)
y_err_I = data.readout__final__adc_I__avg__error.sel(sweep_0=0)
y_data_Q = data.readout__final__adc_Q__avg.sel(sweep_0=0)
y_err_Q = data.readout__final__adc_Q__avg__error.sel(sweep_0=0)

y_data_I -= y_data_I.mean()
y_data_Q -= y_data_Q.mean()

y_data = np.sqrt(y_data_I**2 + y_data_Q**2)
y_err = np.sqrt(y_err_I**2 * y_data_I**2 + y_err_Q**2 * y_data_Q**2) / y_data

x_data = x_data.coarsen(adc_timestamp=4).mean()
y_data = y_data.coarsen(adc_timestamp=4).mean()
y_err = y_err.coarsen(adc_timestamp=4).mean() / np.sqrt(4)

fit_name = "Exponential Decay"
fit_resolution = 1000
fit_delay = 400

mask = x_data > 660e-9


def fit_func(x, offset, Amplitude, kappa):
    return offset + Amplitude * np.exp(-x * kappa)


guesses = {
    "Amplitude": 0.010,
    "offset": 0.02,
    "kappa": 10e6,
}

# Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(x_data[mask], y_data[mask], y_err[mask], model=fit_func)
minimizer = Minuit(ls, **guesses)
# minimizer.interactive()
minimizer.migrad()

pval = chi2.sf(minimizer.fval, minimizer.ndof)


# exec(open("log_and_plot/code_to_run.txt").read())


# Overwrite plotting
fig, ax = plt.subplots()
ax.plot(x_data / scale_x, y_data / scale_y, "-", label="$|I + iQ|$")

xs_fit = np.linspace(*ax.get_xlim(), fit_resolution) * scale_x
xs_fit = xs_fit[fit_delay:]
# ax.plot(
#     xs_fit / scale_x,
#     fit_func(xs_fit, *minimizer.values) / scale_y,
#     "-",
#     color="k",
#     label=fit_name + " Fit",
# )

ax.errorbar(
    x_data / scale_x,
    y_data / scale_y,
    yerr=y_err / scale_y,
    ls="-",
    # marker = ".",
    color="C0",
    capsize=2,
    elinewidth=1,
)

# axes[0].set(
#     title=title,
#     ylabel="Readout Signal (mV)",
# )

ax.set(
    title=title,
    xlabel=xlabel,
    ylabel=ylabel,
)

# fig.align_ylabels(axes)

ax.legend(fontsize=22)

fig.savefig(f"../Figures/{title}.pdf", bbox_inches="tight")
