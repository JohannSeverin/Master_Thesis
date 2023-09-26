# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

data_folder = "../Data/T1"
title = "T1"
xlabel = "Waiting Time (µs)"
scale_x = 1e-6

ylabel = "Readout Signal I (a. u.)"
scale_y = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.readout_delay
y_data = data.readout__final__I__avg
y_err = data.readout__final__I__avg__error

fit_name = "Exponential Decay"
fit_resolution = 1000
fit_delay = 40


def fit_func(x, offset, Amplitude, T1):
    return offset + Amplitude * np.exp(-x / T1)


guesses = {
    "Amplitude": 0.0001630993315741798,
    "offset": -0.0001630993315741798,
    "T1": 10e-6,
}

# Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(x_data, y_data, y_err, model=fit_func)
minimizer = Minuit(ls, **guesses)
minimizer.migrad()

pval = chi2.sf(minimizer.fval, len(x_data) - len(guesses))


exec(open("log_and_plot/code_to_run.txt").read())

# # Priting
# with open(f"../Fit_log/{title}.txt", "w") as f:
#     print(
#         f"chi-squared: {minimizer.fval:.2f} for {len(x_data) - len(guesses)} dof with p-value {pval:.3f}",
#         file=f,
#     )
#     for name in guesses:
#         print(
#             f"{name} = {minimizer.values[name]:.5e} +- {minimizer.errors[name]:.5e}",
#             file=f,
#         )

# # Plotting
# fig, ax = plt.subplots(1, 1)

# ax.plot(x_data / scale_x, y_data / scale_y, "o", label="Data")

# xs_fit = np.linspace(*ax.get_xlim(), fit_resolution) * scale_x
# xs_fit = xs_fit[fit_delay:]
# ax.plot(
#     xs_fit / scale_x,
#     fit_func(xs_fit, *minimizer.values) / scale_y,
#     "--",
#     color="k",
#     label=fit_name + " Fit",
# )

# ax.errorbar(
#     x_data / scale_x,
#     y_data / scale_y,
#     yerr=y_err / scale_y,
#     ls="none",
#     color="C0",
#     capsize=2,
#     elinewidth=1,
# )

# ax.set(
#     title=title,
#     xlabel=xlabel,
#     ylabel=ylabel,
# )

# ax.legend()

# fig.savefig(f"../Figures/{title}.pdf")