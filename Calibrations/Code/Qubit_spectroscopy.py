# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

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


guesses = {"f0": 5.5e9, "A": 0.1, "gamma": 0.1e9, "offset": 0.1}

# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(x_data, y_data, y_err, model=fit_func)
minimizer = Minuit(ls, **guesses)
minimizer.migrad()

pval = chi2.sf(minimizer.fval, len(x_data) - len(guesses))

# Priting
with open(f"../Fit_log/{title}.txt", "w") as f:
    print(
        f"chi-squared: {minimizer.fval:.2f} for {len(x_data) - len(guesses)} dof with p-value {pval:.3f}",
        file=f,
    )
    for name in guesses:
        print(
            f"{name} = {minimizer.values[name]:.2e} +- {minimizer.errors[name]:.2e}",
            file=f,
        )


# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(x_data / scale_x, y_data / scale_y, "o", label="Data")

xs_fit = np.linspace(*ax.get_xlim(), fit_resolution) * scale_x
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

fig.savefig("../Figures/Qubit_spectroscopy.pdf")
