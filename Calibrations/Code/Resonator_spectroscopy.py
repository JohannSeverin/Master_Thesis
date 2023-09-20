# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

data_folder = "../Data/resonator_spectroscopy"
title = "Resonator Spectroscopy"
xlabel = "Frequency (GHz)"
scale_x = 1e9

ylabel = "Readout Signal I (a. u.)"
scale_y = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.readout_frequency

y_data_g = data.sel({"sweep_1": 0}).readout__final__abs__avg
y_err_g = data.sel({"sweep_1": 0}).readout__final__abs__avg__error

y_data_e = data.sel({"sweep_1": 1}).readout__final__abs__avg
y_err_e = data.sel({"sweep_1": 1}).readout__final__abs__avg__error


fit_name = "Lorentzian"
fit_resolution = 1000


def fit_func_ground(x, f0, A, gamma, offset):
    return offset + A * gamma**2 / (gamma**2 + (x - f0) ** 2)


guesses_ground = {"f0": 7.556e9, "A": -0.4e-3, "gamma": 0.001e9, "offset": 0.0008}

# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(x_data, y_data_g, y_err_g, model=fit_func_ground)
minimizer_ground = Minuit(ls, **guesses_ground)
minimizer_ground.migrad()

pval_ground = chi2.sf(minimizer_ground.fval, len(x_data) - len(guesses_ground))


def fit_func_excited(x, f0, A0, gamma0, f1, A1, gamma1, offset):
    return (
        offset
        + A0 * gamma0**2 / (gamma0**2 + (x - f0) ** 2)
        + A1 * gamma1**2 / (gamma1**2 + (x - f1) ** 2)
    )


guesses_excited = {
    "f0": 7.556e9,
    "A0": -0.4e-3,
    "gamma0": 0.001e9,
    "offset": 0.0008,
    "f1": 7.556e9,
    "A1": -0.4e-3,
    "gamma1": 0.001e9,
}

# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(x_data, y_data_e, y_err_e, model=fit_func_excited)
minimizer_excited = Minuit(ls, **guesses_excited)
minimizer_excited.migrad()

pval_excited = chi2.sf(minimizer_excited.fval, len(x_data) - len(guesses_excited))


# Priting
with open(f"../Fit_log/{title}.txt", "w") as f:
    print("Ground state", file=f)
    print(
        f"chi-squared: {minimizer_ground.fval:.2f} for {len(x_data) - len(guesses_ground)} dof with p-value {pval_ground:.3f}",
        file=f,
    )
    for name in guesses_ground:
        print(
            f"{name} = {minimizer_ground.values[name]:.5e} +- {minimizer_ground.errors[name]:.5e}",
            file=f,
        )
    print("Excited state", file=f)
    print(
        f"chi-squared: {minimizer_excited.fval:.2f} for {len(x_data) - len(guesses_excited)} dof with p-value {pval_excited:.3f}",
        file=f,
    )
    for name in guesses_excited:
        print(
            f"{name} = {minimizer_excited.values[name]:.5e} +- {minimizer_excited.errors[name]:.5e}",
            file=f,
        )


# # Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(x_data / scale_x, y_data_g / scale_y, "o", label="Ground Data", color="C0")

ax.plot(x_data / scale_x, y_data_e / scale_y, "o", label="Excited Data", color="C1")

xs_fit = np.linspace(*ax.get_xlim(), fit_resolution) * scale_x
ax.plot(
    xs_fit / scale_x,
    fit_func_ground(xs_fit, *minimizer_ground.values) / scale_y,
    "--",
    color="C0",
    label=fit_name + " Fit",
)

ax.errorbar(
    x_data / scale_x,
    y_data_g / scale_y,
    yerr=y_err_g / scale_y,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)


ax.plot(
    xs_fit / scale_x,
    fit_func_excited(xs_fit, *minimizer_excited.values) / scale_y,
    "--",
    color="C1",
    label="Two Lorentzian Fit",
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
    xlabel=xlabel,
    ylabel=ylabel,
)

ax.legend()

fig.savefig("../Figures/{title}.pdf")
