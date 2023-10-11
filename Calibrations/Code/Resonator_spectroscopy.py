# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

data_folder = "../Data/Resonator_spectroscopy"
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


def fit_func_ground(x, f0, A0, gamma0, offset):
    return offset + A0 * gamma0**2 / (gamma0**2 + (x - f0) ** 2)


guesses_ground = {
    "f0": 7.556e9,
    "A0": -2.5e-3,
    "gamma0": 0.0006e9,
    "offset": 0.00375,
}
# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


lims_ground = [7.55525e9, 7.55725e9]
mask_ground = np.logical_and(x_data > lims_ground[0], x_data < lims_ground[1])


ls = LeastSquares(
    x_data[mask_ground],
    y_data_g[mask_ground],
    y_err_g[mask_ground],
    model=fit_func_ground,
)
minimizer_ground = Minuit(ls, **guesses_ground)
minimizer_ground.migrad()
minimizer_ground.visualize()

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
    "offset": 0.0035,
    "f1": 7.556e9,
    "A1": -0.4e-3,
    "gamma1": 0.001e9,
}

lims_excited = [7.553e9, 7.5575e9]
mask_excited = np.logical_and(x_data > lims_excited[0], x_data < lims_excited[1])

# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(
    x_data[mask_excited],
    y_data_e[mask_excited],
    y_err_e[mask_excited],
    model=fit_func_excited,
)
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
            f"{name} = {minimizer_ground.values[name]:.6e} +- {minimizer_ground.errors[name]:.6e}",
            file=f,
        )
    print("Excited state", file=f)
    print(
        f"chi-squared: {minimizer_excited.fval:.2f} for {len(x_data) - len(guesses_excited)} dof with p-value {pval_excited:.3f}",
        file=f,
    )
    for name in guesses_excited:
        print(
            f"{name} = {minimizer_excited.values[name]:.6e} +- {minimizer_excited.errors[name]:.6e}",
            file=f,
        )


# # Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(x_data / scale_x, y_data_g / scale_y, "o", label="Ground Data", color="C0")

ax.plot(x_data / scale_x, y_data_e / scale_y, "o", label="Excited Data", color="C1")

xs_fit_ground = np.linspace(*lims_ground, fit_resolution)
ax.plot(
    xs_fit_ground / scale_x,
    fit_func_ground(xs_fit_ground, *minimizer_ground.values) / scale_y,
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


xs_fit_excited = np.linspace(*lims_excited, fit_resolution)
ax.plot(
    xs_fit_excited / scale_x,
    fit_func_excited(xs_fit_excited, *minimizer_excited.values) / scale_y,
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

fig.savefig(f"../Figures/{title}.pdf")
