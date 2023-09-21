# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

data_folder = "../Data/Qubit_anharmonicity"
title = "Qubit Anharmonicity"
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
fit_delay = 0


def fit_func(x, f01, f02, A1, A2, gamma1, gamma2, offset):
    return (
        offset
        + A1 * gamma1**2 / (gamma1**2 + (x - f01) ** 2)
        + A2 * gamma2**2 / (gamma2**2 + (x - f02) ** 2)
    )


guesses = {
    "f01": 5.98e9,
    "A1": 0.3e-3,
    "gamma1": 0.001e9,
    "offset": -0.3e-3,
    "f02": 5.84e9,
    "A2": 0.3e-3,
    "gamma2": 0.001e9,
}

# Fitting Fitting
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

ls = LeastSquares(x_data, y_data, y_err, model=fit_func)
minimizer = Minuit(ls, **guesses)

minimizer.interactive()

minimizer.migrad()

pval = chi2.sf(minimizer.fval, len(x_data) - len(guesses))

exec(open("log_and_plot/code_to_run.txt").read())

anharmonicty = 2 * minimizer.values["f02"] - 2 * minimizer.values["f01"]
anharmonicty_err = 2 * np.sqrt(
    minimizer.errors["f02"] ** 2 + minimizer.errors["f01"] ** 2
)

with open(f"../Fit_log/{title}.txt", "a") as f:
    print(
        f"anharmonicty = {anharmonicty:.5e} +- {anharmonicty_err:.5e}",
        file=f,
    )


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

# fig.savefig("../Figures/{title}.pdf")
